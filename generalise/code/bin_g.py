import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
import random
from utils import load_jsonl, save_jsonl
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# All code taken from: https://github.com/ahans30/Binoculars


print(f"\nNumber of available GPUs: {torch.cuda.device_count()}\n")
# Utils functions

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        #print(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


# Binoculars
torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b", # "tiiuae/falcon-7b" tiiuae/Falcon3-7B-Base
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct", # "tiiuae/falcon-7b-instruct"
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"],
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"],
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            #padding="longest" if batch_size > 1 else False,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()

        return observer_logits, performer_logits


    # def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
    #     print(input_text)
    #     batch = [input_text] if isinstance(input_text, str) else input_text
    #     encodings = self._tokenize(batch)
    #     observer_logits, performer_logits = self._get_logits(encodings)
    #     ppl = perplexity(encodings, performer_logits)
    #     x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
    #                     encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
    #     print('test', ppl, x_ppl)
    #     binoculars_scores = ppl / x_ppl
    #     binoculars_scores = binoculars_scores.tolist()
    #     return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def compute_score(self, input_text: Union[list[str], str], batch_size: int = 16) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        all_scores = []

        for i in tqdm(range(0, len(batch), batch_size)):
            sub_batch = batch[i:i+batch_size]
            encodings = self._tokenize(sub_batch)
            observer_logits, performer_logits = self._get_logits(encodings)

            ppl = perplexity(encodings, performer_logits)
            x_ppl = entropy(
                observer_logits.to(DEVICE_1),
                performer_logits.to(DEVICE_1),
                encodings.to(DEVICE_1),
                self.tokenizer.pad_token_id
            )

            binoculars_scores = ppl / x_ppl
            #print('Bin scores', binoculars_scores)
            all_scores.extend(binoculars_scores.tolist())

        return all_scores[0] if isinstance(input_text, str) else all_scores


    # Modify to output binary sores
    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        # pred = np.where(binoculars_scores < self.threshold,
        #                 "Most likely AI-generated",
        #                 "Most likely human-generated"
        #                 ).tolist()

        pred = np.where(binoculars_scores < self.threshold,
                        1,
                        0
                        ).tolist()
        return pred

def load_data(data_dir):
    random.seed(42)

    data = load_jsonl(data_dir)
    data = random.sample(data, 900)

    texts, labels = [], []
    for item in data:
        texts.append(item['trgt'])
        labels.append(0)
        texts.append(item['mgt'])
        labels.append(1)


    return texts, labels

def binoculars_detector(texts, labels):

    binoculars = Binoculars(mode = "accuracy")
    scores = binoculars.predict(texts)

    accuracy = accuracy_score(labels, scores)

    return accuracy


def main():
    data_dirs = [('generalise/data/ds/our/gpt_multi_task_en_gpt.jsonl', 'Our'),
                ('generalise/data/ds/external/mgt/wiki_en_gpt.jsonl', 'Wiki Unc.')]
    
    results = {'binoculars': {}}

    for data_dir, data_name in data_dirs:
        texts, labels = load_data(data_dir)

        acc_binc = binoculars_detector(texts, labels)
        results['binoculars'][data_name] = acc_binc


    save_jsonl([results], "generalise/data/detect/ex1_ots_bin.jsonl")
    print(results)
    
if __name__ == "__main__":
    main()



