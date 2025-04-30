import json
import sys
import numpy as np
import evaluate
from tqdm import tqdm
import argparse 
from collections import defaultdict
from utils import save_jsonl, load_jsonl, merge_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import Dataset

class TSTRunner:
    def __init__(self, 
                 lang, 
                 subset, 
                 prompt_techs):
        self.lang = lang
        self.subset = subset
        self.prompt_techs = prompt_techs
        
        if lang != 'en':
            self.best_model_dir = f"../../data/{lang}/sc/best_model"
        else:
            if subset.endswith('paras'):
                self.best_model_dir = f"../../data/{lang}/sc/best_model_paras"
            else:
                self.best_model_dir = f"../../data/{lang}/sc/best_model_sents"

        if self.lang == 'en':
            self.model = 'distilbert-base-uncased'
        elif self.lang == 'pt':
            self.model = 'xlm-roberta-base'
        elif self.lang == 'vi':
            self.model = 'xlm-roberta-base'
        
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
    
    def style_transfer(self, mgt):

        tokenizer = AutoTokenizer.from_pretrained(self.best_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.best_model_dir)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = Trainer(model=model, 
                          data_collator=data_collator)
        
        def tokenize_function(batch):
            return tokenizer(batch['sents'], truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict({"sents": mgt})
        ds_tok = dataset.map(tokenize_function, batched=True)

        predictions = trainer.predict(ds_tok)
        preds = np.argmax(predictions.predictions, axis=-1).astype(float)
        
        return preds
    
    def evaluate_prompting_technique(self, prompting):
        
        if self.subset == 'paras':
            subset_path='_paras'
        else:
            subset_path=''
        mgt_file = f'../../data/{self.lang}/eval/{self.lang}_eval_{prompting}.jsonl'
        out_file_micro = f'../../data/{self.lang}/eval{subset_path}/metrics/{self.lang}_{prompting}_metrics_micro.jsonl'
        out_file_macro = f'../../data/{self.lang}/eval{subset_path}/metrics/{self.lang}_{prompting}_metrics_macro.jsonl'
        
        # Load data
        data = load_jsonl(mgt_file)
        print('Data size', len(data))

        mgts = []
        trgts = []
        ids = []
        
        for item in data:
            mgt_key = f"mgt_{prompting}"
            if mgt_key not in item or not item[mgt_key]:
                continue
            
            generated = item[mgt_key]
            reference = item['trgt']
            
            mgts.append(generated)
            trgts.append(reference)
            ids.append(item['id'])
            
        assert len(mgts) == len(trgts)
        
        print("Computing BLEU ...", flush=True)
        bleu_scores = [
                self.bleu.compute(predictions=[mgts[i]], references=[[trgts[i]]])["bleu"]
                for i in range(len(mgts))
            ]
        
        print("Computing ROUGE ...", flush=True)
        rouge_scores = self.rouge.compute(predictions=mgts, references=trgts, use_aggregator=False)
        
        print("Computing BERTScore ...", flush=True)
        bertscore_scores = self.bertscore.compute(predictions=mgts, references=trgts, model_type=self.model)["f1"]
        
        
        print("Computing Style Transfer ...", flush=True)
        preds = self.style_transfer(mgts)
        
        out = []
        for i in range(len(mgts)):
            item_metrics = {
                'id': ids[i],
                'bleu': float(bleu_scores[i]),
                'rouge1': float(rouge_scores['rouge1'][i]),
                'rouge2': float(rouge_scores['rouge2'][i]),
                'rougeL': float(rouge_scores['rougeL'][i]),
                'bertscore': float(bertscore_scores[i]),
                'style_transfer': float(preds[i])
            }
            out.append(item_metrics)
        
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore', 'style_transfer'] #'ppl'
        mean_scores = {
            metric: round(float(np.mean([item[metric] for item in out])), 4) if out else 0
            for metric in metrics
        }
        
        print("FINAL SCORES")
        for metric, score in mean_scores.items():
            print(f"{metric.upper()}: {score}")
        
        # Save results
        save_jsonl(out, out_file_micro)
        save_jsonl([mean_scores], out_file_macro)
    
    def run(self):
        """Run evaluation for all specified prompting techniques"""
        for prompting in self.prompt_techs:
            print(f'Eval prompting {prompting}')
            self.evaluate_prompting_technique(prompting)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--prompt_techs',type=str, nargs='+', required=True)
    
    args = parser.parse_args()
    
    evaluator = TSTRunner(args.lang, args.subset, args.prompt_techs)
    evaluator.run()


if __name__ == '__main__':
    main()