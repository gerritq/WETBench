import torch
import json
import argparse
import time
import numpy as np
import os
from collections import defaultdict
from qafacteval import QAFactEval
from utils import load_jsonl, save_jsonl

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

print(torch.__version__)
torch.cuda.empty_cache()

class QAFactEvalRunner:
    def __init__(self, 
                 lang,
                 ds,
                 in_file,
                 out_file,
                 prompt_techs):
        self.ds = ds
        self.lang = lang
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
        
        self.kwargs = {
            "cuda_device": 0,
            "use_lerc_quip": True,
            "verbose": True,
            "generation_batch_size": 16,
            "answering_batch_size": 16,
            "lerc_batch_size": 16
        }
        
        self.model_folder = '/scratch_tmp/users/k21157437/QAFactEval/models'
        self.QAmetric = QAFactEval(
            lerc_quip_path=f"{self.model_folder}/quip-512-mocha",
            generation_model_path=f"{self.model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{self.model_folder}/answering",
            lerc_model_path=f"{self.model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{self.model_folder}/lerc/pretraining.tar.gz",
            **self.kwargs
        )

    def truncate_text(self, text, max_tokens=800):
        '''We ran into an issue where the token length is too long; in qaeval this is not handled in the tokenizer..'''
        tokens = text.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return text

        
    def run_evaluation(self):
        print('Start ... ', flush=True)
        start = time.time()
    
        trans = '_trans' if self.lang != 'en' else ''
        
        for prompt_tech in self.prompt_techs:
            print('\n----------------', flush=True)
            print('----------------', flush=True)
            print(f"Running eval for {self.lang} {prompt_tech} in {self.prompt_techs}", flush=True)

            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            out_file_macro = self.out_file.replace('.jsonl', f'_qa_{prompt_tech}_macro.jsonl')
            out_file_micro = self.out_file.replace('.jsonl', f'_qa_{prompt_tech}_micro.jsonl')

            data = load_jsonl(in_file_mod)
            #data = data[2:3]

            mgts, refs = [], []
            for item in data:

                if not item[f'mgt_{prompt_tech}{trans}']:
                    print('Failure ..')
                    continue

                mgts.append([self.truncate_text(item[f'mgt_{prompt_tech}{trans}'])])
                
                if self.ds == 'sums':
                    ref = self.truncate_text(item[f'src_inf{trans}'])
                    if not ref:
                        print('Failure sums ...')
                        continue
                    refs.append(ref) # can be src alone or src_info
                else:
                    ref = self.truncate_text(item[f'trgt{trans}'])
                    if not ref:
                        print('Failure paras ...')
                        continue
                    refs.append(ref) # trgt in extend is second halve; in first is total paragraph


            # print(mgts)
            # print(refs)
            
            qafacteval_results = self.QAmetric.score_batch_qafacteval(refs, mgts, return_qa_pairs=True)
            qafacteval_scores = [result[0]['qa-eval'] for result in qafacteval_results]
            qafacteval_qa_pairs = [result[1] for result in qafacteval_results]
            
            scores_by_tertile = defaultdict(lambda: defaultdict(list))
            out = []    
            
            for i, item in enumerate(data):
                item_new = item.copy()
                item_new.update({'prompting': prompt_tech, 
                                 "qafact": qafacteval_scores[i],
                                 "qafact_qa_pairs": qafacteval_qa_pairs[i]})
                out.append(item_new)

                for metric, value in qafacteval_scores[i].items():
                    scores_by_tertile[item_new["word_tertile"]][metric].append(value)
                    scores_by_tertile["overall"][metric].append(value)
            
            mean_scores_by_tertile = {
                tertile: {metric: np.mean(values) for metric, values in metrics.items()}
                for tertile, metrics in scores_by_tertile.items()
            }
            
            save_jsonl(out, out_file_micro)
            save_jsonl([mean_scores_by_tertile], out_file_macro)
        
        print(f"Evaluation completed in {time.time() - start:.2f} seconds", flush=True)

if __name__ == "__main__":
    print(torch.cuda.is_available(), flush=True)
    print(torch.cuda.device_count(), flush=True)
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--prompt_techs', type=str, nargs='+', required=True)
    parser.add_argument('--ds', type=str, required=True)
    args = parser.parse_args()
    
    evaluator = QAFactEvalRunner(args.lang, args.ds, args.in_file, args.out_file, args.prompt_techs)
    evaluator.run_evaluation()
