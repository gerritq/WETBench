import json
import numpy as np
import evaluate
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
from utils import save_jsonl, load_jsonl

class TextEvaluator:
    def __init__(self, 
                 lang,
                 in_file,
                 out_file,
                 prompt_techs,
                 total_n):
        self.lang = lang
        self.in_file = in_file
        self.out_file = out_file
        self.prompt_techs = prompt_techs
        self.total_n = total_n
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore", cache_dir='/scratch_tmp/prj/inf_nlg_ai_detection/.cache')
        if lang == 'en':
            self.model = 'distilbert-base-uncased'
        else:
            self.model = 'xlm-roberta-base'

    def load_data(self, in_file):
        data = load_jsonl(in_file)
        n_per_tertile = self.total_n // 3

        data_by_tertiles = {'low': [], 'medium': [], 'high': []}

        for item in data:
            data_by_tertiles[item['word_tertile']].append(item)

        return (data_by_tertiles['low'][:n_per_tertile] + 
                data_by_tertiles['medium'][:n_per_tertile] + 
                data_by_tertiles['high'][:n_per_tertile])

    def evaluate(self):
        
        for prompt_tech in self.prompt_techs:
            print(f'\nEval for LANG {self.lang} PROMPT {prompt_tech} in {self.prompt_techs}', flush=True)
            
            in_file_mod = self.in_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            out_file_mod = self.out_file.replace('.jsonl', f'_brb_{prompt_tech}.jsonl')
            data = self.load_data(in_file_mod)

            generated_texts, reference_texts, ids, word_tertiles = [], [], [], []
            for item in data:                
                generated_texts.append(item[f"mgt_{prompt_tech}"])
                reference_texts.append(item['trgt'])
                ids.append(item['id'])
                word_tertiles.append(item['word_tertile'])

            assert len(generated_texts) == len(reference_texts)
            
            print("Computing BLEU ...", flush=True)
            bleu_scores = [
                self.bleu.compute(predictions=[generated_texts[i]], references=[[reference_texts[i]]])["bleu"]
                for i in range(len(generated_texts))
            ]
            
            print("Computing ROUGE ...", flush=True)
            rouge_scores = self.rouge.compute(predictions=generated_texts, references=reference_texts, use_aggregator=False)
            
            print("Computing BERTScore ...", flush=True)
            bertscore_scores = self.bertscore.compute(predictions=generated_texts, references=reference_texts, model_type=self.model)["f1"]
            
            out = []
            scores_by_tertile = defaultdict(lambda: defaultdict(list))
            for i in range(len(generated_texts)):
                item_metrics = {
                    'id': ids[i],
                    'word_tertile': word_tertiles[i],
                    'bleu': bleu_scores[i],
                    'rouge1': rouge_scores['rouge1'][i],
                    'rouge2': rouge_scores['rouge2'][i],
                    'rougeL': rouge_scores['rougeL'][i],
                    'bertscore': bertscore_scores[i]
                }
                out.append(item_metrics)
                
                for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore']:
                    scores_by_tertile[word_tertiles[i]][metric].append(item_metrics[metric])
                    scores_by_tertile["overall"][metric].append(item_metrics[metric])
            
            final_scores = {
                metric: {
                    tertile: round(np.mean(scores_by_tertile[tertile][metric]), 4) if scores_by_tertile[tertile][metric] else 0
                    for tertile in ["low", "medium", "high", "overall"]
                }
                for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore']
            }
            
            print("FINAL SCORE")
            for metric, values in final_scores.items():
                print(f"\n{metric.upper()} Scores:")
                for tertile, score in values.items():
                    print(f"  {tertile.capitalize()}: {score}")
            
            save_jsonl([final_scores], out_file_mod)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--prompt_techs', type=str, nargs='+', required=True)
    parser.add_argument('--total_n', type=int, required=True)
    args = parser.parse_args()

    evaluator = TextEvaluator(args.lang, args.in_file, args.out_file, args.prompt_techs, args.total_n)
    evaluator.evaluate()
