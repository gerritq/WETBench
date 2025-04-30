import json
import numpy as np
import evaluate
from tqdm import tqdm
import sys
from collections import defaultdict
from utils import save_jsonl, merge_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import Dataset

lang = sys.argv[1]
task = sys.argv[2]
subset = sys.argv[3]
prompting = sys.argv[4]

in_file = f'../../data/{lang}/{task}/4_{lang}_{task}_{subset}.jsonl'
mgt_file = f'../../data/{lang}/{task}/mgt/{lang}_{task}_{subset}_{prompting}.jsonl'
out_file = f'../../data/{lang}/{task}/metrics/{subset}/{lang}_{task}_{subset}_{prompting}_brb.jsonl' #bleu, rouge, bertscore

if lang != 'en':
    best_model_dir = f'../../data/en/sc/best_model'
else: 
    if subset.endswith('paras'):
        best_model_dir = f'../../data/en/sc/best_model_paras'
    else:
        best_model_dir = f'../../data/en/sc/best_model_sents'

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
ppl = evaluate.load("perplexity", module_type="metric")

if lang == 'en':
    model = 'distilbert-base-uncased'
if lang == 'pt':
    model = 'FacebookAI/xlm-roberta-base'
if lang == 'vi':
    model = 'FacebookAI/xlm-roberta-base'


def style_transfer(best_model_dir, mgt):
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(model=model, 
                      data_collator=data_collator)
    
    def tokenize_function(batch):
        return tokenizer(batch['sents'], truncation=True, padding=True, max_length=512)
    
    dataset = Dataset.from_dict({"sents": mgt})
    ds_tok = dataset.map(tokenize_function, batched=True)
    
    # Make predictions
    predictions = trainer.predict(ds_tok)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    # return np.mean(preds == 1)
    return preds




def main():
    data = merge_jsonl(mgt_file, in_file)
    
    generated_texts = []
    reference_texts = []
    ids = []
    word_tertiles = []
    
    for item in data:
        if not item[f"mgt_{prompting}"]:
            continue
        
        generated = item[f"mgt_{prompting}"]
        reference = item['p_second'] if subset == 'extend' else item["p_clean"]
        
        generated_texts.append(generated)
        reference_texts.append(reference)
        ids.append(item['id'])
        word_tertiles.append(item['word_tertile'])

    assert len(generated_texts) == len(reference_texts)

    # Compute scores in batch
    print("Computing BLEU ...", flush=True)
    bleu_scores = bleu.compute(predictions=generated_texts, references=[[ref] for ref in reference_texts])["bleu"]
    print("Computing ROUGE ...", flush=True)
    rouge_scores = rouge.compute(predictions=generated_texts, references=reference_texts, use_aggregator=False)
    print("Computing BERTScore ...", flush=True)
    bertscore_scores = bertscore.compute(predictions=generated_texts, references=reference_texts, model_type=model)["f1"]
    print("Computing PPL ...", flush=True)
    ppl_scores = ppl.compute(model_id='gpt2', predictions=generated_texts)
    print("Computing Style Transfer ...", flush=True)
    preds = style_transfer('best_model', generated_texts)

    # print(rouge_scores)
    out = []
    scores_by_tertile = defaultdict(lambda: defaultdict(list))
    for i in range(len(generated_texts)):
        item_metrics = {
            'id': ids[i],
            'word_tertile': word_tertiles[i],
            'bleu': bleu_scores,
            'rouge1': rouge_scores['rouge1'][i],
            'rouge2': rouge_scores['rouge2'][i],
            'rougeL': rouge_scores['rougeL'][i],
            'bertscore': bertscore_scores[i],
            'ppl': ppl_scores['perplexities'][i],
            'style_transfer': preds[i]
        }
        out.append(item_metrics)
        
        for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore', 'ppl', 'style_transfer']:
            scores_by_tertile[word_tertiles[i]][metric].append(item_metrics[metric])
            scores_by_tertile["overall"][metric].append(item_metrics[metric])
    final_scores = {
        metric: {
            tertile: round(np.mean(scores_by_tertile[tertile][metric]), 4) if scores_by_tertile[tertile][metric] else 0
            for tertile in ["low", "medium", "high", "overall"]
        }
        for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore', 'ppl', 'style_transfer']
    }
    
    print("FINAL SCORE")
    for metric, values in final_scores.items():
        print(f"\n{metric.upper()} Scores:")
        for tertile, score in values.items():
            print(f"  {tertile.capitalize()}: {score}")
    
    save_jsonl([final_scores], out_file)

if __name__ == '__main__':
    main()
