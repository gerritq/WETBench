import os
import json
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
import nltk
import time
from tqdm import tqdm
import sys
import spacy
from collections import Counter
import random
import regex as rex
from underthesea import ner

def get_model(lang_code):
    if lang_code == 'en':
        return spacy.load('en_core_web_sm')
    elif lang_code == 'pt':
        return spacy.load('pt_core_news_sm')
    elif lang_code == 'vi':
        return spacy.load('xx_ent_wiki_sm') # not optimal but the only one that will run for a small sample
        #return "underthesea_vi"


def extract_entities(text, nlp):
    if nlp == "underthesea_vi":
        
        ners = ner(text) # no idea about txt len
        ners = [x for x in ners if x[3] != 'O'] # all but 'O' are proper NER -- confirmed with their documentation
        ners = [(x[0].replace("\n", ""), *x[1:]) for x in ners]
        return ners
    else:
        text = text[:nlp.max_length - 1000]
        doc = nlp(text)
        return [(ent.text.replace('\n', ''), ent.label_) for ent in doc.ents]

def calculate_entity_overlap(text, infobox, summary, nlp):
    
    text_entities = extract_entities(text, nlp)
    infobox_entities = extract_entities(infobox, nlp) if infobox else []
    summary_entities = extract_entities(summary, nlp)

    txt_entity_set = set(entity[0] for entity in text_entities)
    infobox_entity_set = set(entity[0] for entity in infobox_entities)
    summary_entity_set = set(entity[0] for entity in summary_entities)
    # txt + infobox
    body_entities = txt_entity_set.union(infobox_entity_set)

    # entities in bth
    common_entities = summary_entity_set.intersection(body_entities)

    precision = len(common_entities) / len(summary_entity_set) if summary_entity_set else 0
    recall = len(common_entities) / len(body_entities) if body_entities else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
    return {
        'entity_precision': precision,
        'entity_recall': recall,
        'entity_f1': f1,
        'summary_entity_count': len(summary_entities),
        'body_entity_count': len(body_entities),
        'infobox_entity_count': len(infobox_entities),
        }

def load_wikis_sums(ds_name):
    data = []
    #wiki_file = f"../../data/{ds_name[:2]}/ds/{ds_name}.jsonl"
    raw_file = f"/scratch_tmp/users/k21157437/aid/data/{ds_name[:2]}/3_{ds_name[:2]}_text.jsonl"
    
    # need to rm citations in our text
    refs_PATTERN = r'\[sup:.*?:sup\]'

    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
        
            # create src
            body = ""
            for section in entry["sections"]:
                for title, paragraphs in section.items():
                    body += f"{title}\n"
                    body += '\n\n'.join(paragraphs) + '\n\n'
            body = rex.sub(refs_PATTERN, '', body).strip()    
            entry['src'] = body

            # trgt
            lead = '\n\n'.join(entry['lead'])
            lead = rex.sub(refs_PATTERN, '', lead).strip()  
            entry['trgt'] = lead
            
            # infobox
            infobox = '\n\n'.join(entry['infobox']).replace('===', '') if entry['infobox'] else ''
            infobox = rex.sub(refs_PATTERN, '', infobox).strip() 
            entry['infobox'] = infobox

            # if entry["infobox"]:
            #     entry["body"] = entry["infobox"] + "\n\n" + entry["body"]
            data.append({k: v for k, v in entry.items() if k in ['id', 'title', 'trgt', 'src', 'infobox']})
    return data

def calculate_novel_ngrams(reference_tokens, summary_tokens, n):
    reference_ngrams = set(nltk.ngrams(reference_tokens, n))
    summary_ngrams = set(nltk.ngrams(summary_tokens, n))
    novel_ngrams = summary_ngrams - reference_ngrams

    return len(novel_ngrams) / len(summary_ngrams) if summary_ngrams else 0

def calculate_metrics(scorer, text, summary, table, nlp=None):

    rouge_scores = scorer.score(text, summary)
    
    # tok
    text_tokens = text.lower().split()
    summary_tokens = summary.lower().split()
    table_tokens = table.lower().split() if table else [] # case where table is None or ''
    
    # tok lens
    text_len = len(text_tokens)
    summary_len = len(summary_tokens)
    table_len = len(table_tokens) if table_tokens else None
    
    # com ratio
    compression_ratio = text_len / summary_len if summary_len > 0 else 0
    
    # novel n-grams
    novel_unigram_share = calculate_novel_ngrams(text_tokens, summary_tokens, 1)
    novel_bigram_share = calculate_novel_ngrams(text_tokens, summary_tokens, 2)
    novel_trigram_share = calculate_novel_ngrams(text_tokens, summary_tokens, 3)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rouge3': rouge_scores['rouge3'].fmeasure,
        'text_len': text_len,
        'summary_len': summary_len,
        'table_len': table_len,
        'compression_ratio': compression_ratio,
        'novel_unigram_share': novel_unigram_share,
        'novel_bigram_share': novel_bigram_share,
        'novel_trigram_share': novel_trigram_share
    }

def process_dataset(dataset_name, scorer):
    lang_code = 'en'# default for others
    if dataset_name.endswith('sums'):
        lang_code = dataset_name[:2]  # 'en', 'pt', or 'vi'
    
    nlp = get_model(lang_code)

    # Load dataset
    if dataset_name.endswith('sums'):
        dataset = load_wikis_sums(dataset_name)
        text_col = 'src'
        summary_col = 'trgt'
        table_col = 'infobox'
    else:
        if dataset_name == 'abisee/cnn_dailymail':
            subset = "3.0.0"
            text_col = 'article'
            summary_col = 'highlights'
            table_col = ''
        elif dataset_name == 'GEM/wiki_lingua':
            subset = "en"
            text_col = 'source'
            summary_col = 'target'
            table_col = ''
        elif dataset_name == 'ccdv/arxiv-summarization':
            subset = "document"
            text_col = 'article'
            summary_col = 'abstract'
            table_col = ''
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # load hf data
        dataset = load_dataset(dataset_name, 
                               subset, 
                               cache_dir='/scratch_tmp/prj/inf_nlg_ai_detection/.cache', 
                               trust_remote_code=True)
        # retrieve all splits in one ds
        print('Flattens data', flush=True)
        dataset = [item for split in dataset.values() for item in split]
    
    print(f'{dataset_name} original N: {len(dataset)}', flush=True)
    
    # Can only do random subset of entities 
    n_ent_samples = 20000
    random.seed(42)
    random.shuffle(dataset)
    
    basic_metrics_list = []
    entity_metrics_list = []
    
    for idx, item in enumerate(tqdm(dataset, file=sys.stdout)):
        # print('')
        # print(item['title'])
        # print(item['infobox'])
        # print(item['src'])
        # print(item['trgt'])
        basic_metrics = calculate_metrics(scorer, 
                                          item[text_col], 
                                          item[summary_col], 
                                          item[table_col] if table_col else '')
        basic_metrics_list.append(basic_metrics)
        
        # get metrics for first random n_ent_samples
        if idx < n_ent_samples:
            entity_metrics = calculate_entity_overlap(
                item[text_col], 
                item[table_col] if table_col else '', 
                item[summary_col],
                nlp
            )
            entity_metrics_list.append(entity_metrics)
    
    basic_df = pd.DataFrame(basic_metrics_list)
    entity_df = pd.DataFrame(entity_metrics_list) if entity_metrics_list else None
    
    agg_metrics = {
        'Dataset': dataset_name,
        'Size': len(dataset),
        'ROUGE-1': basic_df['rouge1'].mean(),
        'ROUGE-2': basic_df['rouge2'].mean(),
        'ROUGE-3': basic_df['rouge3'].mean(),
        'Body Length': basic_df['text_len'].mean(),
        'Body Std.': basic_df['text_len'].std(),
        'Summary Length': basic_df['summary_len'].mean(),
        'Summary Std.': basic_df['summary_len'].std(),
        'Infobox Length': basic_df['table_len'].mean() if 'table_len' in basic_df else None,
        'Infobox Std.': basic_df['table_len'].std() if 'table_len' in basic_df else None,
        'Compression Rate': basic_df['compression_ratio'].mean(),
        'Novel Unigram Share': basic_df['novel_unigram_share'].mean(),
        'Novel Bigram Share': basic_df['novel_bigram_share'].mean(),
        'Novel Trigram Share': basic_df['novel_trigram_share'].mean()
    }

    
    if entity_df is not None and not entity_df.empty:
        entity_agg_metrics = {
            'Entity Sample Size': len(entity_df),
            'Entity Precision': entity_df['entity_precision'].mean(),
            'Entity Recall': entity_df['entity_recall'].mean(),
            'Entity F1-Score': entity_df['entity_f1'].mean(),
            'Body Entity Count': entity_df['body_entity_count'].mean(),
            'Infobox Entity Count': entity_df['infobox_entity_count'].mean(),
            'Summary Entity Count': entity_df['summary_entity_count'].mean(),
            
        }
        agg_metrics.update(entity_agg_metrics)
        
        entity_file = f'../../data/compare/{dataset_name.replace("/", "_")}_entity_metrics.csv'
        entity_df.to_csv(entity_file, index=False)
        print(f"Saved entity metrics for {len(entity_df)} samples to {entity_file}")
        
    return pd.DataFrame([agg_metrics])

def main():
    start = time.time()

    ds_names = ["vi_sums", "en_sums", "pt_sums", 'abisee/cnn_dailymail', 'GEM/wiki_lingua', 'ccdv/arxiv-summarization']
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'])
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    print(f'Start processing {ds_names[task_id]}', flush=True)

    results_df = process_dataset(ds_names[task_id], scorer)
    
    output_file = f'../../data/compare/{ds_names[task_id].replace("/", "_")}_metrics.xlsx'
    results_df.to_excel(output_file, index=False)

    print('Time in mins', round((time.time() - start) / 60), flush=True)

if __name__ == '__main__':
    main()