import json
import os
import regex as rex
import unicodedata
import sys
import numpy as np
import multiprocessing
from nltk import sent_tokenize
from utils import load_jsonl, save_jsonl
from tqdm import tqdm
import random 

lang = sys.argv[1]
n_sample = 270

CONFIG = {'en': {'infobox': "Infobox"},
               'pt': {'infobox': "Infocaixa"},
               'vi': {'infobox': "Hộp thông tin"}}

CONFIG = CONFIG[lang]

def clean(x):
    refs_PATTERN = r'\[sup:.*?:sup\]'
    x = rex.sub(refs_PATTERN, '', x).strip()
    x = rex.sub(r'([\(\[\{])\s+', r'\1', x)
    x = rex.sub(r'\s+([\)\]\}])', r'\1', x)
    x = rex.sub(r'\s+([.,:;!?])', r'\1', x)
    x = rex.sub(r'[^\S\r\n]{2,}', ' ', x)

    return rex.sub(refs_PATTERN, '', x).strip()

def format_infobox(infobox):
    formatted_infobox = [f"== {CONFIG['infobox']} =="] + [clean(x) for x in infobox] + ['\n']
    formatted_infobox = '\n'.join(formatted_infobox)
    return formatted_infobox.strip()


def assign_tertile(item, tertiles):
    if item['n_toks_trgt'] <= tertiles[0]:
        return 'low'
    elif item['n_toks_trgt'] <= tertiles[1]:
        return 'medium'
    else:
        return 'high'

def process_item(item):
    
    # INFOBOX
    if item['infobox']:
        infobox = format_infobox(item['infobox'])
    else:
        infobox = None

    # BODY
    body = ""
    for section in item["sections"]:
        for title, paragraphs in section.items():
            body += f"{title}\n"
            body += '\n\n'.join(paragraphs) + '\n\n'
            # body += "\n\n"
    body = clean(body)

    # LEAD
    lead = clean('\n\n'.join(item.get('lead', [])))
    
    n_toks_lead = len(lead.split()) # lapata use tokens and just split
    n_toks_body = len(body.split())
    n_sents_lead = len(sent_tokenize(lead))
    n_sents_body = len(sent_tokenize(body))

    return {
        'revid': item.get('revid', None),
        'page_title': item.get('title', ""),
        'pageid': item.get('pageid', None),
        'n_toks_trgt': n_toks_lead,
        'n_toks_src': n_toks_body,
        'n_sents_trgt': n_sents_lead,
        'n_sents_src': n_sents_body,
        'word_ratio': n_toks_body / n_toks_lead if n_toks_lead > 0 else 0,
        'n_refs': len(item.get('refs', [])),
        'infobox': infobox,
        'trgt': lead,
        'src': body
    }

def main():
    
    in_file = f"/scratch_tmp/users/k21157437/aid/data/{lang}/3_{lang}_text.jsonl"
    out_file_eval = f"../../data/{lang}/ds/{lang}_sums_eval.jsonl"
    out_file_full = f"../../data/{lang}/ds/{lang}_sums.jsonl"
    out_file_tertiles = f"../../data/{lang}/ds/{lang}_tertiles.jsonl"

    data = load_jsonl(in_file)
    #data = data[:2]
    print(f"Total items to process: {len(data)}", flush=True)

    out = [process_item(item) for item in data]


    # Gen trgt and src stats
    avg_n_toks_trgt = round(np.mean([x['n_toks_trgt'] for x in out]), 2)
    std_n_toks_trgt = round(np.std([x['n_toks_trgt'] for x in out]), 2)
    
    avg_n_toks_src = round(np.mean([x['n_toks_src'] for x in out]), 2)
    std_n_toks_src = round(np.std([x['n_toks_src'] for x in out]), 2)
    

    # OLD FILTERING
    n = len(out)
    if lang == 'vi':
        # For Vietnamese, add the additional 2900 token constraint
        out = [x for x in out if (
            (10 <= x['n_toks_trgt'] <= avg_n_toks_trgt + 2 * std_n_toks_trgt) and 
            (100 <= x['n_toks_src'] <= min(2900, avg_n_toks_src + 2 * std_n_toks_src))
        )]
    else:
        out = [x for x in out if (
            (10 <= x['n_toks_trgt'] <= avg_n_toks_trgt + 2 * std_n_toks_trgt) and 
            (100 <= x['n_toks_src'] <= avg_n_toks_src + 2 * std_n_toks_src)
        )]
       
    print('Outliers removed', n - len(out), flush=True)

    # Tertiles for trgt
    trgt_counts = [pair['n_toks_trgt'] for pair in out]
    trgt_tertiles = np.percentile(trgt_counts, [33.33, 66.67])

    src_counts = [pair['n_toks_src'] for pair in out]
    src_tertiles = np.percentile(src_counts, [33.33, 66.67])


    meta_info = [{'trgt_tertiles': list(trgt_tertiles),
                  'trgt_min_len': min(trgt_counts),
                  'trgt_max_len': max(trgt_counts)},
                 {'src_tertiles': list(src_tertiles),
                  'src_min_len_trgt': min(src_counts),
                  'src_max_len_trgt': max(src_counts)}
                ] 
    save_jsonl(meta_info, out_file_tertiles)
    print(f"Tertiles: {meta_info}", flush=True)

    for i, x in enumerate(out):
        x['word_tertile'] = assign_tertile(x, trgt_tertiles)
        x['id'] = i

    save_jsonl(out, out_file_full)

    # SAMPLE
    sample = {"low": [], "medium": [], "high": []}

    for x in out:
        sample[x["word_tertile"]].append(x)

    random.seed(42)
    for key in sample:
        np.random.shuffle(sample[key])

    size = n_sample // 3
    sampled_data = (
        sample["low"][:size] +
        sample["medium"][:size] +
        sample["high"][:size]
    )

    print(f"Sampled {len(sampled_data)} items.", flush=True)

    save_jsonl(sampled_data, out_file_eval)

if __name__ == "__main__":
    main()
