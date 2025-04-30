from bs4 import BeautifulSoup
import requests
import sys
import json
import os
import numpy as np
from utils import load_jsonl

def get_diff(revision_id, lang):
    '''Takes a revision id
    Returns the diff page to the previous revision'''
    
    S = requests.Session()
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "compare",
        "fromrev": revision_id,
        "torelative": "prev",
        "format": "json"
    }
    response = S.get(url=URL, params=PARAMS)
    data = response.json()
    
    if 'warnings' in data.keys() or 'error' in data.keys():
        return None
    else:
        return data['compare']['*']

def get_revs(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Explicit table structure of diff
    trs = soup.find_all('tr')
    
    # Collect pairs of only parallel edits (src, trgt)
    # Parallel edits are those that have both src and trgt
    pairs=[]
    non_edits=False
    for tr in trs:
        # Get source
        src = tr.find('td', {'class': 'diff-deletedline'})
        if src:
            src_markers = str(src.div)
        # Get target
        trgt = tr.find('td', {'class': 'diff-addedline'})
        if trgt:
            trgt_markers = str(trgt.div)
        # Append to pairs
        if src and trgt: # does not mean both has been done
            pairs.append((src_markers, trgt_markers))
        if (src and not trgt) or (not src and trgt): # del/added chunks
            non_edits = True
            
    return pairs, non_edits

def main():
    
    lang = sys.argv[1]  
    
    # dirs
    
    in_file = f'../../data/{lang}/processing/1_{lang}_nrevs.jsonl'
    out_dir = f'../../data/{lang}/processing/temp/'
    job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    out_file = f'{out_dir}j{job_id}_{lang}_crawl.jsonl'
    
    # Load data
    data = load_jsonl(in_file)
    #data=data[:12]
    splits = np.array_split(data, n_jobs)
    data = splits[job_id]
    del splits
    data = list(data)
    
    # Crawl diffs
    for revision in data:
        html = get_diff(revision['revid'], lang)
        if html is None:  # case where there exists no previous page
            continue
        
        pairs, non_edits = get_revs(html)
        
        revision.update({
            'non_edits': non_edits,
            'pairs': pairs
        })
        
        with open(out_file, 'a', encoding='utf-8') as out_f:
            json.dump(revision, out_f, ensure_ascii=False)
            out_f.write('\n')


if __name__ == '__main__':
    main()
