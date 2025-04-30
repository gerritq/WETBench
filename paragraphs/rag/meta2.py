import json 
import random
import logging
from utils import save_jsonl, load_jsonl
from collections import defaultdict
import sys
from openai import OpenAI
import os
from fake_useragent import UserAgent
import time

from content_prompts import ContentPromptGenerator
from urls import URLFetcher
from docs2 import WebContentLoader
# from retrieve import Retriever
import multiprocessing

print("CPU cores available:", os.cpu_count())
print("Max workers ", multiprocessing.cpu_count())


class Runner:
    def __init__(self, 
                 lang, 
                 subset,
                 n,
                 api_key,
                 cx_id,
                 max_workers=8):

        self.prompt_generator = ContentPromptGenerator(lang, subset, max_workers)
        self.url_fetcher = URLFetcher(lang, api_key, cx_id)
        self.web_content_loader = WebContentLoader(lang)
        #self.retriever = Retriever(subset, lang)
        self.n = n
        self.lang = lang
        self.subset = subset
        self.max_workers = max_workers
        self.existing_items = None
        self.keys_to_keep = {'id', 'revid', 'subset', 'word_tertile', 'refs', 'page_title', 'section_title', 'trgt_first'}
    

    def log_invalid(self, item):
        with open(invalid_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"id": item['id']}, ensure_ascii=False) + '\n')

    def rm_existing(self, all_paras, *file_dirs):
        existing_ids = set()

        for file_dir in file_dirs:
            try:
                existing_data = load_jsonl(file_dir)
                existing_ids.update(item["id"] for item in existing_data)
                print(f"Loaded {len(existing_data)} existing items from {file_dir}", flush=True)
            except FileNotFoundError:
                pass

        filtered_paras = [p for p in all_paras if p["id"] not in existing_ids]
        print(f"Removed {len(all_paras) - len(filtered_paras)} items in total", flush=True)

        return filtered_paras


    def load_sample(self, in_file, first_file, extend_file, invalid_file):
        
        # load, rm unwanted (will be done on prev cleaning) and shuffle
        all_paras = load_jsonl(in_file)
        assert len(all_paras) == len(set([item['id'] for item in all_paras])), f'duplicates found'
        print('\n\nTotal N', len(all_paras), flush=True)
        #all_paras = [item for item in all_paras if item['refs']]
        all_paras = [item for item in all_paras if item['refs'] and any(refs for refs in item['refs'].values())]
        print('Total after min one reference and min 20 chars', len(all_paras), flush=True)

        random.seed(42)
        random.shuffle(all_paras)

        # IMPORTANT: get first paras only if subset first
        if self.subset == 'first':
            all_paras = [p for p in all_paras if p['loc_para'] == 1]
            print('Total after keeping first paras only', len(all_paras))

        # rm existing data (avoid overlap in eval samples)
        all_paras = self.rm_existing(all_paras, *( [first_file, invalid_file] if self.subset == 'extend' else [extend_file, invalid_file] ))
        print('Total after rm existing paras', len(all_paras))

        # get existing items for logging
        try:
            file_ = extend_file if self.subset == 'extend' else first_file
            existing_data = load_jsonl(file_)
            self.existing_items = len(existing_data)
            print(f"Found {len(existing_data)} items from {file_}", flush=True)
        except FileNotFoundError:
            self.existing_items = 0

        # not necessary to keep all data 
        processing_paras = []

        for item in all_paras:
            item_new = {key: item[key] for key in self.keys_to_keep if key in item}
            item_new['trgt'] = item['trgt_second'] if self.subset == 'extend' else item['trgt']
            item_new['trgt_n_toks'] = item['n_toks_trgt_second'] if self.subset == 'extend' else item['n_toks']
            processing_paras.append(item_new)

        # sample by size
        sample_by_size = {"low": [], "medium": [], "high": []}
        for item in processing_paras:
            sample_by_size[item["word_tertile"]].append(item)

        size = self.n // 3

        final_sample = (
            sample_by_size["low"][:size] +
            sample_by_size["medium"][:size] +
            sample_by_size["high"][:size]
        )

        return final_sample
            
    def run(self):
        sample = self.load_sample(in_file, first_file, extend_file, invalid_file)
        print(f'Desired sample size for {self.lang} {self.subset} {self.n} - existing items {self.existing_items}')
        time.sleep(10)

        print(f'\n{time.ctime()} Fetching content prompts ....', flush=True)
        inter_data = self.prompt_generator.generate_prompts_parallel(sample)
        # rm if cp could not be found
        inter_data = [x for x in inter_data if x['cps'] != 'FAILURE']
        print('Len after rm failed cps', len(inter_data))

        print(f'\n{time.ctime()} Fetching URLs ....', flush=True)
        inter_data = self.url_fetcher.fetch_urls(inter_data)
        print(f'\n{time.ctime()} Fetching docs ....', flush=True)
        inter_data = self.web_content_loader.fetch_content(inter_data)
        # print('\nFetching context  ....', flush=True)
        # inter_data = self.retriever.retrieve(inter_data)
        
        print(f'\n{time.ctime()} DONE ....', flush=True)

        # KKEP CONTEXT ONLY
        valid_items = []
        self.keys_to_keep.update({'trgt', 'trgt_n_toks', 'cps', 'docs_urls'})

        count=0
        for item in inter_data:
            if not item['docs_urls']:
                self.log_invalid(item)
                count+=1
                continue
            else:
                item_new = {key: item[key] for key in self.keys_to_keep if key in item}
                valid_items.append(item_new)
                
        print(f'Found {count} invalid items')
        with open(out_file, 'w', encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__': 

    lang = sys.argv[1]
    subset = sys.argv[2]
    n = int(sys.argv[3])
    api_key = sys.argv[4]
    cx_id = sys.argv[5]

    if n % 3 != 0:
        raise ValueError(f"Error: {n} is not divisible by 3")


    in_file = f'../../data/{lang}/ds/{lang}_paras.jsonl'
    first_file = f'../../data/{lang}/ds/{lang}_paras_meta_first.jsonl'
    extend_file = f'../../data/{lang}ds/{lang}_paras_meta_extend.jsonl'
    out_file = f'../../data/{lang}/ds/{lang}_paras_meta_{subset}.jsonl'
    invalid_file = f'../../data/{lang}/ds/{lang}_invalid.jsonl'

    runner = Runner(
        lang=lang,
        subset=subset,
        n=n,
        max_workers=8,
        api_key=api_key,
        cx_id=cx_id
    )

    runner.run()
