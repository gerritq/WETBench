import numpy as np
import json
import sys
import random
import regex as rex
import os
from nltk import sent_tokenize, word_tokenize
from utils import load_jsonl, save_jsonl

lang = sys.argv[1]
in_file = f"/scratch_tmp/users/k21157437/aid/data/{lang}/3_{lang}_text.jsonl"
out_file = f"../../data/{lang}/ds/{lang}_paras.jsonl"
out_file_tertiles = f"../../data/{lang}/ds/{lang}_tertiles.jsonl"
n_data=20000

def linearise_paragraphs(data):
    '''Takes data where item contains lead and sections.
    Reshapes to paragraph level data with info about section, location, length etc.'''
        
    def clean(x):
        refs_PATTERN = r'\[sup:.*?:sup\]'
        x = rex.sub(refs_PATTERN, '', x).strip()
        return x
    
    def obtain_relevant_refs(item):
        '''Extracts relevant references from the text and filter them in the refs dict'''
        refs_PATTERN = r"\[sup:(.*?):sup\]"
        # Extract and process the references
        relevant_refs = [r.replace('#', '') for r in rex.findall(refs_PATTERN, item['trgt_refs'])]
        relevant_refs = {ref: item['refs'][ref] for ref in relevant_refs if ref in item['refs'].keys()}
        return relevant_refs
        
    
    def assign_tertile(para, tertiles):
        if para['n_toks'] <= tertiles[0]:
            return 'low'
        elif para['n_toks'] <= tertiles[1]:
            return 'medium'
        else:
            return 'high'

    def split_paragraph(item):
        """Split paragraph into two halves while avoiding cutting mid sents!! """
        
        sentences = sent_tokenize(item['trgt'])

        assert len(sentences) > 2, f'Expected more than 2 sentences but got {len(sentences)} {item["trgt"]}'
        
        total_unigrams = sum(len(word_tokenize(sent)) for sent in sentences)
        half_word_count = total_unigrams // 2

        # Find the closest sentence boundary to 50% of words
        first_half, second_half = [], []
        current_word_count = 0

        # ensure first sentence is always added in first half
        # thse pertains to just a few cases so we do not worry about imbalance
        first_half.append(sentences[0].strip())
        current_word_count += len(word_tokenize(sentences[0]))
        
        for sentence in sentences[1:]:
            current_word_count += len(word_tokenize(sentence))
            if current_word_count <= half_word_count:
                first_half.append(sentence.strip())
            else:
                second_half.append(sentence.strip())

        item['trgt_first'] = ' '.join(first_half)
        item['n_toks_trgt_first'] = len(word_tokenize(item['trgt_first']))

        item['trgt_second'] = ' '.join(second_half)
        item['n_toks_trgt_second'] = len(word_tokenize(item['trgt_second']))

        return item

    paras = []
    unique_counter = 0 

    for item in data: 
        for el in item['sections']:
            for section_title, paragraphs in el.items():
                if not paragraphs:
                    continue
                n_ps = len(paragraphs)
                for i, p in enumerate(paragraphs, start=1):
                    unique_counter += 1

                    trgt = clean(p)
                    n_sents = len(sent_tokenize(trgt))
                    n_toks = len(word_tokenize(trgt))
                    
                    if i == 1:
                        prev_p = None
                    else:
                        prev_p = clean(paragraphs[i-2])
                    
                    out = {
                        'id': unique_counter,
                        'page_title': item['title'],
                        'pageid': item['pageid'],
                        'revid': item['revid'],
                        'timestamp': item['timestamp'],
                        'section_title': section_title,
                        'trgt': trgt,
                        'loc_para': i,
                        'n_paras': n_ps,
                        'n_sents': n_sents,
                        'n_toks': n_toks,
                        'trgt_refs': p,
                        'prev_p': prev_p,
                        'refs': item['refs']
                    }
        
                    out['refs'] = obtain_relevant_refs(out)
                    paras.append(out)


    # FILTER
    # min sentence length of three and min char count
    n = len(paras)
    print(f'Total n before min len and rm outliers {n}', '\n')
    paras = [p for p in paras if p['n_sents'] > 2 and len(p['trgt']) > 20] # latter already done in 3_get_text bu does nto hurt to do it
    print(f'Dropped {n - len(paras)} paragraphs as having too few sentences; total paras {len(paras)}', '\n')

    # drop outliers
    n = len(paras)
    avg_tokens = round(np.mean([p['n_toks'] for p in paras]), 2)
    std_tokens = round(np.std([p['n_toks'] for p in paras]), 2)
    paras = [p for p in paras if (avg_tokens - 2 * std_tokens) <= p['n_toks'] <= (avg_tokens + 2 * std_tokens)]
    
    print(f'Dropped {n - len(paras)} paragraphs as outliers; total paras {len(paras)}', '\n')
    
    # TERTILES
    token_counts = [p['n_toks'] for p in paras]
    tertiles = np.percentile(token_counts, [33.33, 66.67])
    save_jsonl([list(tertiles)], out_file_tertiles)
    print(f"Tertiles: {tertiles}", flush=True)
    
    # APPLY PARA SPLITTING
    paras = [split_paragraph(p) for p in paras]

    for p in paras:
        p['word_tertile'] = assign_tertile(p, tertiles)
        assert p['trgt_first'] and p['trgt_second'], f"Empty paragraph split 1. {p['trgt_first']} 2. {p['trgt_second']}"

    return paras
def main():

    data = load_jsonl(in_file)
    print('Original data size', len(data))
    # We restrict this to a max of 20k random pages
    data = data[:n_data]

        
    print(' -------- Linearise ..... --------')
    paras = linearise_paragraphs(data)
    print('Total paras', len(paras), '\n', flush=True)
    
    save_jsonl(paras, out_file)

if __name__ == '__main__':
    main()
