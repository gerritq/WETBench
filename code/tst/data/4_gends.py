import json
import numpy as np
import argparse
from collections import Counter
import copy
import regex as rex
from utils import load_jsonl, save_jsonl
import sys
from datetime import datetime

lang = sys.argv[1]
generation = sys.argv[2]


def drop_lengthy_pairs(data, filter_counts):
    '''Drops pairs where the length ratio is more than 2 stds away from the mean for drop=False items.
    Updates the filter_counts with the number of length_ratio drops.'''
    
    for item in data:
        if not item['drop']:
            item['ratio'] = len(item['trgt']) / len(item['src'])  # >1 means trgt is longer
    
    ratios = np.array([item['ratio'] for item in data if not item['drop']])
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)

    upper = mean_ratio + 2 * std_ratio
    lower = mean_ratio - 2 * std_ratio
    
    filtered_data = []
    for item in data:
        new_item = copy.deepcopy(item)
        if not new_item['drop'] and 'ratio' in new_item:
            if new_item['ratio'] > upper or new_item['ratio'] < lower:
                new_item['drop'] = True
                new_item['reason'] = 'length_ratio'
                filter_counts['length_ratio'] += 1
        filtered_data.append(new_item)
    
    return filtered_data, filter_counts

def drop_dups(data, filter_counts):
    unique_data = []
    seen = set()

    for item in data:
        if item['drop']:
            unique_data.append(item)
            continue

        # For non-dropped items, check if the (src, trgt) pair is a duplicate
        pair = (item['src'], item['trgt'])
        if pair not in seen:
            unique_data.append(item)
            seen.add(pair)
        else:
            new_item = copy.deepcopy(item)
            new_item['drop'] = True
            new_item['reason'] = 'duplicates'
            unique_data.append(new_item)
            filter_counts['duplicates'] = filter_counts.get('duplicates', 0) + 1

    return unique_data, filter_counts

def write_summary_report(data, filtered_data, filter_counts, drop_counts, lang, out_dir, generation):

    keep_count = sum(1 for item in filtered_data if not item.get('drop'))
    output_path = out_dir + f"4_{lang}_{generation}_sum.txt"

    with open(output_path, 'w', encoding='utf-8') as summary_f:
        summary_f.write(f"Total: {len(data)}\n")
        summary_f.write(f"Keep: {keep_count}/{keep_count*2} pairs\n\n")

        summary_f.write("Pair-Filter counts:\n")
        summary_f.write(f"Total: {sum(filter_counts.values())}\n")
        for reason, count in filter_counts.items():
            summary_f.write(f"{reason:<30}{count:>10}\n")

        summary_f.write("\nRevision-Filter counts:\n")
        summary_f.write(f"Total drops: {sum(drop_counts.values())}\n")
        for reason, count in drop_counts.items():
            summary_f.write(f"{reason:<30}{count:>10}\n")


# --------------------------------------------------------------------------------------------------
# Pair-level filters
# --------------------------------------------------------------------------------------------------


def apply_filters(data, lang, gentype, chunk_level_filters, pair_level_filters, out_dir): 
    
    filter_counts = Counter()
    drop_counts = Counter()

    if lang == 'vi':
        pair_level_filters['min_edit'] = lambda item: item['lev'] < 3

    if gentype == 'default':
        # rm revert and only one sentence edit
        chunk_level_filters.pop('revert')
        # if  lang == 'vi': # to be decided
        #     chunk_level_filters.pop('n_matches')
    
    if gentype == 'mpairs':
        # rm revsion and multiple edit pair filters
        chunk_level_filters.pop('revert')
        chunk_level_filters.pop('n_matches') # to be decided
        if lang == 'vi':
            chunk_level_filters["non_edits"] = lambda item: item["non_edits"] == True # add this bc we tested to included those cases - wasn't working
            chunk_level_filters['n_pairs'] = lambda item: item['n_pairs'] > 2
        else:
            chunk_level_filters['n_pairs'] = lambda item: item['n_pairs'] > 2
        
    if gentype == 'ft':
        # rm revert filter
        chunk_level_filters.pop('revert')
        # add pov only
        if lang == 'en':
            regex_pov = rex.compile(r"((?:^|[\W_])(?:n?pov)(?![\w]))|(?:neutral point of view)",
                                    rex.IGNORECASE | rex.UNICODE)
        chunk_level_filters['no_npov'] = lambda item: not regex_pov.search(item['comment'])
        

  # apply high and low level filters
    filtered_data = []
    for item in data:
        new_data = copy.deepcopy(item)
        # high level first 
        if not new_data['drop']:
            # Apply high level filters
            for reason, condition in chunk_level_filters.items():
                if condition(new_data):
                    filter_counts[reason] += 1
                    new_data['drop'] = True
                    new_data['reason'] = reason
                    break
            # will not be exceuted if high level filter was triggered; ie if break was called
            else:
                # Apply low level filters only if no high level filter was triggered
                for reason, condition in pair_level_filters.items():
                    if condition(new_data):
                        filter_counts[reason] += 1
                        new_data['drop'] = True
                        new_data['reason'] = reason
                        break
        filtered_data.append(new_data)
    
    # drop lengthy pairs
    filtered_data, filter_counts = drop_lengthy_pairs(filtered_data, filter_counts)
    
    # drop duplicates
    filtered_data, filter_counts = drop_dups(filtered_data, filter_counts)
    
    # drop counter
    for item in data:
        if item['drop']:
            drop_counts[item['reason'].split()[0]] += 1
    
    # sum
    write_summary_report(data, filtered_data, filter_counts, 
                         drop_counts, lang, out_dir, generation)
        
    # if ft, sort the data by timestamp and pick the latest 150k pairs    
    # order latest to oldest             
    if gentype == 'ft':
        filtered_data = sorted(filtered_data, key=lambda x: datetime.strptime(x['timestamp'], 
                                                                              "%Y-%m-%dT%H:%M:%SZ"), 
                                                                              reverse=True)
        
    # apply unique ids
    for i in range(len(filtered_data)):
        filtered_data[i]['id'] = i
    
    return filtered_data

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------

def main():
    
    in_file = f'../../data/{lang}/processing/3_{lang}_proc.jsonl'
    out_dir = f'../../data/{lang}/datasets/'
    
    data = load_jsonl(in_file)

    # ----------------------------------------------------------------------------------------------
    # Define reverts
    # ---------------------------------------------------------------------------------------------- 
    if lang == 'pt':
        revert_rex = rex.compile(r"(desfeita)|(wp:(rev|ra))|(revertendo para)|(wikipédia:reversão)|(revertidas edições)|\b(bot|robô|robôs)\b",
                                rex.IGNORECASE | rex.UNICODE)

    # if lang == 'vi':
    #     revert_rex = rex.compile(r"xx",
    #                             rex.IGNORECASE | rex.UNICODE)

    # if lang == 'en':
    #     revert_rex = rex.compile(r"undo|undid|revert",
    #                             rex.IGNORECASE | rex.UNICODE)

    # ----------------------------------------------------------------------------------------------
    # Define filters
    # ---------------------------------------------------------------------------------------------- 
    # FILTERS
    chunk_level_filters = {
        'revert': lambda item: revert_rex.search(item['comment']),
        'n_pairs': lambda item: item['n_pairs'] > 1,
        'n_matches': lambda item: item['n_matches'] > 1,
    }

    pair_level_filters = {
        'bleu=1': lambda item: item['bleu'] == 1,
        'low_bleu': lambda item: item['bleu'] < 0.25,
        'min_edit': lambda item: item['lev'] < 4,
        'max_edit': lambda item: item['sh_del'] > 0.5,
        'punct_only': lambda item: item['sh_punct_add'] == 1 and item['sh_punct_del'] == 1,
        'sent_pnouns': lambda item: item['src_sh_pnouns_sent'] > .5,

    }
    out_data = apply_filters(data, lang, generation, chunk_level_filters, pair_level_filters, out_dir)
    save_jsonl(out_data, out_dir + f"4_{lang}_{generation}.jsonl")

if __name__ == '__main__':
    main()

