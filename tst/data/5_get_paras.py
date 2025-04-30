import json
import sys
from utils import load_jsonl, save_jsonl
from collections import defaultdict
from nltk import sent_tokenize
import nltk
nltk.download('punkt_tab')

lang = sys.argv[1]
ds = sys.argv[2]
p_len = 3

def get_paras(sents):
    
    if len(sents) < p_len:
        return []
    
    # sprt by index
    sorted_sents = sorted(sents, key=lambda x: x[2])
    
    # find longest adjacents sents based on index
    para = []
    paras = []
    for i in range(1, len(sorted_sents)):
        # diff
        diff=sorted_sents[i][2]-sorted_sents[i-1][2]
        
        # adjacent
        if diff == 1:
            para.append(sorted_sents[i-1])
            if i == len(sorted_sents) - 1:
                para.append(sorted_sents[i])
        if diff > 1:
            if para:
                para.append(sorted_sents[i-1])
                paras.append(para)
                para = []
                
    if para:
        paras.append(para)
        
    # hcheck 
    for p in paras:
        idx = [x[2] for x in p]
        assert len(idx) == sum([abs(y - x) for x, y in zip(idx, idx[1:])]) +1
    paras = [para for para in paras if len(para) >= p_len]
    
    # return text
    p_text = [(' '.join(x[0] for x in para), ' '.join(x[1] for x in para)) for para in paras]
    
    return p_text

def main():
    
    in_file = f'../../data/{lang}/datasets/4_{lang}_{ds}.jsonl'
    out_file = f'../../data/{lang}/datasets/4_{lang}_{ds}_paras.jsonl'
    data = load_jsonl(in_file)


    data = [item for item in data if not item['drop']] 
    
    print(' Group revids', flush=True)
    revision_sens = defaultdict(list)
    for item in data:
        revision_sens[item['revid'] + '_' + str(item['idx_pair'])].append((item['src'], item['trgt'], item['src_idx']))
    
    print(' Get paras', flush=True)
    out = []
    ids=0
    for revid, sents in revision_sens.items():
        print(revid)
        paras = get_paras(sents)
        if paras:
            for para in paras:
                assert len(sent_tokenize(para[0])) >= p_len, f"Less than {p_len} what is going on: {para[0]}"
                out.append({
                        'revid': revid,
                        'id': ids, 
                        'src': para[0],
                        'src_n': len(para[0].split()),
                        'src_sents': len(sent_tokenize(para[0])),
                        'trgt': para[1],
                        'trgt_n_words': len(para[1].split()),
                        'trgt_sents': len(sent_tokenize(para[1])),
                        'drop': False})
                ids+=1

    print('Len paras', len(out))                
    save_jsonl(out, out_file)

if __name__ == '__main__':
    main()