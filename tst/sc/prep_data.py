import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy
import sys
from utils import load_jsonl

def load_filter_jsonl(in_file):
    data = load_jsonl(in_file)
    if 'drop' in data[0].keys():
        data = [item for item in data if not item['drop']]
    
    if 'en_ft' in in_file:
        data = data[:150000]
        print('     EN FT date N max - min', data[0]['timestamp'], data[-1]['timestamp'], flush=True)

    return data

def gen_ds(in_file, args):
    
    data = load_filter_jsonl(in_file)

    sents, labels = [], []
    for p in data:
        sents.append(p['src'])
        labels.append(0)
        sents.append(p['trgt'])
        labels.append(1)
    
    # Create ads
    ds = Dataset.from_dict({'sents': sents, 'labels': labels})
    ds = ds.shuffle(seed=args.seed)
    
    # limit by n
    if str(args.n) != "all":
        ds = ds.select(range(int(args.n)))
    
    # show distribution of labels
    print('   Distribution of labels:', numpy.unique(ds['labels'], return_counts=True), flush=True)
    print('   Number of total samples:', len(ds), flush=True)
    
    # Split into sets
    ds_full = ds.train_test_split(test_size=0.2, seed=args.seed)
    
    dev_test = ds_full['test'].train_test_split(test_size=0.5, seed=args.seed)
    
    ds_full.pop('test')
    ds_full['val'] = dev_test['train']
    ds_full['test'] = dev_test['test']

    # print('Total size:', len(ds_full))
    # print('Size TRAIN', len(ds_full['train']))
    # print('Size VAL', len(ds_full['val']))
    # print('Size TEST', len(ds_full['test']))
    assert len(ds_full['train']) + len(ds_full['val']) + len(ds_full['test']) == sum(len(ds_full[k]) for k in ds_full)

    
    return ds_full


if __name__ == '__main__':
    pass