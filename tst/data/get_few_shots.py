from utils import load_jsonl
import random
import sys

lang = sys.argv[1]
dset = sys.argv[2]

in_file = f'../../data/{lang}/datasets/4_{lang}_{dset}.jsonl'

data = load_jsonl(in_file)
data = [item for item in data if not item['drop']]

random.seed(42)
rdata = random.sample(data, 50)

for i, sample in enumerate(rdata):
    print('\nSample', i)
    print(sample['revid'], ':', )
    print('src:', sample['src'])
    print('trgt:', sample['trgt'])