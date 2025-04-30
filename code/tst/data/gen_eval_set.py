from utils import load_jsonl, save_jsonl
import random
import sys

lang = sys.argv[1]
subset = sys.argv[2]

in_file = f'../../data/{lang}/datasets/4_{lang}_{subset}.jsonl'
out_file = f'../../data/{lang}/datasets/eval/{lang}_{subset}_eval.jsonl'

def main():
    
    data = load_jsonl(in_file)
    print('Total N' , len(data))
    data = [item for item in data if not item['drop']]
    print('Keeping' , len(data), f'of {subset}')
    # draw random sample
    random.seed(42)
    random.shuffle(data)
    data = data[:270]
    
    save_jsonl(data, out_file)

if __name__ == "__main__":
    main()