import argparse
import json
import sys

def load_jsonl(file_dir):
    # data = []
    with open(file_dir, 'r', encoding='utf-8') as f:

        return  [json.loads(line) for line in f]

def save_jsonl(out, out_dir):
    # save as jsonl
    with open(out_dir, 'w', encoding='utf-8') as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_jsonl(main_file, update_file):
    
    data1 = load_jsonl(main_file)
    data2 = load_jsonl(update_file)
    
    data2_lookup = {item["id"]: item for item in data2}
    
    merged_data = []
    for item in data1:
        key = item["id"]
        if key in data2_lookup:
            item.update(data2_lookup[key])
        else:
            raise ValueError(f"Item not found in data2: {key}")
        merged_data.append(item)

    return merged_data
    
    
def merge_jsonl_data(data1, data2):
    
    data2_lookup = {item["id"]: item for item in data2}
    
    merged_data = []
    for item in data1:
        key = item["id"]
        if key in data2_lookup:
            item.update(data2_lookup[key])
        else:
            raise ValueError(f"Item not found in data2: {key}")
        merged_data.append(item)
    
    return merged_data


def check_ids(dir1, dir2):
    data1 = load_jsonl(dir1)
    data2 = load_jsonl(dir2)
    
    # check if all ids are the same
    ids1 = {item['id'] for item in data1}
    ids2 = {item['id'] for item in data2}
    
    #sort
    ids1 = sorted(list(ids1))
    ids2 = sorted(list(ids2))
    
    return ids1 == ids2

if __name__ == "__main__":
    
    print(check_ids(sys.argv[1], sys.argv[2]))

