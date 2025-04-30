import os
import json
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from utils import load_jsonl, save_jsonl
import gc

assert torch.cuda.is_available(), "CUDA is not available"
torch.cuda.empty_cache()

def clear_memory():
    print(f"Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory cleared.")

def gen_ds(args):
    data = load_jsonl(args.in_file)
    #data=data[:100]
    
    texts, labels = [], []
    for item in data:


        if args.task in ['sums', 'first']:
            trgt_text = item['trgt']    
            mgt_text = item['mgt']

        if args.task == 'extend':
            trgt_text = item['trgt_first'].strip() + ' ' + item['trgt'].strip()
            mgt_text = item['trgt_first'].strip() + ' ' + item['mgt'].strip()

        # new task (testing)
        if args.task == 'mix':
            if ('task' in item.keys() and item['task'] == 'extend'):
                trgt_text = item['trgt_first'].strip() + ' ' + item['trgt'].strip()
                mgt_text = item['trgt_first'].strip() + ' ' + item['mgt'].strip()
            else:
                trgt_text = item['trgt']    
                mgt_text = item['mgt']

        # Human text
        texts.append(trgt_text)
        labels.append(0)
        # Machine
        texts.append(mgt_text)
        labels.append(1)
    
    # Create ds
    ds = Dataset.from_dict({'texts': texts, 'labels': labels})
    ds = ds.shuffle(seed=args.seed)
    
    # limit by n
    if str(args.n) != "all":
        ds = ds.select(range(int(args.n)))
    
    # show distribution of labels
    print('   Distribution of labels:', np.unique(ds['labels'], return_counts=True), flush=True)
    print('   Number of total samples:', len(ds), flush=True)

    ds_full = ds.train_test_split(test_size=0.2, seed=args.seed)
    dev_test = ds_full['test'].train_test_split(test_size=0.5, seed=args.seed)
    ds_full.pop('test')
    ds_full['val'] = dev_test['train']
    ds_full['test'] = dev_test['test']

    for subset in ds_full:
        print('\n Subset name: ', subset, flush=True)
        print('N: ', len(ds_full[subset]), flush=True)
    
    return ds_full

def train_model(model_name, args, ds_tok, model_number):
    '''Code mostly taken from HF trainer documentation'''
    print(f"\n======================", flush=True)
    print(f"Training model {model_name}...", flush=True)
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            num_labels=2
        )
        model.config.output_attentions = False
        model.gradient_checkpointing_enable()
        return model

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp_results",
        save_strategy="no", 
        logging_strategy="no",
        eval_strategy="no",
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        metric_for_best_model="accuracy",
        report_to='none',
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=args.weight_decay,
        fp16=True,
        warmup_steps=args.warmup
    )


    # Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"]
        }

    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=ds_tok['train'],
        eval_dataset=ds_tok['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    print('Start training...', flush=True)
    trainer.train()
    
    # val set results
    val_results = trainer.evaluate()

    # test set results
    test_results = trainer.evaluate(eval_dataset=ds_tok['test'])

    results = {
        "start_date": args.date, # same start date is the same run
        "model_number": model_number,
        "name": model_name,
        "in_file": args.in_file,
        "val_accuracy": val_results["eval_accuracy"],
        "test_accuracy": test_results["eval_accuracy"],
        "val_f1": val_results["eval_f1"],
        "test_f1": test_results["eval_f1"],
        "seed": args.seed,
        "train_n": ds_tok['train'].num_rows,
        "val_n": ds_tok['val'].num_rows,
        "test_n": ds_tok['test'].num_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    
    return results, trainer

def save_jsonl_append(data, filename):
    try:
        with open(filename, 'r') as f:
            existing_data = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    with open(filename, 'w') as f:
        for item in existing_data + data:
            f.write(json.dumps(item) + '\n')

def get_leader_stats(args, model_key):
    mod_lead_file = args.lead_file.replace('.jsonl', f'_{model_key}.jsonl')
    os.makedirs(os.path.dirname(os.path.abspath(mod_lead_file)), exist_ok=True)

    if not os.path.exists(mod_lead_file):
        with open(mod_lead_file, 'w') as f:
            pass

    with open(mod_lead_file, "r", encoding="utf-8") as f:
        all_results = [json.loads(line) for line in f]
        if all_results:

            # check if there exist same run data, else 0
            latest_run = max(result["start_date"] for result in all_results)    
            if latest_run == args.date:
                best_score = max([result["val_accuracy"] for result in all_results])
                model_number = [r for r in all_results if r['start_date'] == latest_run][-1]['model_number']
                model_number = f"model_{int(model_number.split('_')[1]) + 1}"
            else:
                best_score = float("-inf")
                model_number = "model_0"
        else:
            # First run
            best_score = float("-inf")
            model_number = "model_0"

        return best_score, model_number

def save_leader(args, trainer, results, best_score, model_key):
    '''Saves leader stats file and overwrite best model, if run was better'''
    mod_lead_file = args.lead_file.replace('.jsonl', f'_{model_key}.jsonl')

    os.makedirs(os.path.dirname(os.path.abspath(args.lead_file)), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_model_dir), exist_ok=True)

    # save best model of better
    if args.save:
        if results["val_accuracy"] > best_score:
            trainer.save_model(args.best_model_dir)
            with open(os.path.join(args.best_model_dir, "best_model_name.txt"), "w", encoding="utf-8") as f:
                f.write(results['model_number'] + f" ({results['start_date']})" )
    
    # append to leaderboard file
    with open(mod_lead_file, "a", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
        f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--lead_file", type=str, required=True)
    parser.add_argument("--best_model_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', default=["microsoft/mdeberta-v3-base", "FacebookAI/xlm-roberta-base"], 
                        help="List of model names to train sequentially")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument('--save', action='store_true')
    
    # HPs 
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)

    
        
    # Load data
    print('\n======================', flush=True)
    print('Load and prep data...', flush=True)
    ds = gen_ds(args)
    print('Total data size: ', sum(len(ds[split]) for split in ds), flush=True)
    
    for model_name in args.models:
        model_key = "mdeberta" if "mdeberta" in model_name.lower() else "roberta"
        best_score, model_number = get_leader_stats(args, model_key)
        
        global tokenizer, data_collator
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Tokenize data
        # def tokenize_function(batch):
        #     return tokenizer(batch['texts'], truncation=True, padding=False, max_length=1024)

        def tokenize_function(batch):
            return tokenizer(
                batch['texts'], 
                truncation=True, 
                padding=False, 
                max_length=512,
                #return_tensors="pt"
            )
            
        ds_tok = ds.map(tokenize_function, batched=True)
        
        results, trainer = train_model(model_name, args, ds_tok, model_number)
        clear_memory()
        
        save_leader(args, trainer, results, best_score, model_key)

        print(results)
        
if __name__ == "__main__":
    main()