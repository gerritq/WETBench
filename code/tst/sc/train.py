import os
import json
import argparse
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
import evaluate
from prep_data import gen_ds
from datetime import datetime
from utils import load_jsonl

#assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."

# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--dsubset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--n", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.0)
    args = parser.parse_args()
    
    # create in and out directories
    in_file = f"../../data/{args.language}/datasets/4_{args.language}_{args.dsubset}.jsonl"
    out_dir = f"../../data/{args.language}/sc"

    # for en we have sentences or paras
    if args.language == 'en':
        if args.dsubset.endswith('paras'):
            level = '_paras'
        else:
            level = '_sents'
    else:
        level = ''
    
    # From the overview file, obtain the best score and latest model
    with open(f"{out_dir}/scoreboard_{args.language}{level}.jsonl", "r", encoding="utf-8") as f:
        all_results = [json.loads(line) for line in f]
        if all_results:
            best_score = max(result["eval_accuracy"] for result in all_results)
            latest = [int(result["name"].split('_')[1]) for result in all_results]
            mname = f"model_{max(latest) + 1}"
        else:
            best_score = float("-inf")
            mname = "model_0"
    
    
    # Confirm model name is not taken
    # Set seed
    # does not lead to perfect reproducibility! but close to, compare leaderboard results of same models
    set_seed(args.seed)
    

    # Tokeniser and data collator
    def get_appropriate_tokenizer(model_name, language):
        if 'deberta' in model_name.lower():
            return AutoTokenizer.from_pretrained("xlm-roberta-base")
        
        return AutoTokenizer.from_pretrained(model_name)
    tokeniser = get_appropriate_tokenizer(args.model, args.language)

    #tokeniser = AutoTokenizer.from_pretrained(args.model)
    def tokenize_function(batch):
        return tokeniser(batch['sents'], truncation=True, padding=False)
    #train_ds, val_ds = gen_data(tokeniser, args)
    data_collator = DataCollatorWithPadding(tokenizer=tokeniser)

    # Load data
    ds = gen_ds(in_file, args)
    ds_tok = ds.map(tokenize_function, batched=True)
    
    
    # Define model
    torch.cuda.empty_cache()

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp_results",
        save_strategy="no", 
        logging_strategy="no",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        metric_for_best_model="accuracy",
        report_to='none',
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=args.weight_decay,
        fp16=False,
        warmup_ratio=args.warmup_ratio
    )

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=ds_tok['train'],
        eval_dataset=ds_tok['val'],
        tokenizer=tokeniser,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Save model only when it is better than the best model
    if eval_results["eval_accuracy"] > best_score:
        trainer.save_model(f"{out_dir}/best_model{level}")
        with open(f"{out_dir}/best_model{level}/best_model_name{level}.txt", "w", encoding="utf-8") as f:
            f.write(mname)
    #tokeniser.save_pretrained(f"{out_dir}/{mname}/tokeniser")
    
    # Save log history to a file
    with open(f"{out_dir}/logs{level}/{mname}_log.json", "w", encoding="utf-8") as log_file:
        json.dump(trainer.state.log_history, log_file, ensure_ascii=False, indent=3)


    # Append results to leaderboard
    results = {
        "name": mname,
        "language": args.language,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_accuracy": eval_results["eval_accuracy"],
        "model": args.model,
        "data": args.dsubset,
        "seed": args.seed,
        "train_n": ds_tok['train'].num_rows,
        "val_n": ds_tok['val'].num_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio
    }

    test_results = trainer.evaluate(ds_tok['test'])

    results["test_accuracy"] = test_results["eval_accuracy"]
    
    with open(f"{out_dir}/scoreboard_{args.language}{level}.jsonl", "a", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
        f.write("\n")
        
if __name__ == "__main__":
    main()
