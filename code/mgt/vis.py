from utils import load_jsonl
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
import sys
import re
from collections import Counter

def load_data(dname, dir, max_samples):
    data = load_jsonl(dir)

    # neutral
    if dname in ['TST', 'WOE']:
        data_out = data[:max_samples]
    else:
        # other
        n_per_tertile = max_samples // 3

        data_by_tertiles = {'low': [], 'medium': [], 'high': []}

        for item in data:
            data_by_tertiles[item['word_tertile']].append(item)

        data_out =  (data_by_tertiles['low'][:n_per_tertile] + 
                data_by_tertiles['medium'][:n_per_tertile] + 
                data_by_tertiles['high'][:n_per_tertile])

    # return hwt, mgt lists
    return [item['trgt'] for item in data_out], [item['mgt'] for item in data_out]
    
def compute_levenshtein(texts1, texts2):
    assert len(texts1) == len(texts2), "different length"
    
    scores = []
    for text1, text2 in zip(texts1, texts2):
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        edit_distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        normalized_distance = edit_distance / max_len
        scores.append(normalized_distance)
    
    return scores

def compute_semantic_similarities(texts1, texts2, model):
    assert len(texts1) == len(texts2), "different length"

    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)

    similarities = []
    for i in range(len(texts1)):
        # https://huggingface.co/tasks/sentence-similarity
        similarity = util.pytorch_cos_sim(embeddings1[i], embeddings2[i]).item()
        similarities.append(similarity)
    return similarities

def compute_unigram_overlap(texts1, texts2):
    assert len(texts1) == len(texts2), "check length"
    
    overlaps = []
    
    for text1, text2 in zip(texts1, texts2):
        
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        overlap_tokens = counter1 & counter2
        
        intersection_size = sum(overlap_tokens.values())
        union_size = sum(counter1.values()) + sum(counter2.values()) - intersection_size
        
        overlap_score = intersection_size / union_size
        overlaps.append(overlap_score)
    
    return overlaps

def plot_metric_distributions(dataset_results, metric_name, save_path=None):
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (dataset_name, scores) in enumerate(dataset_results.items()):
        sns.kdeplot(scores, label=f"{dataset_name}", 
                   fill=True, alpha=0.3, color=colors[i % len(colors)])
    
    if metric_name == 'levenshtein':
        plt.xlabel("Normalized Levenshtein Distance", fontsize=14)
    elif metric_name == 'semantic':
        plt.xlabel("Cosine Similarity", fontsize=14)
    elif metric_name == 'unigram':
        plt.xlabel("Unigram Overlap", fontsize=14)
    
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_combined_metrics(levenshtein_results, semantic_results, unigram_results, save_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (dataset_name, scores) in enumerate(levenshtein_results.items()):
        sns.kdeplot(scores, label=f"{dataset_name}", 
                   fill=True, alpha=0.3, color=colors[i % len(colors)], ax=axes[0])
    
    axes[0].set_xlabel("Normalized Levenshtein Distance", fontsize=14)
    axes[0].set_ylabel("Density", fontsize=14)
    axes[0].set_title("Syntactic Similarity", fontsize=16)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, (dataset_name, scores) in enumerate(semantic_results.items()):
        sns.kdeplot(scores, label=f"{dataset_name}", 
                   fill=True, alpha=0.3, color=colors[i % len(colors)], ax=axes[1])
    
    axes[1].set_xlabel("Cosine Similarity", fontsize=14)
    axes[1].set_title("Semantic Similarity", fontsize=16)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    for i, (dataset_name, scores) in enumerate(unigram_results.items()):
        sns.kdeplot(scores, label=f"{dataset_name}", 
                   fill=True, alpha=0.3, color=colors[i % len(colors)], ax=axes[2])
    
    axes[2].set_xlabel("Unigram Overlap", fontsize=14)
    axes[2].set_title("N-gram Similarity", fontsize=16)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              ncol=len(labels), fontsize=12)
    
    for ax in axes:
        ax.get_legend().remove()
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(max_samples=600):
    
    lang = sys.argv[1]
    print(f'Start building plots for {lang}')

    model = SentenceTransformer("BAAI/bge-m3")
    
    base_dir = "/scratch/users/k21157437"
    data_sources = [
        ('WOE', os.path.join(base_dir, f'generalise/data/ds/external/mgt/wiki_{lang}_gpt.jsonl')),
        ('PW', os.path.join(base_dir, f'paras/data/{lang}/ds/mgt/{lang}_paras_rag_first_gpt.jsonl')),
        ('SUM', os.path.join(base_dir, f'sums/data/{lang}/ds/{lang}_sums_mgt_few1_gpt.jsonl')),
        ('TST', os.path.join(base_dir, f'neutral_new/data/{lang}/datasets/mgt/{lang}_default_mgt_few5_gpt.jsonl'))
    ]
    
    levenshtein_results = {}
    semantic_results = {}
    unigram_results = {}
    
    output_dir = os.path.join(base_dir, 'similarity_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    for dname, dpath in data_sources:
        print(f"\nProcessing {dname} dataset...", flush=True)
        src_texts, gen_texts = load_data(dname, dpath, max_samples)
        
        lev_scores = compute_levenshtein(src_texts, gen_texts)
        levenshtein_results[dname] = lev_scores
        
        sem_scores = compute_semantic_similarities(src_texts, gen_texts, model)
        semantic_results[dname] = sem_scores
        
        unigram_scores = compute_unigram_overlap(src_texts, gen_texts)
        unigram_results[dname] = unigram_scores
    
    out_dir = '/scratch/users/k21157437/aid/data/plots'
    os.makedirs(out_dir, exist_ok=True)
    
    levenshtein_plot_path = os.path.join(out_dir, f"lev_dis_{lang}.png")
    semantic_plot_path = os.path.join(out_dir, f"sem_dis_{lang}.png")
    unigram_plot_path = os.path.join(out_dir, f"unigram_dis_{lang}.png")
    combined_plot_path = os.path.join(out_dir, f"combined_metrics_{lang}.png") 
    
    plot_metric_distributions(levenshtein_results, 'levenshtein', levenshtein_plot_path)
    plot_metric_distributions(semantic_results, 'semantic', semantic_plot_path)
    plot_metric_distributions(unigram_results, 'unigram', unigram_plot_path)
    
    plot_combined_metrics(levenshtein_results, semantic_results, unigram_results, combined_plot_path)


if __name__ == "__main__":
    main()