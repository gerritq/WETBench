import os
import numpy as np
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from langchain_core.documents import Document
import time
import hashlib
from diskcache import Cache
import unicodedata
from utils import load_jsonl, save_jsonl
from FlagEmbedding import BGEM3FlagModel
import copy
import sys
import argparse

class Retriever:
    def __init__(self, 
                subsets: list, 
                langs: list):
        self.subsets = subsets
        self.langs = langs
        self.cache_dir = None
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=500,
                                chunk_overlap=100,
                                length_function=len,
                                separators=[
                                    "\n\n", "\n", ".", "\uff0e", "\u3002", ",", "\uff0c", "\u3001", " ", "\u200B", ""
                                ],
                            )
        self.batch_size = 32
        self.model = BGEM3FlagModel('BAAI/bge-m3',  
                      use_fp16=True,
                      cache_dir="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache")

    def _embeddings(self, chunks):
        embeddings = self.model.encode(chunks, 
                                    batch_size=self.batch_size)['dense_vecs']  # returns n_chunks x 1024
        return embeddings

    def _retrieve_context(self, embeddings, chunks, queries, search_top_k: int = 2):

        
        if embeddings.shape[0] == 0:
            return None

        selected_chunks = []
        seen_chunks = set()

        for query in queries:
            query_embedding = self.model.encode([query], batch_size=self.batch_size)['dense_vecs'][0]
            sim = cosine_similarity([query_embedding], embeddings)[0]
            sorted_indices = np.argsort(sim)[-search_top_k:][::-1]
            for i in sorted_indices:
                chunk = chunks[i]
                if chunk not in seen_chunks:
                    selected_chunks.append((chunk, float(sim[i])))
                    seen_chunks.add(chunk)
        return selected_chunks

    def _single_item(self, item, cache):
        urls = item['docs_urls']
        queries = item['cps']

        docs = []
        for url in urls:
            try:
                cached_item = cache[url]
                if cached_item != "FAILED" and isinstance(cached_item, dict):
                    docs.append(Document(page_content=cached_item["page_content"]))
            except KeyError:
                print(f"[Warning] URL not found in cache: {url}")
                continue
            
        
        # chunk
        chunks = self.text_splitter.split_documents(docs)
        chunks = [chunk.page_content for chunk in chunks] 

        if not chunks:
            return None
        
        # embds
        embeddings = self._embeddings(chunks)

        assert len(chunks) == embeddings.shape[0]

        # query
        context=self._retrieve_context(embeddings, chunks, queries)

        return context

    def run(self):

        for lang in self.langs:
            
            # update cache
            self.cache_dir = os.path.join("/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.meta_cache", f".docs_cache_{lang}")
            cache = Cache(self.cache_dir)
            
            for subset in self.subsets:
                
                in_file = f"../../data/{lang}/ds/{lang}_paras_meta_{subset}.jsonl"
                out_file = f"../../data/{lang}/ds/{lang}_paras_context_{subset}.jsonl"

                data = load_jsonl(in_file)
                print(f'\n\nLANG {lang} in {self.langs} SUBSET {subset} in {self.subsets} SIZE {len(data)}')

                out = []
                for item in data:
                    item_new = copy.deepcopy(item)
                    context = self._single_item(item, cache)
                    item_new.update({'context': context})
                    out.append(item_new)
                save_jsonl(out, out_file)
        return out


def main():

    parser = argparse.ArgumentParser()    
    parser.add_argument('--subsets', nargs='+')
    parser.add_argument('--langs', nargs='+')
    args = parser.parse_args()
    
    retriever = Retriever(subsets=args.subsets, langs=args.langs)
    results = retriever.run()
    
    
if __name__ == "__main__":
    main()
