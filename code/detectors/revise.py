# Code is taken from: https://github.com/thunlp/LLM-generated-text-detection/blob/main/finance_bart_auroc.py

from bart_score import BARTScorer
import json
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import torch
torch.cuda.empty_cache()

print('GPU available', torch.cuda.is_available())

client = OpenAI(api_key='') 

class ReviseDetect:

    def __init__(self,
                lang,
                model="gpt-3.5-turbo",
                workers=8):
        self.bartscorer = BARTScorer(device='cuda:0',checkpoint="facebook/bart-large-cnn")
        self.workers = workers
        self.model = model
        self.lang = lang
        self.prompts = {
                "en": "Revise the following text: {text}",
                "pt": "Revise o seguinte texto: {text}",
                "vi": "Sửa lại văn bản sau: {text}"
                }

    def _chat(self, prompt):
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("Exception found: ", e, flush=True)
            return " "
            

    def run(self, texts):
        
        results = {}
        
        def process_text(idx, text):
            prompt = self.prompts[self.lang].format(text=text)
            rev_text = self._chat(prompt)
            return idx, text, rev_text
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_idx = {executor.submit(process_text, idx, text): idx for idx, text in enumerate(texts)}
            
            for future in tqdm(as_completed(future_to_idx), total=len(texts)):
                idx, text, rev_text = future.result()
                results[idx] = (text, rev_text)

        
        og_texts = [results[i][0] for i in results]
        revised = [results[i][1] for i in results]


        scores = self.bartscorer.score(revised, og_texts)
        return scores





