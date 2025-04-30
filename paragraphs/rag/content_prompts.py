import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from diskcache import Cache
import hashlib
import ast
# dir_to_disk = os.path.join('/scratch_tmp/users/k21157437', ".cp_cache")
# cache = Cache(dir_to_disk)
# print(dir_to_disk)
# print(f"Cache contains {len(cache)} items")

os.environ['OPENAI_API_KEY'] = ''

class ContentPromptGenerator:
    def __init__(self, 
                lang,
                subset,
                cache_dir,
                max_workers=5):

        self.lang = lang
        self.subset = subset
        self.prompt_dir = f'../prompts/{lang}/content_prompt_{lang}.txt'
        self.max_workers = max_workers
        self.client  = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

        # old: '/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.meta_cache'
        dir_to_disk = os.path.join("/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.meta_cache", f".cps_cache_{self.lang}")
        self.cache = Cache(dir_to_disk)
        print(f"Cache directory for {self.lang}: {dir_to_disk}")
        print(f"Cache contains {len(self.cache)} items")

    def _get_text_hash(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_prompt(self):
        with open(self.prompt_dir, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _format_prompt(self, prompt, item):
        item_prompt = item.copy()
        
        item_prompt['min_cp'] = 5 if item_prompt['word_tertile'] == 'low' else 8

        return prompt.format(**item_prompt)

    def _query_gpt(self, prompt, p_id):
        def minimal_cleaning(response):
            response = response.replace('```python', '').replace('```', '').strip()
            try:
                out = ast.literal_eval(response)
            except (ValueError, SyntaxError):
                print("Model output is not valid Python literal:")
                print(response)
                out = "FAILURE"
                return out
            
            if not isinstance(out, list):
                raise ValueError("Output is not a list")
            return out
        
        if p_id in self.cache:
            print(f'Found cached prompt for id {p_id}', flush=True)
            return self.cache[p_id]

        # response = litellm.completion(
        #     model="gpt-4o",
        #     messages=[{ "content": prompt,"role": "user"}],
        #     temperature=0,
        #     caching=True)
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = completion.choices[0].message.content.strip()
        # print(response)
        # response = response.choices[0].message.content.strip()
        return minimal_cleaning(response_text)

    def generate_prompt(self, item):
        p_id = item['id']
        prompt = self._load_prompt()
        formatted_prompt = self._format_prompt(prompt, item)
        output = self._query_gpt(formatted_prompt, p_id)
        #print(formatted_prompt)

        self.cache[p_id] = output
        
        item.update({'cps_n': len(output),      
                    'cps': output,
                    })
        return item

    def generate_prompts_parallel(self, data):

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.generate_prompt, data))

        return results

def main():
    pass

if __name__ == "__main__":
    main()
