import os
import litellm
from utils import load_jsonl, save_jsonl
import copy
import argparse
import concurrent.futures
import random

os.environ['OPENAI_API_KEY'] = ''
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

class MachineTextGeneratorAPI:

    def __init__(self,
                 total_n,
                 lang,
                 task,
                 in_file,
                 out_file,
                 model_name,
                 prompt_template_file,
                 n_shots=None,
                 few_shots_file=None,
                 num_threads=6):
        self.total_n = total_n
        self.lang = lang
        self.task = task
        self.in_file = in_file
        self.out_file = out_file
        self.model_name = model_name
        self.prompt_template_file = prompt_template_file
        self.few_shots_file = few_shots_file

        self.num_threads = num_threads

        self.n_shots = n_shots
        self.template = self._load_template()
        if self.task not in ['first', 'extend', 'external']:
            self.shots = self._load_few_shot_examples()

        if self.model_name not in ['gpt-4o-mini', 'gemini/gemini-2.0-flash']
            raise ValueError(f"'{self.model_name}' not supported")

    def load_data(self):
        data = load_jsonl(self.in_file)

        if self.task == 'external':
            return data[:self.total_n]
        else:
            n_per_tertile = self.total_n // 3

            data_by_tertiles = {'low': [], 'medium': [], 'high': []}

            for item in data:
                data_by_tertiles[item['word_tertile']].append(item)

            return (data_by_tertiles['low'][:n_per_tertile] + 
                    data_by_tertiles['medium'][:n_per_tertile] + 
                    data_by_tertiles['high'][:n_per_tertile])


    def _load_few_shot_examples(self):
        if not self.few_shots_file:
            raise TypeError('No fewshots file provided:= ...')
        return load_jsonl(self.few_shots_file)

    def _load_template(self):
        with open(self.prompt_template_file, 'r', encoding='utf-8') as f:
            return f.read().strip()


    def _format_paras(self, item):
        item_new = copy.deepcopy(item)
        item_new['content_prompts'] = "\n- " + "\n- ".join(item_new['cps'])
        item_new['trgt_n_toks'] -= 10  # we often see gens that are to long (10 for tokens + 5 in general)
        item_new['section_title'] = item_new['section_title'].replace('=', '').strip()
        
        context_word = "Context" if self.lang == "en" else "Ngữ cảnh" if self.lang == "vi" else "Contexto" if self.lang == "pt" else "Context"
        if not item_new['context']:
            print(f'No context for {item_new["revid"]}') 
            return None
        context = set([x[0] for x in item_new['context']])  # obtain context without scores
        item_new['context'] = f'\n{context_word} ' + f'\n\n{context_word} '.join(f'{i}: "[...] {chunk} [...]"' for i, chunk in enumerate(context, 1))

        # trgt lang for rag
        language = {'en': 'English', 'vi': 'tiếng Việt', 'pt': 'português'}
        item_new['language'] = language[self.lang]    
        
        prompt = self.template.format(**item_new)
        messages = [{"role": "user", "content": prompt}]
        
        return messages


    def _format_sums(self, item):
        item_new = copy.deepcopy(item)
        
        # Combine table and body
        if item_new['infobox']:
            item_new['src'] = item_new['infobox'] + "\n\n" + item_new['src']
        
        # Formatting few shots
        messages = []
        for shot in self.shots[:self.n_shots]:
            shot_new = copy.deepcopy(shot)
            shot_new['src'] = (shot_new['infobox'] + "\n\n" + shot_new['src']) if shot_new['infobox'] else shot_new['src']
            messages.append({"role": "user", "content": self.template.format(**shot_new)})
            messages.append({"role": "assistant", "content": shot_new["trgt"]})
        
        # Actual query
        prompt = self.template.format(**item_new)
        messages.append({"role": "user", "content": prompt})
        
        assert (self.n_shots*2)+1 == len(messages)
        return messages

    def _format_tst(self, item):
        item_new = copy.deepcopy(item)
        item_new['trgt_n_words'] = len(item_new['trgt'].strip().split()) - 10
        
        # Formatting few shots
        messages = []
        for shot in self.shots[:self.n_shots]:
            shot_new = copy.deepcopy(shot)
            shot_new['trgt_n_words'] = len(shot_new['trgt'].strip().split()) - 10
            messages.append({"role": "user", "content": self.template.format(**shot_new)})
            messages.append({"role": "assistant", "content": shot_new["trgt"]})
        
        prompt = self.template.format(**item_new)
        messages.append({"role": "user", "content": prompt})
        
        assert (self.n_shots*2)+1 == len(messages), f"{(self.n_shots*2)+1} {len(messages)}"
        
        # for i, m in enumerate(messages):
        #     print('\n\n---------------------------------------')
        #     print('-------------PROMPTING MESSAGE-------------', i)
        #     print(m['role'], '\n', flush=True)
        #     print(m['content'], '\n', flush=True)
            
        return messages

    def _format_external(self, item):
        item_new = copy.deepcopy(item)
        prompt = self.template.format(**item_new)
        messages= [{"role": "user", "content": prompt}]
        print(prompt)
        return messages

    def _query_llm(self, messages):
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n\n[LLM ERROR] {type(e).__name__}: {e}", flush=True)
            print(f"Failed on message {messages}", flush=True)
            return ""

    def run(self):
        if self.task != 'tst':
            data = self.load_data()
        else:
            data = load_jsonl(self.in_file)
            # filter
            data = [item for item in data if not item['drop']]
            random.seed(42)
            random.shuffle(data)
            keys_to_keep = ["title", "pageid", "revid", "timestamp", "comment", "src", "trgt"]
            data = [{k: item[k] for k in keys_to_keep if k in item} for item in data]
            data = data[:self.total_n]


        # if self.task in ['first', 'extend']:
        #     messages = [self._format_paras(item) for item in data]
        #     # filter out non
        #     x=len(messages)
        #     messages = [x for x in messages if x]
        #     print(f'Data size reduced by {x-len(messages)}')

        if self.task == 'sums':
            messages = [self._format_sums(item) for item in data]
        if self.task == 'tst':
            messages = [self._format_tst(item) for item in data]            
        if self.task in ['first', 'extend']:
            messages = [self._format_paras(item) for item in data]
            valid_indices = [i for i, msg in enumerate(messages) if msg]
            original_count = len(messages)
            data = [data[i] for i in valid_indices]
            messages = [messages[i] for i in valid_indices]
            print(f'Data size reduced from {original_count} to {len(messages)} ({original_count-len(messages)} items removed)')
        if self.task == 'external':
            messages = [self._format_external(item) for item in data]

        print('Len messages', len(messages), flush=True)
        assert len(data) == len(messages)

        print('Example message', messages[0], flush=True)
    
        all_responses = []

        print("Querying LLM ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(
                lambda item: self._query_llm(item), 
                messages
            ))

        #assert len(results) == len(data)

        for i in range(len(data)):
            data[i].update({'mgt': results[i]})

        save_jsonl(data, self.out_file)
        return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_n", type=int, default=2700)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--task", type=str, default="sums")
    #parser.add_argument("--prompt_tech", type=str, default="few3")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt_template_file", type=str, required=True)
    parser.add_argument("--few_shots_file", type=str)
    parser.add_argument("--n_shots", type=int, default=3)
    #parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    print('\nInitialising MGT generator...', flush=True)
    inference_engine = MachineTextGeneratorAPI(
        total_n=args.total_n,
        lang=args.lang,
        task=args.task,
        in_file=args.in_file,
        out_file=args.out_file,
        model_name=args.model_name,
        prompt_template_file=args.prompt_template_file,
        few_shots_file=args.few_shots_file,
        n_shots=args.n_shots,
    )

    print('\nRunning inference...', flush=True)
    inference_engine.run()

if __name__ == "__main__":
    main()