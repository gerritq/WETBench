import json
import os
import argparse
import concurrent.futures
import litellm
from utils import load_jsonl
import copy
from diskcache import Cache
import hashlib

os.environ['OPENAI_API_KEY'] = ''

class MGTGenerator:
    def __init__(self, 
                 lang,
                 ds,
                 in_file,
                 out_file,
                 prompt_dir,
                 prompt_techs,
                 subset = '',
                 total_n=None,
                 few_shots_file = None,
                 model="gpt-4.0-mini",
                 num_threads=6):
        self.lang = lang
        self.ds = ds
        self.subset = subset
        self.in_file = in_file
        self.out_file = out_file
        self.total_n = total_n
        self.few_shots_file = few_shots_file
        self.shots = None
        self.n_shots = None
        self.prompt_dir = prompt_dir
        self.prompt_techs = prompt_techs
        self.current_prompt_tech = None
        self.num_threads = num_threads
        self.model = model
        self.templates = {}
        
        # Caching for translations
        dir_to_disk = os.path.join('/scratch_tmp/prj/inf_nlg_ai_detection/.mgt_cache', f".{self.ds}_cache_{self.lang}")
        self.cache = Cache(dir_to_disk)
        print(f"Cache directory for {self.lang}: {dir_to_disk}")
        print(f"Cache contains {len(self.cache)} items")

        # load templates
        for prompt_tech in self.prompt_techs:
            self.templates[prompt_tech] = self._load_template(prompt_tech)

        if self.ds == 'paras':
                self._format_for_dataset = self._format_paras
                self._prepare_output_item = self._prepare_output_item_paras
                self._add_translations = self._add_translations_paras
        elif self.ds == 'sums':
            self._format_for_dataset = self._format_sums
            self._prepare_output_item = self._prepare_output_item_sums
            self._add_translations = self._add_translations_sums
        elif self.ds == 'tst':
            self._format_for_dataset = self._format_tst
            self._prepare_output_item = self._prepare_output_item_tst
            self._add_translations = self._add_translations_tst


    def _load_data(self, in_file):
        data = load_jsonl(in_file)
        n_per_tertile = self.total_n // 3

        data_by_tertiles = {'low': [], 'medium': [], 'high': []}

        for item in data:
            data_by_tertiles[item['word_tertile']].append(item)

        return (data_by_tertiles['low'][:n_per_tertile] + 
                data_by_tertiles['medium'][:n_per_tertile] + 
                data_by_tertiles['high'][:n_per_tertile])

    def _load_template(self, prompt_tech):
        if self.ds == 'paras':
            with open(os.path.join(self.prompt_dir, self.subset + '_' + prompt_tech + '_' + self.lang + '.txt'), 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            with open(os.path.join(self.prompt_dir, (prompt_tech if not prompt_tech.startswith('few') else prompt_tech[:-1]) + ('_' + self.subset if self.subset and self.lang == 'en' else '') + '_' + self.lang + '.txt'), 'r', encoding='utf-8') as f:
                return f.read().strip()
    
    def _load_few_shot_examples(self):
        if not self.few_shots_file:
            raise TypeError('No fewshots file provided')
        return load_jsonl(self.few_shots_file)

    def _format_paras(self, template, prompt_tech, item):
        item_new = copy.deepcopy(item)
        item_new['trgt_n_toks'] -= 10  # we often see gens that are to long (10 for tokens + 5 in general)
        item_new['section_title'] = item_new['section_title'].replace('=', '').strip()
        
        if prompt_tech in ['rag', 'cp']:
            item_new['content_prompts'] = "\n- " + "\n- ".join(item_new['cps'])
            
            if prompt_tech == 'rag':
                context_word = "Context" if self.lang == "en" else "Ngữ cảnh" if self.lang == "vi" else "Contexto" if self.lang == "pt" else "Context"
                context = set([x[0] for x in item_new['context']])  # obtain context without scores
                if not context:
                    raise Exception("No context")  
                item_new['context'] = f'\n{context_word} ' + f'\n\n{context_word} '.join(f'{i}: "[...] {chunk} [...]"' for i, chunk in enumerate(context, 1))

                # trgt lang for rag
                language = {'en': 'English', 'vi': 'tiếng Việt', 'pt': 'português'}
                item_new['language'] = language[self.lang]    
        
        prompt = template.format(**item_new)
        print('\n', prompt)
        messages = [{"role": "user", "content": prompt}]
        
        return messages, prompt


    def _format_sums(self, template, prompt_tech, item):
        item_new = copy.deepcopy(item)
        
        # Combine table and body
        if item_new['infobox']:
            item_new['src'] = item_new['infobox'] + "\n\n" + item_new['src']
        
        # Formatting few shots
        if prompt_tech.startswith('few'):
            messages = []
            for shot in self.shots[:self.n_shots]:
                shot_new = copy.deepcopy(shot)
                shot_new['src'] = (shot_new['infobox'] + "\n\n" + shot_new['src']) if shot_new['infobox'] else shot_new['src']
                messages.append({"role": "user", "content": template.format(**shot_new)})
                messages.append({"role": "assistant", "content": shot_new["trgt"]})
            
            # Actual query
            prompt = template.format(**item_new)
            messages.append({"role": "user", "content": prompt})
            
            assert (self.n_shots*2)+1 == len(messages)
        else:
            prompt = template.format(**item_new)
            messages = [{"role": "user", "content": prompt}]
            
        return messages, prompt

    def _format_tst(self, template, prompt_tech, item):
        item_new = copy.deepcopy(item)
        item_new['trgt_n_words'] = len(item_new['trgt'].strip().split()) - 10
        
        # Formatting few shots
        if prompt_tech.startswith('few'):
            messages = []
            for shot in self.shots[:self.n_shots]:
                shot_new = copy.deepcopy(shot)
                shot_new['trgt_n_words'] = len(shot_new['trgt'].strip().split()) - 10
                messages.append({"role": "user", "content": template.format(**shot_new)})
                messages.append({"role": "assistant", "content": shot_new["trgt"]})
            
            prompt = template.format(**item_new)
            messages.append({"role": "user", "content": prompt})
            
            assert (self.n_shots*2)+1 == len(messages), f"{(self.n_shots*2)+1} {len(messages)}"
            
            for i, m in enumerate(messages):
                print('\n\n---------------------------------------')
                print('-------------PROMPTING MESSAGE-------------', i)
                print(m['role'], '\n', flush=True)
                print(m['content'], '\n', flush=True)
        else:
            prompt = template.format(**item_new)
            messages = [{"role": "user", "content": prompt}]
            
        return messages, prompt

    def _prepare_output_item_paras(self, item):
        return {
            'id': item['id'],
            'revid': item['revid'],
            'section_title': item['section_title'],
            'trgt': item['trgt'],
            'trgt_first': item['trgt_first'],
            'page_title': item['page_title'],
            'word_tertile': item['word_tertile'],
            'cps': item['cps']
        }

    def _prepare_output_item_sums(self, item):
        return {
            'id': item['id'],
            'revid': item['revid'],
            'src_inf': (item['infobox'] + "\n\n" + item['src']) if item['infobox'] else item['src'],
            'src': item['src'],
            'page_title': item['page_title'],
            'word_tertile': item['word_tertile'],
            'trgt': item['trgt']
        }

    def _prepare_output_item_tst(self, item):
        return {
            'id': item['id'],
            'revid': item['revid'],
            'src': item['src'],
            'trgt': item['trgt']
        }

    # Add translations for each dataset
    def _add_translations_paras(self, out_item, item, mgt, mgt_trans):
        trgt_hash = f"trgt_{item['id']}"
        if trgt_hash in self.cache:
            print(f'Found cached src for paras id {trgt_hash}', flush=True)
            trgt_trans = self.cache[trgt_hash]
        else:
            trgt_trans = self._translate(item['trgt'])
            self.cache[trgt_hash] = trgt_trans
        
        out_item.update({'trgt_trans': trgt_trans, f'mgt_{self.current_prompt_tech}_trans': mgt_trans})

    def _add_translations_sums(self, out_item, item, mgt, mgt_trans):
        # Cache for src
        src_hash = f"src_{item['id']}"
        if src_hash in self.cache:
            print(f'Found cached src for sums id {src_hash}', flush=True)
            src_inf_trans = self.cache[src_hash]
        else:
            src_inf_trans = self._translate((item['infobox'] + "\n\n" + item['src']) if item['infobox'] else item['src'])
            self.cache[src_hash] = src_inf_trans
        
        out_item.update({f'mgt_{self.current_prompt_tech}_trans': mgt_trans, 'src_inf_trans': src_inf_trans})

    def _add_translations_tst(self, out_item, item, mgt, mgt_trans):
        # TST doesn't need translations currently
        pass

    def _get_mgt(self, template, prompt_tech, item):        
        messages, prompt = self._format_for_dataset(template, prompt_tech, item)
    
        out_item = self._prepare_output_item(item)
        
        # use cache if we can
        should_use_cache = (
            (self.ds == 'paras') or 
            (self.ds == 'sums' and not prompt_tech.startswith('few')) or
            (self.ds == 'tst' and not prompt_tech.startswith('few'))
        )
        
        if should_use_cache:
            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            if prompt_hash in self.cache:
                print(f'Prompt cache found for item id {item["id"]}')
                mgt = self.cache[prompt_hash]
            else:
                mgt = self._query_llm(messages)
                self.cache[prompt_hash] = mgt
        else:
            mgt = self._query_llm(messages)
        
        out_item.update({
            f'mgt_{prompt_tech}': mgt,
            'prompt': prompt
        })
        
        # Add translations if needed
        if self.lang != 'en':
            # Skip translation for TST dataset
            if self.ds != 'tst':
                mgt_trans = self._translate(mgt)
                self._add_translations(out_item, item, mgt, mgt_trans)
                
        return out_item
        
    def _query_llm(self, messages):
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _translate(self, text):
        language_map = {'en': 'English', 'vi': 'Vietnamese', 'pt': 'Portuguese'}
        lang_name = language_map.get(self.lang, 'English')

        prompt = f"Please translate the following text from {lang_name} to English. Output only the translated text. Text:\n\n{text}"

        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def run(self):
        if self.ds == 'tst':
            data = load_jsonl(self.in_file)
        else:
            data = self._load_data(self.in_file)

        # quick fix for cases have no context bc of a coding error earlier
        if self.ds == "paras":
            data = [item for item in data if not ('context' in item and not item['context'])]

        print('Data size:', len(data))
        
        for prompt_tech in self.prompt_techs:
            print(f'Running MGT LANG {self.lang} N {self.total_n} PROMPTING {prompt_tech} in {self.prompt_techs}')
            self.current_prompt_tech = prompt_tech
            out_file_mod = self.out_file.replace('.jsonl', f'_{prompt_tech}.jsonl')
            
            if prompt_tech.startswith('few'):
                self.shots = self._load_few_shot_examples()
                self.n_shots = int(prompt_tech[-1])  # checked above whether valid

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                results = list(executor.map(
                    lambda item: self._get_mgt(self.templates[prompt_tech], prompt_tech, item), 
                    data
                ))

            with open(out_file_mod, 'w', encoding='utf-8') as f:
                for out_item in results:
                    f.write(json.dumps(out_item, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--ds', type=str, required=True)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--few_shots_file', type=str)
    parser.add_argument('--prompt_dir', type=str, required=True)
    parser.add_argument('--total_n', type=int)
    parser.add_argument('--prompt_techs', nargs='+', required=True)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--num_threads', type=int, default=6)
    
    args = parser.parse_args()
    
    generator = MGTGenerator(
        lang=args.lang,
        ds=args.ds,
        subset=args.subset,
        in_file=args.in_file,
        out_file=args.out_file,
        prompt_dir=args.prompt_dir,
        total_n=args.total_n,
        few_shots_file=args.few_shots_file,
        prompt_techs=args.prompt_techs,
        model=args.model,
        num_threads=args.num_threads
    )
    generator.run()
