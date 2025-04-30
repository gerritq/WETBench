import requests
import json
import random
import argparse
import os
import threading
import queue
import time
from utils import load_jsonl, save_jsonl

stub_tags = {
    'en': '<a href="/wiki/Wikipedia:Stub" title="Wikipedia:Stub">stub</a>',
    'pt': '<a href="/wiki/Wikip%C3%A9dia:Esbo%C3%A7o" title="Wikipédia:Esboço">esboço</a>',
    'vi': '<a href="/wiki/Wikipedia:B%C3%A0i_s%C6%A1_khai" title="Wikipedia:Bài sơ khai">sơ khai</a>'
}

class WikiHtmlFetcher:

    def __init__(self, lang, in_file, out_file, total_count, thread_count=6):
        self.lang = lang
        self.in_file = in_file
        self.out_file = out_file
        self.total_count = total_count
        self.thread_count = thread_count
    
        self.stub_name = stub_tags[lang]

        self.current_items = 0
        self.processed_count = 0
        self.return_count = {}
        # locking for save access
        self.lock = threading.Lock()
        
        try:
            existing_data = load_jsonl(self.out_file)
            self.current_items = len(existing_data)
            print(f'Existing items {self.current_items}')
            self.last_idx = max(x['idx'] for x in existing_data)
            del existing_data
            # more efficient to get remaining items
            self.remaining_target = self.total_count - self.current_items
        except FileNotFoundError:
            self.last_idx = 0
            self.remaining_target = self.total_count
            print('No existing data, initializing with idx 0', flush=True)

        self.data = load_jsonl(self.in_file)
        random.seed(2025)
        random.shuffle(self.data)
        for i, x in enumerate(self.data):
            x['idx'] = i

        print('Data length', len(self.data), flush=True)
        print(f'Last idx {self.last_idx}', flush=True)

        # queue
        self.data = self.data[self.last_idx:]

        self.queue = queue.Queue()
        for item in self.data:
            self.queue.put(item)
        
        #stopping
        self.stop_event = threading.Event()

    def get_html(self, revid, session):
        # this is taken from the mediawiki api!
        URL = f"https://{self.lang}.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "parse",
            "format": "json",
            "oldid": revid,
        }
        try:
            response = session.get(URL, params=PARAMS)
            data = response.json()
        except Exception as e:
            print(e, flush=True)
            data = None

        if data and 'parse' in data and 'text' in data['parse']:
            return data['parse']['text']['*']
        else:
            return None

    def worker(self):
        '''Define a single worker --- each gets its own session'''
        session = requests.Session()
        
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                with self.lock:
                    self.processed_count += 1
                    if self.processed_count % 250 == 0:
                        print('Sleeping ...', flush=True)
                        time.sleep(15)
            except queue.Empty:
                break

            try: 
                html = self.get_html(item['revid'], session)
                
                if html is None:
                    with self.lock:
                        self.return_count['None'] = self.return_count.get('None', 0) + 1
                    continue

                if self.stub_name in html:
                    with self.lock:
                        self.return_count['Stub'] = self.return_count.get('Stub', 0) + 1
                    continue

                item['html'] = html

                with self.lock:
                    self.current_items+=1
                    with open(self.out_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    if self.current_items >= self.total_count:
                        self.stop_event.set()
                        print(f"Target reached {self.current_items} pages", flush=True)
                    
                        while not self.queue.empty():
                            try:
                                self.queue.get_nowait()
                                self.queue.task_done()
                            except queue.Empty:
                                break
                if self.current_items % 250 == 0:
                    print(f"{time.ctime()} Obtained {self.current_items} pages; processed {self.processed_count}", flush=True)
            finally:
                self.queue.task_done()

    def run(self):

        print(f'\nSTART {time.ctime()}')

        threads = []
        for _ in range(self.thread_count):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            threads.append(t)

        while not self.stop_event.is_set():
            time.sleep(1)

        time.sleep(5)

        print(f"Completed with {self.current_items} items", flush=True)

        stats_file = self.out_file.replace('.jsonl', '_stats.jsonl')
        existing_stats = {}
        try:
            stats_data = load_jsonl(stats_file)
            if stats_data and len(stats_data) > 0:
                existing_stats = stats_data[0]
        except FileNotFoundError:
            pass
            
        for key, value in self.return_count.items():
                if key in existing_stats:
                    existing_stats[key] += value
                else:
                    existing_stats[key] = value
        existing_stats['last_idx'] = self.last_idx
        existing_stats['current_items'] = self.current_items
        if 'processed_count' in existing_stats:
            existing_stats['processed_count'] += self.processed_count
        else:
            existing_stats['processed_count'] = self.processed_count

        save_jsonl([existing_stats], stats_file)

        print(f'\nEND {time.ctime()}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--in_file")
    parser.add_argument("--out_file")
    parser.add_argument("--total_count", type=int)
    args = parser.parse_args()

    fetcher = WikiHtmlFetcher(args.lang, args.in_file, args.out_file, args.total_count)
    fetcher.run()
