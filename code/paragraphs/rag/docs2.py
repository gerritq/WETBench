import os
import regex as rex
import httpx
import unidecode
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import unicodedata
from trafilatura import extract
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
import time
import tempfile
from diskcache import Cache

class WebContentLoader:
    def __init__(self, lang, max_workers=8):

        self.lang = lang
        self.max_workers = max_workers
        self.user_agent = UserAgent().random
        self.header = self.get_headers()
        self.session = httpx.Client(verify=False)

        dir_to_disk = os.path.join("/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.meta_cache", f".docs_cache_{self.lang}")
        self.cache = Cache(dir_to_disk)
        print(f"Cache directory for {self.lang}: {dir_to_disk}")
        print(f"Cache contains {len(self.cache)} items")

    def get_headers(self):
        """
        Returns language-specific HTTP headers.
        """
        if self.lang == "en":
            accept_language = "en-US,en;q=0.5"
        elif self.lang == "pt":
            accept_language = "pt-BR,pt;q=0.5"
        elif self.lang == "vi":
            accept_language = "vi-VN,vi;q=0.5"

        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": accept_language,
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    @staticmethod
    def _minimal_cleaning(text):
        """Performs minimal text cleaning."""
        text = unidecode.unidecode(text)
        text = rex.sub(r'\n{3,}', '\n', text)
        text = rex.sub(r'\s{2,}', ' ', text)
        text = rex.sub(r'https?://\S+|www\.\S+', '', text) 
        text = rex.sub(r'\.{3,}', '', text)
        text = rex.sub(r'[\.\s\.]{3,}', '', text)
        return text

    @staticmethod
    def _fix_url(url):
        if url.startswith("//"):
            url = "https:" + url
        elif not url.startswith("http"):
            url = "https://" + url
        return url
    
    @staticmethod
    def _is_pdf(response):
        if str(response.url).endswith('.pdf'):
            return True
        try:
            return response.headers.get("Content-Type") == "application/pdf"
        except httpx.RequestError:
            return False

    def _parse_pdf(self, response):
        try:
            # Delete temp file if it exists
            # if os.path.exists(self.temp_file):
            #     os.remove(self.temp_file)

            # Save temp file
            # with open(self.temp_file, 'wb') as f:
            #     f.write(response.content)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_filename = temp_pdf.name
                temp_pdf.write(response.content)

            reader = PdfReader(temp_filename, strict=False)
            n_pages = min(len(reader.pages), 100)  # Limit to 200 pages
            
            text = ' '.join([reader.pages[i].extract_text() for i in range(n_pages) if reader.pages[i].extract_text()])
            return text.strip()

        except Exception as e:
            print(f"    PDF error {response.url}: {e}", flush=True)
            return None
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def _parse_webpage(self, response):
        try:
            article_text = extract(
                response.text, # content used in STORM?
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )

            return article_text

        except Exception as e:
            print(f"    HTML error {response.url}: {e}", flush=True)
            return None

    @staticmethod
    def _is_valid_doc(text):
        """Checks if the extracted document is valid."""
        return (text and 
                150 <= len(text) <= 75000 and # latter is 12-15k words or about 45-60 pages
                "Attention Required! | Cloudflare" not in text and 
                "Just a moment...Enable JavaScript and cookies to continue" not in text and 
                "Please enable JavaScript" not in text and 
                "503 Service Unavailable" not in text and
                "Page Not Found: 404 Not Found" not in text)

    def _fetch_single_url(self, url):
        """Fetches a single URL, determines type (PDF or webpage), and extracts content. Credits to STORM
        
        2.0: return URL only, retrieving that gets files from cache"""
        
        if url in self.cache:
            print(f'Found cached docs for {url}', flush=True)
            cache_result = self.cache[url]
            return url if cache_result !='FAILED' else None
        
        try:
            time.sleep(1)
            print('Processing', url, flush=True)
            url = self._fix_url(url)
            response = self.session.get(url, headers=self.header, timeout=10, follow_redirects=True)
            
            if response.status_code >= 429:
                print('Sleeping as rate limits were exceeded ...', flush=True)
                time.sleep(30)
            if response.status_code >= 400:
                response.raise_for_status()
        
        except Exception as exc:
            print(f"Error while fetching {url} - {exc!r}", flush=True)
            self.cache[url]= "FAILED"
            return None

        if self._is_pdf(response):
            text=None
            print('Found PDF but skipping ...')
            # text = self._parse_pdf(response)
        else:
            text = self._parse_webpage(response)

        if self._is_valid_doc(text):
            result = {'page_content': self._minimal_cleaning(text),
                      'metadata': {'url': url}}
            self.cache[url] = result
            return url
        else:
            self.cache[url]= "FAILED"
            return None

    def fetch_content(self, data):        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            n=len(data)
            out=[]
            for i, item in enumerate(data, start=1):
                print(f'Item {i}/{n}', flush=True)
                item_new = item.copy()
                results = list(executor.map(self._fetch_single_url, item['urls']))
                # drop None
                results = [x for x in results if x]
                item_new.update({'n_docs': len(results),
                                 'docs_urls': results})
                out.append(item_new)

                if i % 100 == 0:
                    print('Sleeping to reduce bandwidth ..', flush=True)
                    time.sleep(10)
                    
            
        return out

def main():
    pass

if __name__ == "__main__":
    main()
