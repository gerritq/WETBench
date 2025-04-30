import os
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from itertools import chain
import time
from diskcache import Cache

# This is taken from STORM: https://github.com/stanford-oval/storm
GENERALLY_UNRELIABLE = {
    "112_Ukraine",
    "Ad_Fontes_Media",
    "AlterNet",
    "Amazon",
    "Anadolu_Agency_(controversial_topics)",
    "Ancestry.com",
    "Answers.com",
    "Antiwar.com",
    "Anti-Defamation_League",
    "arXiv",
    "Atlas_Obscura_places",
    "Bild",
    "Blaze_Media",
    "Blogger",
    "BroadwayWorld",
    "California_Globe",
    "The_Canary",
    "CelebrityNetWorth",
    "CESNUR",
    "ChatGPT",
    "CNET_(November_2022\u2013present)",
    "CoinDesk",
    "Consortium_News",
    "CounterPunch",
    "Correo_del_Orinoco",
    "Cracked.com",
    "Daily_Express",
    "Daily_Kos",
    "Daily_Sabah",
    "The_Daily_Wire",
    "Discogs",
    "Distractify",
    "The_Electronic_Intifada",
    "Encyclopaedia_Metallum",
    "Ethnicity_of_Celebs",
    "Facebook",
    "FamilySearch",
    "Fandom",
    "The_Federalist",
    "Find_a_Grave",
    "Findmypast",
    "Flags_of_the_World",
    "Flickr",
    "Forbes.com_contributors",
    "Fox_News_(politics_and_science)",
    "Fox_News_(talk_shows)",
    "Gawker",
    "GB_News",
    "Geni.com",
    "gnis-class",
    "gns-class",
    "GlobalSecurity.org",
    "Goodreads",
    "Guido_Fawkes",
    "Heat_Street",
    "History",
    "HuffPost_contributors",
    "IMDb",
    "Independent_Media_Center",
    "Inquisitr",
    "International_Business_Times",
    "Investopedia",
    "Jewish_Virtual_Library",
    "Joshua_Project",
    "Know_Your_Meme",
    "Land_Transport_Guru",
    "LinkedIn",
    "LiveJournal",
    "Marquis_Who's_Who",
    "Mashable_sponsored_content",
    "MEAWW",
    "Media_Bias/Fact_Check",
    "Media_Research_Center",
    "Medium",
    "metal-experience",
    "Metro",
    "The_New_American",
    "New_York_Post",
    "NGO_Monitor",
    "The_Onion",
    "Our_Campaigns",
    "PanAm_Post",
    "Patheos",
    "An_Phoblacht",
    "The_Post_Millennial",
    "arXiv",
    "bioRxiv",
    "medRxiv",
    "PeerJ Preprints",
    "Preprints.org",
    "SSRN",
    "PR_Newswire",
    "Quadrant",
    "Quillette",
    "Quora",
    "Raw_Story",
    "Reddit",
    "RedState",
    "ResearchGate",
    "Rolling_Stone_(politics_and_society,_2011\u2013present)",
    "Rolling_Stone_(Culture_Council)",
    "Scribd",
    "Scriptural_texts",
    "Simple_Flying",
    "Sixth_Tone_(politics)",
    "The_Skwawkbox",
    "SourceWatch",
    "Spirit_of_Metal",
    "Sportskeeda",
    "Stack_Exchange",
    "Stack_Overflow",
    "MathOverflow",
    "Ask_Ubuntu",
    "starsunfolded.com",
    "Statista",
    "TASS",
    "The_Truth_About_Guns",
    "TV.com",
    "TV_Tropes",
    "Twitter",
    "X.com",
    "Urban_Dictionary",
    "Venezuelanalysis",
    "VGChartz",
    "VoC",
    "Washington_Free_Beacon",
    "Weather2Travel",
    "The_Western_Journal",
    "We_Got_This_Covered",
    "WhatCulture",
    "Who's_Who_(UK)",
    "WhoSampled",
    "Wikidata",
    "WikiLeaks",
    "Wikinews",
    "Wikipedia",
    "WordPress.com",
    "Worldometer",
    "YouTube",
    "ZDNet",
}
DEPRECATED = {
    "Al_Mayadeen",
    "ANNA_News",
    "Baidu_Baike",
    "China_Global_Television_Network",
    "The_Cradle",
    "Crunchbase",
    "The_Daily_Caller",
    "Daily_Mail",
    "Daily_Star",
    "The_Epoch_Times",
    "FrontPage_Magazine",
    "The_Gateway_Pundit",
    "Global_Times",
    "The_Grayzone",
    "HispanTV",
    "Jihad_Watch",
    "Last.fm",
    "LifeSiteNews",
    "The_Mail_on_Sunday",
    "MintPress_News",
    "National_Enquirer",
    "New_Eastern_Outlook",
    "News_Break",
    "NewsBlaze",
    "News_of_the_World",
    "Newsmax",
    "NNDB",
    "Occupy_Democrats",
    "Office_of_Cuba_Broadcasting",
    "One_America_News_Network",
    "Peerage_websites",
    "Press_TV",
    "Project_Veritas",
    "Rate_Your_Music",
    "Republic_TV",
    "Royal_Central",
    "RT",
    "Sputnik",
    "The_Sun",
    "Taki's_Magazine",
    "Tasnim_News_Agency",
    "Telesur",
    "The_Unz_Review",
    "VDARE",
    "Voltaire_Network",
    "WorldNetDaily",
    "Zero_Hedge",
}
BLACKLISTED = {
    "Advameg",
    "bestgore.com",
    "Breitbart_News",
    "Centre_for_Research_on_Globalization",
    "Examiner.com",
    "Famous_Birthdays",
    "Healthline",
    "InfoWars",
    "Lenta.ru",
    "LiveLeak",
    "Lulu.com",
    "MyLife",
    "Natural_News",
    "OpIndia",
    "The_Points_Guy",
    "The_Points_Guy_(sponsored_content)",
    "Swarajya",
    "Veterans_Today",
    "ZoomInfo",
}


class URLFetcher:
    def __init__(self,
                lang, 
                api_key,
                cx_id, 
                max_workers=1):

        self.api_key = api_key
        self.cx_id = cx_id
        self.max_workers = max_workers
        self.exclude_pages = [
                'reddit.com', 'facebook.com', 'instagram.com', 'x.com',
                'amazon.com', 'pinterest.com', 'tiktok.com', 'youtube.com',
                'wikipedia.org', 'quora.com', 'wikihow.com', 'linkedin.com',
                'twitter.com'
            ]
        self.lang = lang
        self.lang_params = {
            "en": {"gl": "us", "hl": "en"},
            "pt": {"gl": "br", "hl": "pt-BR"},
            "vi": {"gl": "vn", "hl": "vi"}
            }
        self.max_requests = 0

        dir_to_disk = os.path.join("/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.meta_cache", f".urls_cache_{self.lang}")
        self.cache = Cache(dir_to_disk)
        print(f"Cache directory for {self.lang}: {dir_to_disk}")
        print(f"Cache contains {len(self.cache)} items")

        # Print api key to confirm we use the right prohect
        print("API:", 'rag-search-2' if self.api_key == "" else "rag-search-1")
    @staticmethod
    def _is_valid_wikipedia_source(url):
        """Copy pasted from STORM"""
        parsed_url = urlparse(url)
        # Check if the URL is from a reliable domain
        combined_set = GENERALLY_UNRELIABLE | DEPRECATED | BLACKLISTED
        for domain in combined_set:
            if domain.lower() in parsed_url.netloc.lower():
                return False
        return True

    def _google_api_search(self, query):
        # chekc cache first
        if query in self.cache:
            print(f'Found cached urls for {query}', flush=True)
            return self.cache[query][:5]

        search_url = "https://www.googleapis.com/customsearch/v1"
        lang_params = self.lang_params[self.lang]

        params = {
            "key": self.api_key,
            "cx": self.cx_id,
            "q": query,
            "num": 10,
            "siteSearch": " ".join(self.exclude_pages),
            "siteSearchFilter": "e",
        }
        params.update(lang_params)

        for attempt in range(2):
            try:
                self.max_requests +=1
                response = requests.get(search_url, params=params)

                if response.status_code == 429:
                    print(f'Sleeping as exceeded quota .... attempt {attempt} ', flush=True)
                    print(f'Response text', response.text, flush=True)
                    print(f'Response header', response.headers, flush=True)
                    time.sleep(121 if attempt == 0 else 90) # exceedes QPM
                    continue

                response.raise_for_status()
                result = response.json()
                # return {query: [entry["link"] for entry in result.get("items", [])]}
            
                if self.max_requests >=95:
                    print('Pre-emptively sleep to avoid hitting the RPM Quoata (10k per day! max) ... ', flush=True)
                    self.max_requests = 0
                    time.sleep(90)
                    
                result = [entry["link"] for entry in result.get("items", []) if self._is_valid_wikipedia_source(entry['link'])]
                self.cache[query] = result
                return result[:5]
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"API error for query '{query}': {e}", flush=True)

        raise RuntimeError(f"API request failed after 3 attempts for query '{query}'")

    def _filter_urls(self, web_urls, item):
        urls = set()
        for ref_list in item.get("refs", {}).values():
            urls.update(ref_list)

        for url in web_urls:
            if url not in urls:
                urls.add(url)

        return list(set(urls))


    def _fetch_urls_concurrently(self, item, threading=True):
        if threading:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._google_api_search, item['cps']))
        else:
            results = [self._google_api_search(query) for query in item['cps']]

        #print(item['revid'])
        web_urls = list(chain.from_iterable(results))
        return self._filter_urls(web_urls, item)


    def fetch_urls(self, data):
        n = len(data)
        out=[]
        for i, item in enumerate(data, start=1):
            print(f'Item {i}/{n}', flush=True)
            item_new = item.copy()
            urls = self._fetch_urls_concurrently(item)
            # print(urls)
            # print(item)
            assert urls, f'Returned empty URLs {item}'
            item_new.update({'n_urls': len(urls),
                        'urls': urls})
            out.append(item_new)
        
        return out

def main():
    pass

if __name__ == "__main__":
    main()