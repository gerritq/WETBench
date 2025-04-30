import json
import re
import sys
from datetime import datetime
from utils import save_jsonl

class Article:
    def __init__(self):
        self.revid = []
        self.timestamp = []
        self.title = None
        self.pageid = None
        self.ns = None
        self.is_redirect = False
    
    def incomplete(self):
        return (not self.revid or 
                not self.timestamp or 
                not self.title or 
                not self.ns or
                not self.pageid or
                self.is_redirect) or not (len(self.revid) == len(self.timestamp))

def main():
    
    lang = sys.argv[1]
    in_file = f"/scratch_tmp/prj/inf_nlg_ai_detection/wikidumps/dumps/{lang}wiki-20250201-stub-meta-history.xml"
    out_file = f"../../data/{lang}/processing/1_{lang}_latest_articles.jsonl"
    
    threshold = datetime.strptime('2022-11-30T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    
    out=[]
    with open(in_file, encoding='utf-8') as in_f:
        prev_line = None
        for line in in_f:

            line = line.strip()
            
            if line == '<page>':
                cur_page = Article()
            
            if '<title>' in line:
                cur_page.title = re.sub(r'</?[\w]+>', '', line).strip()

            if '<ns>' in line:
                ns = re.sub(r'</?[\w]+>', '', line).strip()
                if ns != '0':
                    cur_page.ns = None
                else:
                    cur_page.ns = ns
    
            # page id
            if '<id>' in line and '<ns>' in prev_line:
                cur_page.pageid = re.sub(r'</?[\w]+>', '', line)
                
            if '<redirect' in line:
                cur_page.is_redirect = True

            # revision id
            if '<id>' in line and '<revision>' in prev_line:
                cur_page.revid.append(re.sub(r'</?[\w]+>', '', line))

            if '<timestamp>' in line:
                cur_page.timestamp.append(re.sub(r'</?[\w]+>', '', line))
                
            if line == '</page>':
                assert len(cur_page.revid) == len(cur_page.timestamp), f"{len(cur_page.revid)} {len(cur_page.timestamp)}"
                if not cur_page.incomplete():
                    for i in range(len(cur_page.revid) - 1, -1, -1):
                        rev_date = datetime.strptime(cur_page.timestamp[i], '%Y-%m-%dT%H:%M:%SZ')
                        if rev_date < threshold:
                            out.append({'title': cur_page.title,
                                        'pageid': cur_page.pageid,
                                        'revid': cur_page.revid[i],
                                        'timestamp': cur_page.timestamp[i]})
                            break
                    
                cur_page = Article()
                
            prev_line = line
            
    save_jsonl(out, out_file)

if __name__ == "__main__":
    main()
