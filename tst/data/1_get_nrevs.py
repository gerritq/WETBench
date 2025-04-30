# WE MAKE USE OF THE CODE FROM THE OG WNC PAPER: https://github.com/rpryzant/neutralizing-bias
# ALL CREDIT TO THEM

import json
import re
import regex as rex
import sys
from datetime import datetime
from utils import save_jsonl

lang = sys.argv[1]

in_file = f"/scratch_tmp/prj/inf_nlg_ai_detection/wikidumps/dumps/{lang}wiki-20250201-stub-meta-history.xml"
#in_file = f"/scratch_tmp/prj/inf_nlg_ai_detection/wikidumps/dumps/example.xml"
out_file = f"../../data/{lang}/processing/1_{lang}_nrevs.jsonl"

if lang == 'en':
    INVALID_RE = r"\b(bot)\b"
    NPOV_RE = r"((?:^|[\W_])(?:n?pov)(?![\w]))|(?:neutral)"
    drop_words = ["atom", "atoms", "ion", "ions", "molecule", "molecules", "ph", "proton", "protons", "neutron", "neutrons", "electron", "electrons", "net neutrality"]

if lang == 'pt':
    INVALID_RE = r"\b(bot|robô|robôs)\b"
    NPOV_RE = r"((?:^|[\W_])(?:n?pov|pdi|pdvn)(?![\w]))|(?:imparci\w+|parcialidade)|(wp:parcial)|(?:neutr[ao]\w*)"
    drop_words = ["átomo", "átomos", "ião", "iões", "molécula", "moléculas", "ph", "próton", "prótons", "protões", "neutrões", "nêutron", "nêutrons"]

if lang == 'vi':
    INVALID_RE = r"\b(bot)\b"
    #NPOV_RE = r"((?:^|[\W_])(?:n?pov|trunglap|tđtl|tdtl)(?![\w]))|(?:trung lập)"
    vi_templates = [
        "unbalanced", # OK
        "phát biểu quan điểm",  # statement of opinion OK
        "quan điểm người hâm mộ",  # fan opinion
        "tâng bốc",  # flattery OK
        "thiên lệch",  # bias OK
        # "quá ít quan điểm",  # too few views (mentions POV but seems more factual)
        "systemic bias",
        "có xung đột lợi ích",  # conflict of interest OK
        #"dọn dẹp văn phong báo chí",  # journalistic style
        # "tầm nhìn hẹp",  # narrow vision, does not represent global view
        #"văn phong"  # style not encyclopaedic (this is more formality than bias)
        ]
    keywords_pattern = "|".join(vi_templates)
    NPOV_RE = r"((?:^|[\P{L}_])(?:n?pov|trunglap|tđtl|tdtl)(?![\p{L}]))|(?:trung lập)" + rf"|(?:{keywords_pattern})"
    drop_words = ["nguyên tử", "các nguyên tử", "ion", "các ion", "phân tử", "các phân tử", "ph", "proton", "các proton", "nơtron", "các nơtron"]

class Revision:
    def __init__(self):
        self.revid = None
        self.comment = None
        self.timestamp = None
        self.title = None
        self.pageid = None
        self.ns = None
        #self.is_redirect = False
        
        self.INVALID_REV_RE = re.compile(rf"{INVALID_RE}", re.UNICODE)
        self.NPOV_RE = rex.compile(rf"{NPOV_RE}", re.UNICODE)
        print(self.NPOV_RE)
        self.drop_words = set(drop_words)
    
    def incomplete(self):
        return (not self.revid or 
                not self.comment or
                not self.timestamp or 
                not self.title or
                not self.pageid or
                not self.ns) # self.is_redirect
    
    def is_npov(self):
        
        c_lower = self.comment.lower()
        if self.INVALID_REV_RE.search(c_lower):
            return False

        match = self.NPOV_RE.search(c_lower)
        if match:
            if any(word in set(c_lower.split()) for word in self.drop_words):
                return False

            return True
        
        return False

def main():
    
    out=[]
    with open(in_file, encoding='utf-8') as in_f:
        prev_line = None
        for line in in_f:

            line = line.lower().strip()
            
            if line == '<page>':
                current_page_title = None
                current_page_id = None
            
            if '<title>' in line:
                current_page_title = re.sub(r'</?[\w]+>', '', line).strip()

            if '<ns>' in line:
                ns = re.sub(r'</?[\w]+>', '', line).strip()
                if ns != '0':
                    current_page_ns = None
                else:
                    current_page_ns = ns
    
            # page id
            if '<id>' in line and '<ns>' in prev_line:
                current_page_id = re.sub(r'</?[\w]+>', '', line)
                
            # if '<redirect' in line:
            #     cur_page_is_redirect = True

            # revision id
            if '<id>' in line and '<revision>' in prev_line:
                current_revision = Revision()
                current_revision.revid = (re.sub(r'</?[\w]+>', '', line))
            
            # timestamp
            if '<timestamp>' in line:
                current_revision.timestamp = (re.sub(r'</?[\w]+>', '', line))
                
            # comment
            if '<comment>' in line:
                current_revision.comment = re.sub(r'</?[\w]+>', '', line)
            
            if line == '</revision>':
                current_revision.title = current_page_title
                current_revision.pageid = current_page_id
                current_revision.ns = current_page_ns
                # current_revision.is_redirect = cur_page_is_redirect

                if not current_revision.incomplete() and current_revision.is_npov():
                    out.append({'title': current_revision.title,
                                'pageid': current_revision.pageid,
                                'revid': current_revision.revid,
                                'timestamp': current_revision.timestamp,
                                'comment': current_revision.comment})

                
            prev_line = line
            
    save_jsonl(out, out_file)

if __name__ == "__main__":
    main()
