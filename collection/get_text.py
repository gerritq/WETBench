import json
import os
import regex as rex
import unicodedata
import html
import sys
from bs4 import BeautifulSoup, Tag, NavigableString
from utils import load_jsonl, save_jsonl

lang = sys.argv[1]

in_file = f"../data/{lang}/2_{lang}_html.jsonl"
out_file = f"../data/{lang}/3_{lang}_text.jsonl"


LANG_CONFIG = {
    'en': {
        'stub_name': '<a href=\"/wiki/Wikipedia:Stub\" title=\"Wikipedia:Stub\">stub</a>',
        'skip_headers': ['references', 'external links', 'see also', 'further reading',
                        'career statistics', 'notes', 'bibliography', 'filmography',
                        'other', 'discography'],
        'list_name': 'List'
    },
    'pt': {
        'stub_name': '<a href=\"/wiki/Wikip%C3%A9dia:Esbo%C3%A7o\" title=\"Wikipédia:Esboço\">esboço</a>',
        'skip_headers': ['referências', 'notas', 'notas e referências', 'bibliografia', 
                        'ligações externas', 'ver também', 'discografia', 'filmografia'],
        'list_name': 'Lista'
    },
    'vi': {
        'stub_name': '<a href=\"/wiki/Wikipedia:B%C3%A0i_s%C6%A1_khai\" title=\"Wikipedia:Bài sơ khai\">sơ khai</a>',
        'skip_headers': ['ghi chú', 'tham khảo', 'liên kết ngoài', 'danh hiệu và giải thưởng',
                        'xem thêm', 'danh sách đĩa nhạc', 'danh sách phim'],
        'list_name': 'Danh sách'
    }
}

# There should be no more stubs as our data collection does only get non-stubs
# stub_name = {'en': '<a href=\"/wiki/Wikipedia:Stub\" title=\"Wikipedia:Stub\">stub</a>',
#              'pt': '<a href=\"/wiki/Wikip%C3%A9dia:Esbo%C3%A7o\" title=\"Wikipédia:Esboço\">esboço</a>',
#              'vi': '<a href=\"/wiki/Wikipedia:B%C3%A0i_s%C6%A1_khai\" title=\"Wikipedia:Bài sơ khai\">sơ khai</a>'}
# stub_name = stub_name[lang]
 
NONE_REASON_RECORDS = {"total": 0,
                        "valid": 0,
                        "stub": {"count": 0, "titles": []},
                        "no_header": {"count": 0, "titles": []},
                        "no_lead": {"count": 0, "titles": []},
                        "no_sections": {"count": 0, "titles": []},
                        "no_refs": {"count": 0, "titles": []},
                        "list": {"count": 0, "titles": []},
                        }

def clean_text(x, rm_ref=False):

    #x = rex.sub(r'(\.")((\[\d+\])+)', r'\2\1', x)
    x = html.unescape(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.replace('//;', '')
    x = x.replace("\ufeff", '')
    x = rex.sub(r'\s{2,}', ' ', x) # rm multiple white spaces
    # clean white space between letter/chars and other chars
    x = rex.sub(r'([\p{L}\.!:;?])\s+([\.!:;?,])', r'\1\2', x) # pt revid 63231246
    # repl multiple periods with one
    x = rex.sub(r'(?<!\[)\.{2,}(?!(\.|\]))(?!\s?[?!"])', r'.', x) # pt revid 63311377; en revid 1123542998; en revid 1064596718 (allow [...])
    # rm other noise
    x = rex.sub(r'\.\s+\.', r'\.', x) # pt revid 63454804
    x = x.replace('sup] .[sup', 'sup][sup') # rm also period between sups pt revid 64501237
    x = rex.sub(r'([\(\[\{])\s+', r'\1', x)
    x = rex.sub(r'\s+([\)\]\}])', r'\1', x)
    x = rex.sub(r'\s+([.,:;!?])', r'\1', x)
    x = x.replace('Template:', '')
    x = rex.sub(r"(['\"])\s*(.*?)\s*\1", r'\1\2\1', x)
    #x = rex.sub(r'([^.!?"\']|\][^.!?"\'"])$', r'\1.', x) # in case of a blockquote or anything related

    # for lead rm [\d]
    if rm_ref:
        # extract refs as in en 1091641450
        x = rex.sub(r'\[\d+\]', '', x)

    return x

def extract_clean_text(elem, skip_geo=False):

    # NEED to exclude several spans: geo, display-none, and style tags
    #<span class="geo-inline-hidden noexcerpt">
    for child in elem.children:
        if child.name == 'span' and child.get('class') and 'geo-inline-hidden' in child.get('class'):
            return ''

    elem_copy = BeautifulSoup(str(elem), 'html.parser')
    
    # Add this after checking for geo-inline-hidden spans
    for span in elem_copy.find_all('span', attrs={'style': lambda value: value and rex.search(r"display\s*:\s*none", value)}):
        span.decompose()

    for span in elem_copy.find_all('span', attrs={'style': lambda value: value and rex.search(r"visibility\s*:\s*hidden", value)}):
        span.decompose()

    # rm all sttyle tags!
    for style_tag in elem_copy.find_all('style'):
        style_tag.decompose()

    for fig_tag in elem_copy.find_all('figcaption'):
        fig_tag.decompose()

    for br in elem_copy.find_all("br"):
        br.replace_with(" ")

    for sup in elem_copy.find_all('sup'):
        link = sup.find('a', href=True)
        if link:
            citation_ref = f"[sup:{link['href']}:sup]"
            sup.replace_with(citation_ref)

    clean_text = elem_copy.get_text()
    clean_text = ' '.join(clean_text.split())

    return clean_text

def parse_infobox(soup):

    def filter_tbody(tbody):
        if not tbody: 
            return None

        for span in tbody.find_all('span', attrs={'style': lambda value: value and rex.search(r"display\s*:\s*none", value)}):
            span.decompose()

        for span in tbody.find_all('span', attrs={'style': lambda value: value and rex.search(r"visibility\s*:\s*hidden", value)}):
            span.decompose()

        # rm all sttyle tags!
        for style_tag in tbody.find_all('style'):
            style_tag.decompose()

        # rm all sup tags
        for style_tag in tbody.find_all('sup'):
            style_tag.decompose()

        for br in tbody.find_all("br"):
            br.replace_with(" ")
        
        for fig_tag in tbody.find_all('figcaption'):
            fig_tag.decompose()

        return tbody

    rows, table_out = [], []
    # find either table.infobox or div.infobox which contains mutliple table
    table = soup.select_one("table.infobox")
    if table:
        tbody = table.find('tbody')
        if tbody:
            clean_tbody = filter_tbody(tbody)
            if clean_tbody:
                rows = clean_tbody.find_all("tr", recursive=False)
    else:
        table = soup.select_one("div.infobox")
        if table:
            tbodies = table.find_all('tbody')
            if tbodies:
                clean_tbodies = [filter_tbody(x) for x in tbodies if x]
                if clean_tbodies:
                    for clean_tbody in clean_tbodies:
                        if clean_tbody:
                            rows.extend(clean_tbody.find_all("tr", recursive=False))

    if not rows:
        return None            

    # PORTUGUESE
    if lang == 'pt':
        for row in rows:
            header_cell = row.find("th")
            info_cell = row.find_all("td", recursive=False)
            
            # likely a header
            if header_cell and not info_cell:
                txt = header_cell.get_text(separator=" ", strip=True)
                if txt: 
                    txt = f'=== {txt} ==='
                    table_out.append(txt)
                continue
            
            if header_cell and info_cell and len(info_cell)==1:
                key = header_cell.get_text(separator=" ", strip=True)
                value = info_cell[0].get_text(separator=" ", strip=True)
                if key and value:
                    txt = f"{key}: {value}"
                    table_out.append(txt)
                    continue

            # other is pair
            if len(info_cell) == 2:
                key = info_cell[0].get_text(separator=" ", strip=True)
                value = info_cell[1].get_text(separator=" ", strip=True)
                if key and value:
                    txt = f"{key}: {value}"
                    table_out.append(txt)
                    continue
                if (key and not value):
                    txt = f"{key}"
                    table_out.append(txt)
                    continue
                if (not key and value):
                    txt = f"{value}"
                    table_out.append(txt)
                    continue

            if info_cell:
                txt = " ".join(cell.get_text(separator=" ", strip=True) for cell in info_cell)
                if txt:
                    table_out.append(txt)
                continue

    # ENGLISH AND VI
    if lang in ['en', 'vi']:
        for row in rows:

            box_above = row.find("th", class_="infobox-above")
            if box_above:
                txt = box_above.get_text(separator=" ", strip=True) # eg en 1114710105
                if txt: 
                    table_out.append(txt)
                continue

            full_data = row.find("td", class_="infobox-full-data")
            if full_data:
                key_cell = full_data.find("th")
                value_cell = full_data.find("td")
                if key_cell and value_cell:
                    txt = f'{key_cell.get_text(separator=" ", strip=True)}: {value_cell.get_text(separator=" ", strip=True)}'
                    table_out.append(txt)
                else:
                    txt = full_data.get_text(separator=" ", strip=True) # eg en 1114710105
                    if txt: 
                        table_out.append(txt)
                continue
                
            header = row.find("th", class_="infobox-header")
            if header:
                txt = header.get_text(separator=" ", strip=True)
                if txt:
                    txt = f'=== {txt} ==='
                    table_out.append(txt)
                continue
                
            subheader = row.find("td", class_="infobox-subheader")
            if subheader:
                txt = f'=== {subheader.get_text(separator=" ", strip=True)} ==='
                table_out.append(txt)
                continue

            key_cell = row.find("th", scope="row")
            value_cell = row.find("td", class_="infobox-data")
            if key_cell and value_cell:
                key = key_cell.get_text(separator=" ", strip=True)
                value = value_cell.get_text(separator=" ", strip=True)
                if key and value:
                    txt = f"{key}: {value}"
                    table_out.append(txt)

                continue
               
            # This is when we have no classes as described in Wikipedia:Infobox!
            th_header =  row.find("th", recursive=False)
            if th_header:
                txt = th_header.get_text(separator=" ", strip=True)
                if txt:
                    txt = f"=== {txt} ==="
                    table_out.append(txt)
                continue

            tds = row.find_all("td", recursive=False)
            if tds and len(tds) == 2:
                key = tds[0].get_text(separator=" ", strip=True)
                value = tds[1].get_text(separator=" ", strip=True)
                if key and value:
                    txt = f"{key}: {value}"
                    table_out.append(txt)
                continue

            if tds:
                txt = " ".join(cell.get_text(separator=" ", strip=True) for cell in tds)
                if txt:
                    table_out.append(txt)
                continue
    # OUT
    manual_exclusion = ['Esta caixa: ver discutir', '[edite no Wikidata]']
    out = []
    for x in table_out:
        x_clean = clean_text(x, rm_ref=True)
        if x_clean and x_clean not in manual_exclusion:
            out.append(x_clean)

    return out



def extract_text(text, title):
    
    config = LANG_CONFIG[lang]

    if title.startswith(config['list_name']):
        NONE_REASON_RECORDS["list"]["count"] += 1
        return None

    # unicode
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Drop stubs -  there shouldnt be any left
    if config['stub_name'] in text:
        NONE_REASON_RECORDS["stub"]["count"] += 1
        #NONE_REASON_RECORDS["stub"]["titles"].append((title, timestamp))
        return None

    # define headers https://en.wikibooks.org/wiki/Editing_Wikitext/Headings
    headers = ["mw-heading mw-heading2", 
               "mw-heading mw-heading3", 
               "mw-heading mw-heading4",
               "mw-heading mw-heading5",
               "mw-heading mw-heading6"]

    data = {'infobox': None, 'lead': [], 'sections': [], 'refs': {}}
    
    soup = BeautifulSoup(text, 'lxml')

    headers_soup = soup.find_all("div", class_=headers)
    if not headers_soup:
        NONE_REASON_RECORDS["no_header"]["count"] += 1
        #NONE_REASON_RECORDS["no_header"]["titles"].append((title, timestamp))
        return None

    # INFOBOX

    
    infobox = parse_infobox(soup)
    if infobox:
        data['infobox'] = infobox
    else:
        pass

    # RM all infoboxes to not conflict lead section search with p in infobox tables
    for infobox in soup.select("table.infobox, div.infobox"):
        infobox.decompose()

    # LEAD: all paras before the first header excl those in tables
    lead_ps = [p for p in headers_soup[0].find_all_previous("p") if not p.find_parent("td")]

    # for x in headers_soup[0].find_all_previous("p"):
    #     print(x)
    #     print(x.parent.tag)
    
    ps = [extract_clean_text(p, skip_geo=True) for p in lead_ps]
    data['lead'] = list(reversed([p for p in ps if p]))
    if not data['lead']:
        NONE_REASON_RECORDS["no_lead"]["count"] += 1
        #NONE_REASON_RECORDS["no_lead"]["titles"].append((title, timestamp))
        return None


    # MAIN BODY: break by next hedaer
    for header in headers_soup:
        header_text = rex.sub(r'\[.*?\]', '', header.text).strip()
        if header_text.lower() in config['skip_headers']:
            continue

        ps = []
        for sibling in header.find_next_siblings():
            # stops if next sibling is header
            if sibling.get("class") and ' '.join(sibling.get("class")) in headers:
                break
            if sibling.name == 'p': # exlc blockquote
                ps.append(extract_clean_text(sibling, skip_geo=False))
        if ps:
            data['sections'].append({'== '+header_text+' ==': ps})
    if not data['sections']:
        NONE_REASON_RECORDS["no_sections"]["count"] += 1
        #NONE_REASON_RECORDS["no_sections"]["titles"].append((title, timestamp))
        return None


    # REFERENCES: collect links from reference sections
    out_refs = {}
    references = soup.find_all("ol", class_="references")
    for ref in references:
        ref_list = ref.find_all("li", id=True)
        for r in ref_list:
            refs = r.find_all("a", href=True)
            if refs:
                out_refs[r['id']] = [a['href'] for a in refs[1:]]
    if not out_refs:
        NONE_REASON_RECORDS["no_refs"]["count"] += 1
        #NONE_REASON_RECORDS["no_refs"]["titles"].append((title, timestamp))
        return None

    data['refs'] = out_refs
    return data

def process_text(data):
    
    # CLEAN LEAD
    data['lead'] = [clean_text(p, rm_ref=True) for p in data['lead'] if p != '' and p != '\n' and len(p) > 20]
    
    # CLEAN SECTIONS
    clean_sections = []
    for section in data['sections']:
        for title, paras in section.items():
            new_paras = []
            for p in paras:
                if p and p != '\n' and len(p) > 20:
                    new_paras.append(p) # we need the one with refs for later
            if new_paras:
                clean_sections.append({title: new_paras})
    
    data['sections']  =clean_sections
        
    # CLEAN REFS
    ref_clean = {}
    for ref, links in data['refs'].items():
        #ref.split('-')[-1]
        ref_clean[ref] = [l for l in links if (not l.startswith('/wiki/') and 
                                               not l.startswith('#cite') and 
                                               not l.startswith('#CITEREF'))]
        
    data['refs'] = ref_clean
    
    return data

def rm_dups(data):
    uniques = {item['revid']: item for item in data}
    out = list(uniques.values())
    print(f'RM {len(data) - len(out)} dups')

    return out

def main():
    data = load_jsonl(in_file)
    data = rm_dups(data)
    
    out = []
    valid=0
    
    for i, item in enumerate(data):

        print('-------------------', flush=True)
        print('-------------------', flush=True)
        print('-------------------', flush=True)
        print('-------------------\n', flush=True)
        print(i, item['title'], item['revid'], item.keys(), '\n', flush=True)
        
        item_new = extract_text(item['html'], item['title'])
        if item_new is None:
            print('     NONE', flush=True)
            continue
        
        item_new = process_text(item_new)

        # encode title
        item['title'] = html.unescape(item['title'])
        item['title'] = unicodedata.normalize("NFKC", item['title'])
        
        # update item
        item.update(item_new)
        
        del item['html'] # we do not need the html text
        del item['idx']
        
        out.append(item)
        
        print(item['infobox'])
        print('')
        print(item['lead'])
        print('')
        print(item['sections'])

        valid +=1

    NONE_REASON_RECORDS["total"] = i + 1
    NONE_REASON_RECORDS["valid"] = valid
    save_jsonl([NONE_REASON_RECORDS], out_file.replace('.jsonl', '_stats.jsonl'))
    save_jsonl(out, out_file)
    
if __name__ == "__main__":
    out = main()

