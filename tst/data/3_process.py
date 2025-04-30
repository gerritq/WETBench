import regex as rex
import os
import sys
import html
import copy
import json
import mwparserfromhell
import numpy as np
import Levenshtein
from spellchecker import SpellChecker
from simplediff import diff
from collections import Counter
from datetime import datetime
import stanza
from utils import load_jsonl, save_jsonl

# NTLK
import nltk
# nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

lang = sys.argv[1]

def load_tools(lang):
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
    
    if lang == 'en':
        
        # sent splitter
        custom_abbreviations = set(['etc', 'St', 'i.e.']) # en 222200693; en 374320719
        nltk.data.path.append('/users/k21157437/nltk_data')
        en_model = nltk.data.load("tokenizers/punkt/english.pickle")
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = en_model._params.abbrev_types.union(custom_abbreviations)
        sent_splitter = PunktSentenceTokenizer(punkt_param)
        
        # word splitter
        word_splitter = lambda text: nltk.word_tokenize(text)
                
        # pos spacy
        import spacy
        pos_tagger = spacy.load('en_core_web_sm')
    
    if lang == 'pt':
        
        # sent splitter
        custom_abbreviations = set(['etc'])
        nltk.data.path.append('/users/k21157437/nltk_data')
        portuguese_model = nltk.data.load("tokenizers/punkt/portuguese.pickle")
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = portuguese_model._params.abbrev_types.union(custom_abbreviations)
        sent_splitter = PunktSentenceTokenizer(punkt_param)
    
        # word splitter
        word_splitter = lambda text: nltk.word_tokenize(text, language='portuguese')
    
        # pos spacy
        import spacy
        pos_tagger = spacy.load('pt_core_news_md')
    
    if lang in ['vi', 'id']:
        
        # sent splitter
        if lang == 'vi':
            custom_abbreviations = set(['tp.', 'gs.', 'ts.', 'pgs.', 'cn.', 'ths.', 'đ/c', 'ông.', 
                                         'bà.', 'p.', 'q.', 'kts.', 'nxb.', 'tt.', 'cty.', 'tp.', 
                                         'tr.', 'đt.', 'km.'])
        elif lang == 'id':
            custom_abbreviations = set(['sdr.', 'bpk.', 'dr.', 'ir.', 'prof.', 'jl.', 'no.', 'ds.', 
                                        'kec.', 'kab.', 'rp.', 'tn.', 'ny.', 'dr.', 'sh.', 'mm.', 
                                        'se.', 'st.', 'km.'])
        nltk.data.path.append('/users/k21157437/nltk_data')
        portuguese_model = nltk.data.load("tokenizers/punkt/english.pickle")
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = portuguese_model._params.abbrev_types.union(custom_abbreviations)
    
        sent_splitter = PunktSentenceTokenizer(punkt_param)

        # sent and word (use default nltk)
        word_splitter = lambda text: nltk.word_tokenize(text)
        
        # pos stanza
        # to match the spacy nlp function 1) pretokenise 2) join to string 3) process
        nlp = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_pretokenized=True)
        pos_tagger = lambda sent: nlp(' '.join(nltk.word_tokenize(sent))) # important! check stanza_check file
    

    return word_splitter, sent_splitter, pos_tagger

word_splitter, sent_splitter, nlp = load_tools(lang)

# --------------------------------------------------------------------------------------------------
# CLEANING
# --------------------------------------------------------------------------------------------------

# REM REFS
def rm_refs(x):
    '''Takes text.
    First removes all references with a name.
    Second, removes all non-named refes'''
    
    # Remove named refs
    NAMEDREF_RE = r'<ref\s+name="([^"]+)"\s*>.*?<ref\s+name="\1"\s*/>'
    x = rex.sub(NAMEDREF_RE, '', x, flags=rex.DOTALL)
    
    # Remove remaining refs
    REF_RE = '<ref([-\p{L}=" <>]+)?>.*?<([ ]+)?\/([ ]+)?ref>'
    x = rex.sub(REF_RE, ' ', x)
    return x


def remove_third_refs(x):
    """
    Takes x.
    Removes all text to the left of a closing </ref> tag in the upper third,
    and removes all text to the right of an opening <ref> tag in the lower third.
    
    """ 
    n = len(x)
    s=0
    e=len(x)
    
    # upper
    upper_match = rex.search(r'</ref>', x[:n // 3]) # make this named!!
    if upper_match:
        s = upper_match.end()
    
    # lower
    lower_match = rex.search(r'<ref', x[2 * n // 3:])
    if lower_match:
        e = 2 * n // 3 + lower_match.start()
    
    return x[s:e]

# MAIN CLEANER
def clean_wikitext(x):
    
    # remove div at the beginning of the string and at the end of the string
    x = rex.sub(r'^<div>', '', x)
    x = rex.sub(r'</div>$', '', x)
    
    # unicode chars handling, is this optimal?
    x = x.replace("\xa0", " ")
    x = x.strip()
    
    # = header; table artifacts; categories; lang links
    if (x.startswith('=') or x.startswith('?') or x.startswith('|') or
        x.startswith('[[Categoria:') or x.startswith('[[Category:') or
        x.startswith('[[Kategori') or x.startswith('[[Thể loại:') or # indonesian and vietnamese
        x.startswith('#') or # redirect as in pt 52495504
        rex.match(r'^\[\[\s?[a-z]{2}:', x)): 
        return ''
        
    # This is removed in by rm refs
    # if 'retrieved on' in x.lower():
    #     return ''
    
    # ignore lines without text, e.g. ( , , , , ) or ]]
    if not rex.findall('\p{L}', x):
        return ''
    
    # unescape html chars and ensure proper encoding
    x = html.unescape(x)
    x = x.encode('utf-8').decode('utf-8')
    
    # keep inner edits
    x = rex.sub(r'<del(.*?)>(.*?)</del>', r'\2', x)
    x = rex.sub(r'<ins(.*?)>(.*?)</ins>', r'\2', x)
    
    # rem refs
    x = rm_refs(x)
    
    # Remove nested links as in pt 58375432
    # logic: open link- nested links with any content - close link
    x = rex.sub(r'\[\[[^\[\]]*?\[\[.*?\]\][^\[\]]*?\]\]', ' ', x)
    
        
    # Remove any remaining tags (span, blockquote, etc.)
    x = rex.sub(r'<+(.*?)>(.*?)</\p{L}+>', '', x)
    
        
    # mw parser twice & reformat links to be detected
    x = x.replace('[ ', '[').replace(' ]', ']')
    
    x = mwparserfromhell.parse(x)
    x = x.strip_code()
    x = mwparserfromhell.parse(x)
    x = x.strip_code()
    
    # if any non-parallel refs remain, rm those conservatively
    x = remove_third_refs(x)
    
    # Remove sinlge tags
    x = rex.sub('\<\/?\p{L}+(.*?)\>', '', x)
        
    # battery of general cleaning
    x = x.strip()
    x = rex.sub('[ ]+', ' ', x)
    x = x.replace(']', '').replace('[', '') # if brackets are left, rm
    x = rex.sub('http\S+', '', x) # same with html
    x = rex.sub('\([^\p{L}]*\)', '', x) # rm parantheses with nothing in them
    x = x.replace('*', '') # rm stars
    x = rex.sub('(right[ ]?\||left[ ]?\||thumb[ ]?\||frame[ ]?\||\d+px[ ]?\|)', '', x) # rm table fragments
    # rm remainder of IPAs
    
    x = rex.sub(r'[\']{2,}', '', x)
    x = rex.sub(r'[\/]{2,}', '', x) # en 94293531
    x = x.replace('\t', ' ')
    x = x.replace('\n', ' ')
    x = x.replace('\r', '')
    x = rex.sub('[ ]+', ' ', x)
    x = rex.sub(r"\s+\)", ")", x)
    x = rex.sub(r"\(\s+", "(", x)
    x = rex.sub(r"\s+,", ",", x)
    x = rex.sub('\(\s?\;\s?', '(', x) # as in 66710217
    # Some punctuation adjustments enabling splitting later on
    # special dot adjustment when </del>.<del>... as in pt 6566930
    #x = rex.sub(r'(?<=[>])\.(?=[<])', '. ', x)
    x = rex.sub(r'\.(?!NET)(?=\p{Lu})', '. ', x).strip() # add space after dot for sentence splitting later  # exclude NEt for this it 5096058 # \p{Lu} for any unicode uppercase letter pt 26316896
    #x = rex.sub(r'[^\.]{1}\p{L}+\.(?=[A-Z])', '. ', x) # add space after dot for sentence splitting later
    x = rex.sub('\s+\.', '.', x) # rm whitespace before dot
    if x.endswith(';') or x.endswith(':') or x.endswith(','):
        x = x[:-1] + '.'
    if not x.endswith('.'):
        x = x + '.'
    
    # # ensure proper encoding
    #x = x.encode('utf-8').decode('utf-8')
    
    return x

# --------------------------------------------------------------------------------------------------
#   MATHCING AND FILTERS
# --------------------------------------------------------------------------------------------------

# match sentence using the cleaned once only
def match_sents(src_chunk, trgt_chunk):
    '''Input src and trgt __clean__ chunks. '''
    
    src_sents = sent_splitter.tokenize(src_chunk)
    trgt_sents = sent_splitter.tokenize(trgt_chunk)
    
    src_sents_with_positions = {s: i for i, s in enumerate(src_sents)}
    
    # get and rm context
    context = set(src_sents).intersection(set(trgt_sents))
    src_sents = [s for s in src_sents if s not in context]
    trgt_sents = [s for s in trgt_sents if s not in context]

    # excpetion caused due to incorrect sent splitting of sent tokeniser
    if not src_sents or not trgt_sents:
        return [], ''

    # Macth each s1 to the s2 with highest bleu score
    matches = []
    targets = {}
    info=''
    for i in range(len(src_sents)):
        bleu = []
        for j in range(len(trgt_sents)):
            # rm del and ins and tokenise
            src_tok = word_splitter(src_sents[i].lower())
            trgt_tok = word_splitter(trgt_sents[j].lower())
            
            # use tokenised sentences for BLEU
            bleu.append(sentence_bleu([src_tok], trgt_tok, smoothing_function=SmoothingFunction().method4))
        
        # Get the best match
        idx = np.argmax(bleu)
        best_trgt = trgt_sents[idx]
        best_bleu = bleu[idx]
        
        src_pos = src_sents_with_positions[src_sents[i]]
        
        # ensure if trgt is selected >1, keep the pair with the highest bleu
        if best_trgt in targets:
            if best_bleu > targets[best_trgt][1]:
                # rm the old match
                matches = [m for m in matches if m[0][1] != best_trgt]
                # update dict with new, better match
                targets[best_trgt] = ((src_sents[i], best_trgt), src_pos, best_bleu)
                # append to list
                matches.append(((src_sents[i], best_trgt), src_pos, best_bleu))
                info = 'multi_trgt'
        else:
            targets[best_trgt] = ((src_sents[i], best_trgt), src_pos, best_bleu)
            matches.append(((src_sents[i], best_trgt), src_pos, best_bleu))
        
    return matches, info


def gen_diffs(src_toks, trgt_toks):
    return diff(src_toks, trgt_toks)


def edited_words(mod_diff):
    
    added, removed = [], []
    
    for op, tok in mod_diff:
        if op == '+':  # Added
            added.extend(tok)
        elif op == '-':  # Removed
            removed.extend(tok)
    
    return (list(removed), list(added))

# --------------------------------------------------------------------------------------------------
# DROPPING FUNCTIONS
# --------------------------------------------------------------------------------------------------

def drop_spelling_edit(mod_diff, spell):
    '''Takes mod diff.
    Checks whether words were removed and added back with a spelling correction.
    Returns True if so.'''
    
    for i, (tag, words) in enumerate(mod_diff):
        if tag == '-':
            # Get all adjacent added words
            added = []
            for j in range(i + 1, len(mod_diff)):
                if mod_diff[j][0] == '+':
                    added.extend(mod_diff[j][1])
                else:
                    break # stop as soon as we hit a different tag
                
            if added:
                # Check spelling and if adjacent word is the correction
                for word in words:
                    correction = spell.correction(word)
                    if not correction == word and correction in added:
                        return True
    
    return False


def sh_edits(src_tok, deleted, trgt_tok, added):
    '''Takes both src and trgt sentences, and deleted and added tokens.
    Returns share fo edits in src and trgt, in that order.'''
    
    sh_del = len(deleted) / len(src_tok) if len(src_tok) > 0 else 0
    sh_insert = len(added) / len(trgt_tok) if len(trgt_tok) > 0 else 0
    
    return sh_del, sh_insert


def levenshtein_edit(src_clean, trgt_clean):
    '''Takes clean src and trgt sents,
    Returns True if LD is smaller 4.'''
    return Levenshtein.distance(src_clean, trgt_clean)


def sh_punctuation(deleted, added):
    '''Takes deleted and added tokens.
    Returns share deleted and share added, in that order.'''
    
    # make this a share of the total number of edits
    i=0
    for tok in deleted:
        if rex.fullmatch(r'[^\p{L}]*',tok):
            i+=1
    
    j=0
    for tok in added:
        if rex.fullmatch(r'[^\p{L}]*',tok):
            j+=1
    
    return i/len(deleted) if deleted else 0, j/len(added) if added else 0


def get_tags(x, lang):
    '''Takes a string and returns paired pos and entities.'''
    doc = nlp(x)
    if lang in ['id', 'vi']:
        assert len(doc.sentences) == 1
        return [(token.upos, token.text) for sent in doc.sentences for token in sent.words], [] # stanza has no ner for those langs
    return [(token.pos_, token.text) for token in doc], [(token.text, ent.label_) for ent in doc.ents for token in ent]


def sh_n_nouns(edited_words, tok_pos):
    '''Takes the added words and pos of tokens.
    Returns the share and number of nouns in the added words.
    '''
    # edit words counts
    n_nouns, n_pnouns = 0, 0
    edited_words_cp = copy.deepcopy(edited_words)
    
    # sentences counts
    n_pnouns_sent = 0
    for pos, tok in tok_pos:
        # there are two noun tags: https://universaldependencies.org/u/pos/
        if pos in ['NOUN', "PROPN"]:
            if pos == "PROPN":
                n_pnouns_sent += 1
            if tok in edited_words_cp:
                n_nouns += 1
                if pos=="PROPN":
                    n_pnouns+=1
                edited_words_cp.remove(tok) # rm to avoid double counting; rm first instance
    
    sh_nouns = n_nouns/len(edited_words) if len(edited_words) > 0 else 0
    sh_pnouns = n_pnouns/len(edited_words) if len(edited_words) > 0 else 0
    sh_pnouns_sent = n_pnouns_sent/len(tok_pos) if len(tok_pos) > 0 else 0
    
    return sh_nouns, sh_pnouns, sh_pnouns_sent, n_nouns, n_pnouns, n_pnouns_sent

def failed_cleaning(x):
    
    issues = []
    
    # x contains artefacts
    match = rex.search(r'[\{\}\|\[\]\<\>]', x)
    if match:
        issues.append(match.group())
    drop_words = ['Imagem:']
    for word in drop_words:
        if word in x:
            issues.append(word)
    
    # count all parants
    if x.count('(') != x.count(')'):
        issues.append('unmatched_parantheses')
    
    # more than one period
    mperiods = rex.findall(r'\.{2,}', x)
    if mperiods:
        issues.append('multiple_periods')
    return issues




def drop_dates(x, threshold):
    date = datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
    if date > threshold:
        return True
    return False


def drop_chemistry(src_toks, trgt_toks, drop_words):
    
    src_toks_low = [x.lower() for x in src_toks]
    trgt_toks_low = [x.lower() for x in trgt_toks]

    for word in drop_words:
        if word in src_toks_low or word in trgt_toks_low:
            return True, word
    return False, ''

# --------------------------------------------------------------------------------------------------
#   MAIN
# --------------------------------------------------------------------------------------------------

def load_and_split(in_file, n_jobs, job_id):

    with open(in_file, 'r', encoding='utf-8') as in_f:
        data = []
        for line in in_f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Malformed line: {e}", file=sys.stderr)
                continue
    
    #data = data[:10]
    # split the lines into chunks
    print(f"Loaded {len(data)} lines", file=sys.stderr)
    splits = np.array_split(data, n_jobs)
    data = splits[job_id]
    return data


def main():
    print('LANG', lang)
    # dirs
    job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    n_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    in_file = f'../../data/{lang}/processing/2_{lang}_crawl.jsonl'

    if n_jobs == 1:
        out_file = f'../../data/{lang}/processing/3_{lang}_proc.jsonl'
    else:
        out_file = f'../../data/{lang}/processing/temp/j{job_id}_{lang}_proc.jsonl'

    if lang == 'en':
        drop_words = ["atom", "atoms", "ion", "ions", "molecule", "molecules", "ph", "proton", "protons", "neutron", "neutrons", "electron", "electrons", "net neutrality"]

    if lang == 'pt':
        drop_words = ["átomo", "átomos", "ião", "iões", "molécula", "moléculas", "ph", "próton", "prótons", "protões", "neutrões", "nêutron", "nêutrons"]

    if lang == 'vi':
        drop_words = ["nguyên tử", "các nguyên tử", "ion", "các ion", "phân tử", "các phân tử", "ph", "proton", "các proton", "nơtron", "các nơtron"]

    if lang not in ['id', 'vi']:
        spell = SpellChecker(language=lang)

    threshold = datetime.strptime('2022-11-30T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    reason_counts = Counter()
    sents_out = []

    
    data = load_and_split(in_file, n_jobs, job_id)
    print('N Data', len(data))
    
    # loop over the lines
    for page in data:

        pairs = page['pairs']
        
        # drop pairs in the dict as not needed later
        del page['pairs']
        
        # --------------------------------------------------------------------------------------
        # Revision-level filtering
        # --------------------------------------------------------------------------------------
        
        # Drop if post release
        if drop_dates(page['timestamp'], threshold):
            reason_counts['post_release'] += 1
            page.update({'drop': True, 'reason': 'post_release'})
            sents_out.append(page)
            continue
        
        # Drop if no parallel edits (only either del or add)
        if not lang == 'vi':
            if page['non_edits'] is True:
                reason_counts['non_edits'] += 1
                page.update({'drop': True, 'reason': 'non_edits'})
                sents_out.append(page)
                continue
            
        # Drop if no edit pair (filter n edits in gends file)
        if len(pairs) == 0:
            reason_counts['no_pair'] += 1
            page.update({'drop': True, 'reason': 'no_pair'})
            sents_out.append(page)
            continue
            
        # --------------------------------------------------------------------------------------
        # Chunk level filtering
        # --------------------------------------------------------------------------------------
        # We allow more than one edit pair, loop parallel chunks
        for idx_pair, pair in enumerate(pairs):
            
            src, trgt = pair
            src_clean = clean_wikitext(src)
            trgt_clean = clean_wikitext(trgt)
            
            # Drop if empty after cleaning
            if src_clean == '' or trgt_clean == '':
                reason_counts['empty_cleaning'] += 1
                page.update({'drop': True, 'reason': 'empty_cleaning'})
                sents_out.append(page)
                continue
            
            # Drop if src and trgt are the same
            if src_clean == trgt_clean:
                reason_counts['src=trgt'] += 1
                page.update({'drop': True, 'reason': 'src=trgt'})
                sents_out.append(page)
                continue
            
            # Drop if unequal number of sentences (we want edits not pure additions)
            if len(sent_splitter.tokenize(src_clean.lower())) != len(sent_splitter.tokenize(trgt_clean.lower())):
                reason_counts['unequal_sents'] += 1
                page.update({'drop': True, 'reason': 'unequal_sents'})
                sents_out.append(page)
                continue
            
            # --------------------------------------------------------------------------------------
            # Match sentences
            # --------------------------------------------------------------------------------------
            matches, info = match_sents(src_clean, trgt_clean)
            
            if not matches:
                reason_counts['no_matches'] += 1
                page.update({
                    'drop': True,
                    'reason': 'no_matches',
                    'src': src_clean,
                    'trgt': trgt_clean
                })
                sents_out.append(page)
                continue
            
            # --------------------------------------------------------------------------------------
            # Pair-level filtering 
            # Add statistics to the item to filter later in gends
            # --------------------------------------------------------------------------------------
            # Loop matches
            n_matches = len(matches)
            for m in matches:
                
                # do not change og page!
                page_copy = copy.deepcopy(page)
                
                # get cleans
                src_clean, trgt_clean = m[0]
                src_idx = m[1]
                bleu = m[-1]
                
                # word tokenise sents
                src_toks, trgt_toks = word_splitter(src_clean), word_splitter(trgt_clean)
    
                # Get diff and edited words
                s_diff = gen_diffs(src_toks, trgt_toks)
                del_words, added_words = edited_words(s_diff)
                
                # update n_matches
                page_copy.update({'n_matches': n_matches})
                
                # ----------------------------------------------------------------------------------
                # Binding filters
                # ----------------------------------------------------------------------------------
                # Drop if failed cleaning
                src_issues, trgt_issues = failed_cleaning(src_clean), failed_cleaning(trgt_clean)
        
                if src_issues or trgt_issues:
                    reason_counts['cleaning_issues'] += 1
                    page_copy.update({
                        'drop': True,
                        'reason': 'cleaning_issues ' + ' '.join(src_issues + trgt_issues),
                        'src': src_clean,
                        'trgt': trgt_clean
                    })
                    sents_out.append(page_copy)
                    continue
                    
                # Drop if at least one spelling edit
                if lang not in ['id', 'vi']:
                    if drop_spelling_edit(s_diff, spell):
                        reason_counts['spelling_edit'] += 1
                        page_copy.update({
                            'drop': True,
                            'reason': 'spelling_edit',
                            'src': src_clean,
                            'trgt': trgt_clean
                        })
                        sents_out.append(page_copy)
                        continue
                
                # Drop if revision is related to chemistry
                drop_c = drop_chemistry(src_toks, trgt_toks, drop_words)
                
                if drop_c[0] == True:
                    reason_counts['chem'] += 1
                    page_copy.update({
                        'drop': True,
                        'reason': 'chem ' + drop_c[1],
                        'src': src_clean,
                        'trgt': trgt_clean
                    })
                    sents_out.append(page_copy)
                    continue
                # ----------------------------------------------------------------------------------
                # Discretionary filters
                # ----------------------------------------------------------------------------------
                
                # Drop if max edits (max del or add)
                sh_del, sh_insert = sh_edits(src_toks, del_words, trgt_toks, added_words)

                # Drop if minimal edit
                lev=levenshtein_edit(src_clean, trgt_clean)
                
                # Drop if non letter edit only
                sh_punct_del, sh_punct_add = sh_punctuation(del_words, added_words)

                # Drop nouns are edited or min proper noun
                src_pos, src_ents = get_tags(src_clean, lang)
                trgt_pos, trgt_ents = get_tags(trgt_clean, lang)
                
                src_sh_nouns, src_sh_pnouns, src_sh_pnouns_sent, src_n_nouns, src_n_pnouns, src_n_pnouns_sent = sh_n_nouns(del_words, src_pos)
                trgt_sh_nouns, trgt_sh_pnouns, trgt_sh_pnouns_sent, trgt_n_nouns, trgt_n_pnouns, trgt_n_pnouns_sent = sh_n_nouns(added_words, trgt_pos)
                
                # ----------------------------------------------------------------------------------
                # Passed binding filters and adds data of disc ones
                # ----------------------------------------------------------------------------------
                reason_counts['keep'] += 1
                page_copy.update({
                    'drop': False,
                    'n_pairs': len(pairs),
                    'idx_pair': idx_pair, # needed for pairing paras
                    'reason': 'keep',
                    'bleu': bleu,
                    'src_idx': src_idx,
                    'src': src_clean,
                    'trgt': trgt_clean,
                    'del_words': del_words,
                    'added_words': added_words,
                    'sh_del': sh_del,
                    'sh_insert': sh_insert,
                    'lev': lev,
                    'sh_punct_del': sh_punct_del,
                    'sh_punct_add': sh_punct_add,
                    # 'src_n_nouns': src_n_nouns,
                    # 'src_n_pnouns': src_n_pnouns,
                    # 'src_sh_nouns': src_sh_nouns,
                    # 'src_sh_pnouns': src_sh_pnouns,
                    # 'trgt_n_nouns': trgt_n_nouns,
                    # 'trgt_n_pnouns': trgt_n_pnouns,
                    # 'trgt_sh_nouns': trgt_sh_nouns,
                    # 'trgt_sh_pnouns': trgt_sh_pnouns,
                    # 'src_n_pnouns_sent': src_n_pnouns_sent,
                    'src_sh_pnouns_sent': src_sh_pnouns_sent,
                    'trgt_n_pnouns_sent': trgt_n_pnouns_sent,
                    # 'trgt_sh_pnouns_sent': trgt_sh_pnouns_sent,
                    # 'src_pos': src_pos,
                    # 'trgt_pos': trgt_pos,
                    # 'src_ents': src_ents,
                    # 'trgt_ents': trgt_ents
                })
                sents_out.append(page_copy)
    
                
    with open(out_file, 'w', encoding='utf-8') as out_f:
        for rev in sents_out:
            out_f.write(json.dumps(rev, ensure_ascii=False) + '\n')

    

if __name__ == '__main__':
    main()
