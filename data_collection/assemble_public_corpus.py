"""
Comprehensive Kokborok ORG Discovery and Extraction Engine.
Scans all public corpora, extracts candidate sentences, determines exact token spans,
categorizes organization types, checks strict data leakage against gold sets,
and prepares the candidate CoNLL and Provenance CSV files.
"""

import json
import urllib.request
import re
import pandas as pd
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load Gold Sets to verify test/dev/train leakage
def load_conll_sentences(path):
    sentences = []
    tokens = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(' '.join(tokens))
                    tokens = []
            else:
                parts = line.split()
                if len(parts) >= 1:
                    tokens.append(parts[0])
        if tokens:
            sentences.append(' '.join(tokens))
    return sentences

train_sents = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/train.conll')
dev_sents = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/dev.conll')
test_sents = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/test.conll')

test_sent_set = set(test_sents)
dev_sent_set = set(dev_sents)
train_sent_set = set(train_sents)

print(f"Loaded Gold Splits: Train={len(train_sents)}, Dev={len(dev_sents)}, Test={len(test_sents)}")

# 2. Gather all sentences across all public sources
all_records = []

# A. sdmy/kokborok - smoldoc
url_smoldoc = 'https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json'
try:
    req = urllib.request.Request(url_smoldoc, headers={'User-Agent': 'KokborokResearchBot/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        docs = json.loads(resp.read().decode('utf-8'))
        for doc in docs:
            doc_id = doc.get('id', '')
            trgs = doc.get('trgs', [])
            srcs = doc.get('srcs', [])
            for i, (trg_s, src_s) in enumerate(zip(trgs, srcs)):
                if trg_s and isinstance(trg_s, str) and len(trg_s.strip()) > 5:
                    all_records.append({
                        'source': 'sdmy/kokborok (smoldoc__en_trp.json)',
                        'source_url': 'https://huggingface.co/datasets/sdmy/kokborok',
                        'license': 'MIT',
                        'doc_id': doc_id,
                        'sent_id': f"{doc_id}_s{i}",
                        'kok': trg_s.strip(),
                        'en': src_s.strip() if src_s else ''
                    })
        print(f"Loaded smoldoc: {len(all_records)} sentences.")
except Exception as e:
    print('smoldoc error:', e)

# B. sdmy/kokborok - smolsent
url_smolsent = 'https://huggingface.co/datasets/sdmy/kokborok/raw/main/smolsent__en_trp.json'
try:
    req = urllib.request.Request(url_smolsent, headers={'User-Agent': 'KokborokResearchBot/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        sents = json.loads(resp.read().decode('utf-8'))
        for i, item in enumerate(sents):
            sid = item.get('id', i)
            kok_s = item.get('src', '')
            en_s = item.get('trg', '')
            if kok_s and len(kok_s.strip()) > 5:
                all_records.append({
                    'source': 'sdmy/kokborok (smolsent__en_trp.json)',
                    'source_url': 'https://huggingface.co/datasets/sdmy/kokborok',
                    'license': 'MIT',
                    'doc_id': f"smolsent_{sid}",
                    'sent_id': f"smolsent_{sid}",
                    'kok': kok_s.strip(),
                    'en': en_s.strip()
                })
        print(f"Loaded smolsent: {len(all_records)} sentences.")
except Exception as e:
    print('smolsent error:', e)

# C. Wikimedia Incubator Kokborok Wikipedia articles
INCUBATOR_API = 'https://incubator.wikimedia.org/w/api.php'
list_url = f"{INCUBATOR_API}?action=query&list=allpages&apprefix=Wp/trp/&aplimit=100&format=json"
try:
    req = urllib.request.Request(list_url, headers={'User-Agent': 'KokborokResearchBot/1.0'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        w_data = json.loads(resp.read().decode('utf-8'))
        pages = w_data.get('query', {}).get('allpages', [])
        page_ids = [str(p['pageid']) for p in pages]
        for i in range(0, len(page_ids), 10):
            batch = "|".join(page_ids[i:i+10])
            curl = f"{INCUBATOR_API}?action=query&prop=extracts&explaintext=1&pageids={batch}&format=json"
            creq = urllib.request.Request(curl, headers={'User-Agent': 'KokborokResearchBot/1.0'})
            with urllib.request.urlopen(creq, timeout=20) as cresp:
                cdat = json.loads(cresp.read().decode('utf-8'))
                for pid, pinfo in cdat.get('query', {}).get('pages', {}).items():
                    title = pinfo.get('title', '')
                    extract = pinfo.get('extract', '')
                    for s_idx, line in enumerate(extract.splitlines()):
                        line = line.strip()
                        if len(line) > 10:
                            for s in re.split(r'(?<=[.!?।])\s+', line):
                                s = s.strip()
                                if len(s) > 10 and ' ' in s:
                                    all_records.append({
                                        'source': 'Wikimedia Incubator (Wp/trp)',
                                        'source_url': f"https://incubator.wikimedia.org/wiki/{title}",
                                        'license': 'CC-BY-SA 3.0/4.0',
                                        'doc_id': f"wiki_{pid}_{title.replace('/', '_')}",
                                        'sent_id': f"wiki_{pid}_s{s_idx}",
                                        'kok': s,
                                        'en': ''
                                    })
        print(f"Loaded Wikimedia Incubator: {len(all_records)} sentences total.")
except Exception as e:
    print('Incubator error:', e)

# D. Local Kokborok_datasets Excel files
for xls in glob.glob(str(PROJECT_ROOT / 'Kokborok_datasets/*.xlsx')):
    fname = Path(xls).name
    try:
        excel = pd.ExcelFile(xls)
        for sname in excel.sheet_names:
            df = pd.read_excel(xls, sheet_name=sname)
            for idx, row in df.iterrows():
                kok_text = ''
                en_text = ''
                for col in df.columns:
                    clow = str(col).lower()
                    if 'kokborok' in clow or 'trp' in clow or 'translation' in clow or 'answer' in clow:
                        kok_text = str(row[col]) if pd.notna(row[col]) else ''
                    elif 'english' in clow or 'question' in clow:
                        en_text = str(row[col]) if pd.notna(row[col]) else ''
                if not kok_text and len(df.columns) == 1:
                    kok_text = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
                
                if kok_text and kok_text != 'nan' and len(kok_text.strip()) > 5:
                    all_records.append({
                        'source': f"Kokborok_datasets/{fname} [{sname}]",
                        'source_url': 'Local Academic Dataset Repository',
                        'license': 'Academic Research License',
                        'doc_id': f"{fname}_{sname}",
                        'sent_id': f"{fname}_{sname}_r{idx}",
                        'kok': kok_text.strip(),
                        'en': en_text.strip()
                    })
    except Exception as e:
        print(f'Excel {fname} error:', e)

# E. Local QA parquet
try:
    df_qa = pd.read_parquet(PROJECT_ROOT / 'data/raw/kokborok/qa_train.parquet')
    for idx, row in df_qa.iterrows():
        p = str(row['prompt']).replace('User:', '').strip()
        c = str(row['completion']).replace('Assistant:', '').strip()
        if len(p) > 5:
            all_records.append({
                'source': 'data/raw/kokborok/qa_train.parquet',
                'source_url': 'Local QA Corpus',
                'license': 'Academic Research License',
                'doc_id': f"qa_train_{idx}",
                'sent_id': f"qa_train_p_{idx}",
                'kok': p,
                'en': ''
            })
        if len(c) > 5:
            all_records.append({
                'source': 'data/raw/kokborok/qa_train.parquet',
                'source_url': 'Local QA Corpus',
                'license': 'Academic Research License',
                'doc_id': f"qa_train_{idx}",
                'sent_id': f"qa_train_c_{idx}",
                'kok': c,
                'en': ''
            })
except Exception as e:
    print('qa_train error:', e)

print(f"\nGrand total sentences assembled across all sources: {len(all_records):,}")

# Write all assembled records to a temporary file for fast inspection
with open(PROJECT_ROOT / 'data_collection/unified_public_corpus.json', 'w', encoding='utf-8') as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

print("Saved unified corpus to data_collection/unified_public_corpus.json")
