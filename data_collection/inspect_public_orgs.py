"""
Comprehensive Kokborok Public ORG Discovery & Inspection Script.
Extracts sentences from public sources, searches for candidate organizations,
validates context, checks test set overlap, and categorizes entity types.
"""

import json
import urllib.request
import re
import pandas as pd
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load Gold Sets to check test set leakage
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

all_records = []

# 2. sdmy/kokborok smoldoc
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
        print(f"Loaded smoldoc: {len(all_records)} sentences so far.")
except Exception as e:
    print('smoldoc error:', e)

# 3. sdmy/kokborok smolsent
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
        print(f"Loaded smolsent: {len(all_records)} sentences so far.")
except Exception as e:
    print('smolsent error:', e)

# 4. Local Excel files in data/raw/kokborok
for xls in glob.glob(str(PROJECT_ROOT / 'data/raw/kokborok/*.xlsx')):
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
                        'source': f"data/raw/kokborok/{fname} [{sname}]",
                        'source_url': 'Local Academic Dataset Repository',
                        'license': 'Academic Research License',
                        'doc_id': f"{fname}_{sname}",
                        'sent_id': f"{fname}_{sname}_r{idx}",
                        'kok': kok_text.strip(),
                        'en': en_text.strip()
                    })
    except Exception as e:
        print(f'Excel {fname} error:', e)

# 5. qa_train.parquet
try:
    df_qa = pd.read_parquet(PROJECT_ROOT / 'data/raw/kokborok/qa_train.parquet')
    for idx, row in df_qa.iterrows():
        p = str(row['prompt']).replace('User:', '').strip()
        c = str(row['completion']).replace('Assistant:', '').strip()
        if len(p) > 5:
            all_records.append({
                'source': 'data/raw/kokborok/qa_train.parquet',
                'source_url': 'Local QA Repository',
                'license': 'Academic Research License',
                'doc_id': f"qa_train_{idx}",
                'sent_id': f"qa_train_p_{idx}",
                'kok': p,
                'en': ''
            })
        if len(c) > 5:
            all_records.append({
                'source': 'data/raw/kokborok/qa_train.parquet',
                'source_url': 'Local QA Repository',
                'license': 'Academic Research License',
                'doc_id': f"qa_train_{idx}",
                'sent_id': f"qa_train_c_{idx}",
                'kok': c,
                'en': ''
            })
except Exception as e:
    print('qa_train error:', e)

print(f"Total unified sentences across all local and remote sources: {len(all_records)}")

# 6. Search for candidate organizations
# Pattern indicators
org_patterns = [
    r'\b(?:Tripura|Agartala|Khumulwng|Udaipur|Dharmanagar|Kailashahar|Belonia|Khowai|Teliamura|Ambassa|Sabroom)\s+[A-Za-z]+',
    r'\b(?:University|College|School|Board|Department|Police|Hospital|Mission|Council|Committee|Association|Federation|Bank|Company|Limited|Agency|Office|Samiti|Sangha|Sabha|Party|Union|Club|Institute|Academy|Bedek|Daphani)\b',
    r'\b(?:TSF|TTAADC|TBSE|TSR|ONGC|NEEPCO|IIT|NIT|MBB|Tata|Doordarshan|AIR|TIDC|THDC|TFDPC|TRTC|TSECL|GAIL|IOCL|BSNL|SBI|LIC|UNESCO|WHO|UNICEF|ISRO|DRDO)\b',
    r'\b(?:board|police|university|college|school|department|hospital|council|committee|association|federation|bank|company|agency|office|club|institute|academy|bedek)\b'
]

compiled_regex = re.compile('|'.join(org_patterns), re.IGNORECASE)

candidates = []
for r in all_records:
    k_text = r['kok']
    # Check match
    m = compiled_regex.findall(k_text)
    if m:
        # Check test leakage
        is_in_test = k_text in test_sent_set
        is_in_dev = k_text in dev_sent_set
        is_in_train = k_text in train_sent_set
        candidates.append({
            'source': r['source'],
            'source_url': r['source_url'],
            'license': r['license'],
            'doc_id': r['doc_id'],
            'sent_id': r['sent_id'],
            'kok': k_text,
            'en': r['en'],
            'matches': m,
            'in_test': is_in_test,
            'in_dev': is_in_dev,
            'in_train': is_in_train
        })

print(f"Found {len(candidates)} sentences containing candidate organization patterns.")
print(f"Overlaps with Gold: Test={sum(1 for c in candidates if c['in_test'])}, Dev={sum(1 for c in candidates if c['in_dev'])}, Train={sum(1 for c in candidates if c['in_train'])}")

# Inspect first 25 candidate sentences
for i, c in enumerate(candidates[:25]):
    print(f"\n[{i+1}] Source: {c['source']} (Doc: {c['doc_id']})")
    print(f"     Kokborok: {c['kok']}")
    if c['en']:
        print(f"     English:  {c['en']}")
    print(f"     Matches:  {c['matches']}")
