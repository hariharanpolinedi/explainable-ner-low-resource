"""
Systematic ORG Discovery, Validation, and Annotation Script for Kokborok.
"""

import json
import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load unified sentences
with open(PROJECT_ROOT / 'data_collection/unified_public_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Load gold sets
def load_conll_data(path):
    sents = []
    tokens = []
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sents.append((' '.join(tokens), tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
        if tokens:
            sents.append((' '.join(tokens), tokens, tags))
    return sents

train_data = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/train.conll')
dev_data = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/dev.conll')
test_data = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/test.conll')

test_sent_texts = set(s[0] for s in test_data)
dev_sent_texts = set(s[0] for s in dev_data)
train_sent_texts = set(s[0] for s in train_data)

print(f"Loaded Gold Splits: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")

# Search and extraction rules
# We search for known and unknown organizations, institutions, political parties, universities, schools,
# companies, government departments, media channels, sports boards, tribal councils, cultural bodies, etc.
# We also detect false positives (e.g. general common nouns like 'party', 'office', 'bank' without an org name, or pure locations).

# Let's inspect potential organization phrases
candidates = []
seen_sentences = set()

for item in corpus:
    kok = item['kok']
    en = item.get('en', '')
    if kok in seen_sentences:
        continue
    seen_sentences.add(kok)

    # Check if exactly matches test set
    if kok in test_sent_texts:
        # STRICT RULE: Locked test set must NOT be used
        continue

    # Let's find patterns in Kokborok text
    # 1. Indian National Congress / Congress party / BJP / CPI(M) / Scheduled Castes Federation
    # 2. Columbia University / University of Cambridge / Harvard / Oxford / Tripura University / MBB College / NIT Agartala
    # 3. World Health Organization / WHO / UNICEF / UNESCO / United Nations / UN
    # 4. Doordarshan / DD news / AIR / Akashvani / Kokborok Hukbar Bedek / Tripura channel / Chini Khorang
    # 5. Tata / Tata Technologies / Google / Microsoft / Apple / Amazon / Infosys / Wipro / ISRO / NASA
    # 6. Tripura board / TBSE / Tripura Police / TTAADC / TSF / TSR / Agartala Municipal Corporation / Tripura Gramin Bank / ONGC / NEEPCO
    # 7. Kokborok Tei Hukumu Mission / Kokborok Sahitya Sabha / Tripura Baptist Christian Union / TBCU
    # 8. Department / bedek phrases: e.g. Tripura Sorkar, Tripura police, etc.

# Let's write a script to inspect and tag verified organizations
