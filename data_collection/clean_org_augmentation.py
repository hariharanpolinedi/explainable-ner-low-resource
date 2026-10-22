"""
Step 1: Clean Kokborok ORG Augmentation Data.
Removes train overlap, checks duplicates, validates BIO tags and Kokborok text quality.
Generates data/processed/kokborok_org_aug_clean.conll and provenance CSV.
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load Gold Train
def normalize_spacing(s):
    return ' '.join(s.strip().split())

def load_conll_sentences(path):
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

train_data = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/train.conll')
dev_data = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/dev.conll')
test_data = load_conll_sentences(PROJECT_ROOT / 'data/gold/kokborok/test.conll')

train_texts = set(normalize_spacing(s[0]) for s in train_data)
dev_texts = set(normalize_spacing(s[0]) for s in dev_data)
test_texts = set(normalize_spacing(s[0]) for s in test_data)

# Load existing candidate CoNLL and Provenance
cand_conll = load_conll_sentences(PROJECT_ROOT / 'data/processed/kokborok_org_candidates.conll')

cand_prov = []
with open(PROJECT_ROOT / 'data/processed/kokborok_org_candidates_provenance.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cand_prov = list(reader)

print(f"Loaded original candidate instances: {len(cand_prov)} (CoNLL sentences: {len(cand_conll)})")

# Auditing and cleaning
cleaned_records = []
removed_train_overlap = []
removed_duplicates = []
removed_invalid = []

seen_sentences = set()

for prov, (sent_text, tokens, tags) in zip(cand_prov, cand_conll):
    norm_sent = normalize_spacing(prov['sentence'])
    
    # 1. Check Gold Train Overlap
    if norm_sent in train_texts:
        removed_train_overlap.append({
            'sentence': norm_sent,
            'entity': prov['entity'],
            'reason': 'Exact match with Gold-Train sentence'
        })
        continue

    # 2. Check Gold Dev & Test (Safety verification)
    if norm_sent in test_texts or norm_sent in dev_texts:
        removed_invalid.append({
            'sentence': norm_sent,
            'entity': prov['entity'],
            'reason': 'Matches Gold Dev or Test set'
        })
        continue

    # 3. Check duplicate sentences
    if norm_sent in seen_sentences:
        removed_duplicates.append({
            'sentence': norm_sent,
            'entity': prov['entity'],
            'reason': 'Duplicate sentence already included'
        })
        continue
    seen_sentences.add(norm_sent)

    # 4. Check BIO validity: B-ORG must precede I-ORG, only B-ORG, I-ORG, O
    bio_valid = True
    in_org = False
    for tag in tags:
        if tag == 'B-ORG':
            in_org = True
        elif tag == 'I-ORG':
            if not in_org:
                bio_valid = False
                break
        elif tag == 'O':
            in_org = False
        else:
            bio_valid = False
            break

    if not bio_valid:
        removed_invalid.append({
            'sentence': norm_sent,
            'entity': prov['entity'],
            'reason': f'Invalid BIO sequence: {tags}'
        })
        continue

    # 5. Check valid Kokborok text
    if len(tokens) < 3:
        removed_invalid.append({
            'sentence': norm_sent,
            'entity': prov['entity'],
            'reason': 'Sentence too short (<3 tokens)'
        })
        continue

    # Retain
    cleaned_records.append({
        'prov': prov,
        'tokens': tokens,
        'tags': tags,
        'norm_sent': norm_sent
    })

print(f"\nCleaning Report:")
print(f"  - Original examples:            {len(cand_prov)}")
print(f"  - Removed due to train overlap: {len(removed_train_overlap)}")
for r in removed_train_overlap:
    print(f"      * {r['entity']} in: {r['sentence'][:60]}...")
print(f"  - Removed duplicates:           {len(removed_duplicates)}")
for r in removed_duplicates:
    print(f"      * {r['entity']} in: {r['sentence'][:60]}...")
print(f"  - Removed invalid/ambiguous:    {len(removed_invalid)}")
print(f"  - Final usable ORG examples:    {len(cleaned_records)}")

# Write data/processed/kokborok_org_aug_clean.conll
clean_conll_path = PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean.conll'
with open(clean_conll_path, 'w', encoding='utf-8') as f:
    for rec in cleaned_records:
        for tok, tag in zip(rec['tokens'], rec['tags']):
            f.write(f"{tok} {tag}\n")
        f.write("\n")

# Write data/processed/kokborok_org_aug_clean_provenance.csv
clean_prov_path = PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean_provenance.csv'
with open(clean_prov_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'source', 'source_url', 'license', 'document_id', 'sentence_id',
        'sentence', 'entity', 'entity_type', 'annotation_basis', 'confidence'
    ])
    for rec in cleaned_records:
        p = rec['prov']
        writer.writerow([
            p['source'], p['source_url'], p['license'], p['document_id'], p['sentence_id'],
            p['sentence'], p['entity'], p['entity_type'], p['annotation_basis'], p['confidence']
        ])

print(f"\nSaved cleaned files to:")
print(f"  {clean_conll_path.relative_to(PROJECT_ROOT)}")
print(f"  {clean_prov_path.relative_to(PROJECT_ROOT)}")

# Sector distribution of clean examples
cat_counts = Counter(r['prov']['entity_type'] for r in cleaned_records)
print("\nFinal Clean Distribution by Sector:")
for cat, cnt in cat_counts.most_common():
    print(f"  {cat:15s}: {cnt:2d} ({cnt/len(cleaned_records)*100:.1f}%)")
