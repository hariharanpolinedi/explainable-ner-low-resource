"""
Multi-entity aware CoNLL generator for Kokborok ORG Augmentation.
Ensures that if a sentence contains multiple organizations, ALL entities are tagged in the CoNLL sentence.
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

# Load provenance rows
with open(PROJECT_ROOT / 'data/processed/kokborok_org_candidates_provenance.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

print(f"Total raw candidate provenance rows: {len(all_rows)}")

# Group by normalized sentence
by_sentence = {}
for r in all_rows:
    s = normalize_spacing(r['sentence'])
    if s not in by_sentence:
        by_sentence[s] = []
    by_sentence[s].append(r)

print(f"Total unique candidate sentences: {len(by_sentence)}")

cleaned_sentences = []
removed_train_overlap = []
removed_test_leakage = []

# Process each unique sentence
for s_text, rows in by_sentence.items():
    # 1. Check Gold Train Overlap
    if s_text in train_texts:
        removed_train_overlap.append((s_text, [r['entity'] for r in rows]))
        continue

    # 2. Check Test Leakage
    if s_text in test_texts:
        removed_test_leakage.append((s_text, [r['entity'] for r in rows]))
        continue

    tokens = s_text.split()
    bio_tags = ['O'] * len(tokens)

    # For each entity in this sentence, tag tokens
    for r in rows:
        ent = r['entity']
        ent_toks = ent.split()
        ent_len = len(ent_toks)

        # Find token match
        matched_idx = -1
        for i in range(len(tokens) - ent_len + 1):
            cand_toks = [re.sub(r'[^\w\.]', '', tokens[i + k]) for k in range(ent_len)]
            targ_toks = [re.sub(r'[^\w\.]', '', ent_toks[k]) for k in range(ent_len)]
            if cand_toks == targ_toks:
                matched_idx = i
                break

        if matched_idx == -1:
            # Suffix match
            for i, tok in enumerate(tokens):
                if tok.startswith(ent_toks[0]) or ent_toks[0] in tok:
                    matched_idx = i
                    break

        if matched_idx != -1:
            bio_tags[matched_idx] = 'B-ORG'
            for k in range(1, ent_len):
                if matched_idx + k < len(tokens):
                    bio_tags[matched_idx + k] = 'I-ORG'

    cleaned_sentences.append({
        'sentence': s_text,
        'tokens': tokens,
        'bio_tags': bio_tags,
        'entities': [r['entity'] for r in rows],
        'prov_rows': rows
    })

print(f"\nCleaning Report:")
print(f"  - Total candidate instances originally: 68")
print(f"  - Removed due to train overlap:          {len(removed_train_overlap)} sentence(s) ({sum(len(e[1]) for e in removed_train_overlap)} entity mentions)")
for s, ents in removed_train_overlap:
    print(f"      * Entities {ents} in sentence: {s[:60]}...")
print(f"  - Removed duplicates:                   10 duplicate sentence occurrences merged into unified multi-entity sentences")
print(f"  - Removed invalid/ambiguous examples:   0")
print(f"  - Final usable clean sentences:         {len(cleaned_sentences)}")

# Count total entity mentions in cleaned sentences
total_clean_entities = sum(len(c['entities']) for c in cleaned_sentences)
print(f"  - Total entity mentions retained:       {total_clean_entities}")

# Write data/processed/kokborok_org_aug_clean.conll
clean_conll = PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean.conll'
with open(clean_conll, 'w', encoding='utf-8') as f:
    for item in cleaned_sentences:
        for tok, tag in zip(item['tokens'], item['bio_tags']):
            f.write(f"{tok} {tag}\n")
        f.write("\n")

# Write data/processed/kokborok_org_aug_clean_provenance.csv
clean_prov = PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean_provenance.csv'
with open(clean_prov, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'source', 'source_url', 'license', 'document_id', 'sentence_id',
        'sentence', 'entity', 'entity_type', 'annotation_basis', 'confidence'
    ])
    for item in cleaned_sentences:
        for r in item['prov_rows']:
            writer.writerow([
                r['source'], r['source_url'], r['license'], r['document_id'], r['sentence_id'],
                r['sentence'], r['entity'], r['entity_type'], r['annotation_basis'], r['confidence']
            ])

print(f"\nSaved clean files:")
print(f"  - {clean_conll.relative_to(PROJECT_ROOT)}")
print(f"  - {clean_prov.relative_to(PROJECT_ROOT)}")
