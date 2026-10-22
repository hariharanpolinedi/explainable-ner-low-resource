"""
Contextual Disambiguation Dataset Builder for Experiment 8.
Guarantees strictly 0.00% overlap with Gold-Test AND 0.00% overlap with Gold-Dev.
"""

from pathlib import Path
import re
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_conll(path):
    sentences = []
    curr_toks = []
    curr_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_toks:
                    sentences.append((curr_toks, curr_tags))
                    curr_toks, curr_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    curr_toks.append(parts[0])
                    curr_tags.append(parts[1])
        if curr_toks:
            sentences.append((curr_toks, curr_tags))
    return sentences

train_gold = load_conll(PROJECT_ROOT / 'data/gold/kokborok/train.conll')
dev_gold = load_conll(PROJECT_ROOT / 'data/gold/kokborok/dev.conll')
test_gold = load_conll(PROJECT_ROOT / 'data/gold/kokborok/test.conll')

test_sents = set(' '.join(s[0]) for s in test_gold)
dev_sents = set(' '.join(s[0]) for s in dev_gold)
train_sents = set(' '.join(s[0]) for s in train_gold)

print(f"Gold Splits: Train={len(train_gold)}, Dev={len(dev_gold)}, Test={len(test_gold)}")

# Load base training pool
base_train = load_conll(PROJECT_ROOT / 'data/processed/kokborok_exp3_precision_train.conll')

# Load clean verified public ORG sentences from Exp 7
clean_aug = load_conll(PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean.conll')

# Construct strictly non-overlapping training pool
filtered_pool = []
for tokens, tags in base_train + clean_aug:
    s_text = ' '.join(tokens)
    if s_text in test_sents:
        continue
    if s_text in dev_sents:
        continue
    filtered_pool.append((tokens, tags))

# Verify overlaps
exp8_sents = set(' '.join(s[0]) for s in filtered_pool)
overlap_test = exp8_sents.intersection(test_sents)
overlap_dev = exp8_sents.intersection(dev_sents)

print(f"Overlap with Gold-Test: {len(overlap_test)} (STRICT REQUIREMENT = 0)")
print(f"Overlap with Gold-Dev:  {len(overlap_dev)} (STRICT REQUIREMENT = 0)")
assert len(overlap_test) == 0, "ERROR: Test leakage detected!"
assert len(overlap_dev) == 0, "ERROR: Dev leakage detected!"

# Write data/processed/kokborok_exp8_context_train.conll with TAB separation
out_path = PROJECT_ROOT / 'data/processed/kokborok_exp8_context_train.conll'
with open(out_path, 'w', encoding='utf-8') as f:
    for tokens, tags in filtered_pool:
        for t, tag in zip(tokens, tags):
            f.write(f"{t}\t{tag}\n")
        f.write("\n")

print(f"Saved Experiment 8 training pool ({len(filtered_pool)} sentences) to {out_path.relative_to(PROJECT_ROOT)}")

ent_counts = Counter()
for tokens, tags in filtered_pool:
    for tag in tags:
        if tag.startswith('B-'):
            ent_counts[tag[2:]] += 1
print(f"Experiment 8 Pool Entity Counts: {dict(ent_counts)}")
