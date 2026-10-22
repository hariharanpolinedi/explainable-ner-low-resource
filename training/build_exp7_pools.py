"""
Step 2 & 3: Data Balance and Dataset Pool Construction.
Computes entity distribution for Baseline (0x), 0.5x Augmentation, and 1.0x Augmentation.
Generates the corresponding training pools in data/processed/ formatted with TAB separation.
"""

from collections import Counter
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_conll_entities_and_tokens(path):
    sentences = []
    tokens_list = []
    tags_list = []
    
    curr_tokens = []
    curr_tags = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens:
                    sentences.append((curr_tokens, curr_tags))
                    tokens_list.extend(curr_tokens)
                    tags_list.extend(curr_tags)
                    curr_tokens, curr_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    curr_tokens.append(parts[0])
                    curr_tags.append(parts[1])
        if curr_tokens:
            sentences.append((curr_tokens, curr_tags))
            tokens_list.extend(curr_tokens)
            tags_list.extend(curr_tags)

    entity_counts = Counter()
    for s_tokens, s_tags in sentences:
        for tag in s_tags:
            if tag.startswith('B-'):
                entity_counts[tag[2:]] += 1

    tag_counts = Counter(tags_list)
    return sentences, entity_counts, tag_counts

# 1. Baseline Training Pool (kokborok_exp3_precision_train.conll)
base_path = PROJECT_ROOT / 'data/processed/kokborok_exp3_precision_train.conll'
base_sents, base_ent_counts, base_tag_counts = load_conll_entities_and_tokens(base_path)

# 2. Clean ORG Augmentation (kokborok_org_aug_clean.conll)
aug_path = PROJECT_ROOT / 'data/processed/kokborok_org_aug_clean.conll'
aug_sents, aug_ent_counts, aug_tag_counts = load_conll_entities_and_tokens(aug_path)

print(f"Baseline Sentences: {len(base_sents)}")
print(f"Baseline Entities:  {dict(base_ent_counts)}")
print(f"Clean Aug Sentences: {len(aug_sents)}")
print(f"Clean Aug Entities:  {dict(aug_ent_counts)}")

# Create 0.5x Augmentation Pool (28 sentences deterministic sample)
random.seed(42)
aug_indices = list(range(len(aug_sents)))
random.shuffle(aug_indices)
half_indices = set(aug_indices[:len(aug_indices)//2])

aug_sents_half = [s for i, s in enumerate(aug_sents) if i in half_indices]
aug_sents_full = aug_sents

# Write Pool 0x (identical to baseline)
pool_0x_path = PROJECT_ROOT / 'data/processed/kokborok_exp7_train_0x.conll'
with open(pool_0x_path, 'w', encoding='utf-8') as f:
    for tokens, tags in base_sents:
        for t, tag in zip(tokens, tags):
            f.write(f"{t}\t{tag}\n")
        f.write("\n")

# Write Pool 0.5x (baseline + 50% org aug)
pool_05x_path = PROJECT_ROOT / 'data/processed/kokborok_exp7_train_05x.conll'
with open(pool_05x_path, 'w', encoding='utf-8') as f:
    for tokens, tags in base_sents + aug_sents_half:
        for t, tag in zip(tokens, tags):
            f.write(f"{t}\t{tag}\n")
        f.write("\n")

# Write Pool 1.0x (baseline + 100% org aug)
pool_1x_path = PROJECT_ROOT / 'data/processed/kokborok_exp7_train_1x.conll'
with open(pool_1x_path, 'w', encoding='utf-8') as f:
    for tokens, tags in base_sents + aug_sents_full:
        for t, tag in zip(tokens, tags):
            f.write(f"{t}\t{tag}\n")
        f.write("\n")

print(f"\nGenerated Pools (TAB-delimited):")
for name, p_path in [("0x", pool_0x_path), ("0.5x", pool_05x_path), ("1.0x", pool_1x_path)]:
    sents, ent_c, tag_c = load_conll_entities_and_tokens(p_path)
    total_ents = sum(ent_c.values())
    print(f"  Pool {name:<5}: {len(sents):4d} sents | PER: {ent_c['PER']:3d} ({ent_c['PER']/total_ents*100:4.1f}%), LOC: {ent_c['LOC']:3d} ({ent_c['LOC']/total_ents*100:4.1f}%), ORG: {ent_c['ORG']:3d} ({ent_c['ORG']/total_ents*100:4.1f}%), MISC: {ent_c['MISC']:3d} ({ent_c['MISC']/total_ents*100:4.1f}%) | Total Ents: {total_ents:3d}")
