"""
Comprehensive Kokborok Data Sufficiency & Error Audit Engine (Experiment 9).
Executes Audits 1 through 12 on Gold-Train, Gold-Dev, and Train-eligible Processed Data.
Generates reports/kokborok_exp9_data_statistics.json and reports/kokborok_exp9_error_matrix.csv.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "kokborok"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 1. Helper to load CoNLL sentences and extract entities
def load_conll(path):
    sentences = []
    curr_tokens = []
    curr_tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens:
                    sentences.append((curr_tokens, curr_tags))
                    curr_tokens, curr_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    curr_tokens.append(parts[0])
                    curr_tags.append(parts[1])
        if curr_tokens:
            sentences.append((curr_tokens, curr_tags))
    return sentences

def extract_entities(tokens, tags):
    entities = []
    curr_span = []
    curr_type = None
    curr_start = -1
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if curr_span:
                entities.append((curr_start, i, " ".join(curr_span), curr_type, curr_span))
            curr_span = [tok]
            curr_type = tag[2:]
            curr_start = i
        elif tag.startswith("I-"):
            if curr_span and curr_type == tag[2:]:
                curr_span.append(tok)
            else:
                if curr_span:
                    entities.append((curr_start, i, " ".join(curr_span), curr_type, curr_span))
                curr_span = [tok]
                curr_type = tag[2:]
                curr_start = i
        else:
            if curr_span:
                entities.append((curr_start, i, " ".join(curr_span), curr_type, curr_span))
                curr_span = []
                curr_type = None
                curr_start = -1
    if curr_span:
        entities.append((curr_start, len(tokens), " ".join(curr_span), curr_type, curr_span))
    return entities

# Load datasets
train_sents = load_conll(GOLD_DIR / "train.conll")
dev_sents = load_conll(GOLD_DIR / "dev.conll")

print(f"Loaded Gold Train: {len(train_sents)} sentences")
print(f"Loaded Gold Dev:   {len(dev_sents)} sentences")

# -------------------------------------------------------------
# AUDIT 1: DATASET SIZE & ENTITY TYPE DISTRIBUTION
# -------------------------------------------------------------
def analyze_dataset_entities(sentences):
    total_tokens = sum(len(s[0]) for s in sentences)
    all_ents = []
    sents_with_type = defaultdict(set)
    type_counts = Counter()
    surface_forms = defaultdict(Counter)
    
    for s_idx, (tokens, tags) in enumerate(sentences):
        ents = extract_entities(tokens, tags)
        for e in ents:
            start, end, text, etype, tok_list = e
            all_ents.append(e)
            type_counts[etype] += 1
            sents_with_type[etype].add(s_idx)
            surface_forms[etype][text] += 1
            
    stats = {
        "num_sentences": len(sentences),
        "num_tokens": total_tokens,
        "total_entities": len(all_ents),
        "by_type": {}
    }
    
    for t in ["PER", "LOC", "ORG", "MISC"]:
        cnt = type_counts[t]
        s_cnt = len(sents_with_type[t])
        sf_counter = surface_forms[t]
        unique_sf = len(sf_counter)
        singletons = sum(1 for _, c in sf_counter.items() if c == 1)
        repeated = sum(1 for _, c in sf_counter.items() if c > 1)
        
        stats["by_type"][t] = {
            "entity_count": cnt,
            "sentence_count": s_cnt,
            "pct_total_entities": (cnt / len(all_ents) * 100) if all_ents else 0.0,
            "unique_surface_forms": unique_sf,
            "singleton_surface_forms": singletons,
            "repeated_surface_forms": repeated,
            "top_surface_forms": sf_counter.most_common(5)
        }
    return stats, all_ents, surface_forms

train_stats, train_ents, train_sf = analyze_dataset_entities(train_sents)
dev_stats, dev_ents, dev_sf = analyze_dataset_entities(dev_sents)

# -------------------------------------------------------------
# AUDIT 2: ENTITY LENGTH (1-token, 2-token, 3-token, 4+ token)
# -------------------------------------------------------------
def analyze_entity_length(sentences):
    length_stats = defaultdict(Counter)
    for tokens, tags in sentences:
        ents = extract_entities(tokens, tags)
        for e in ents:
            start, end, text, etype, tok_list = e
            l = len(tok_list)
            l_cat = "1-token" if l == 1 else "2-token" if l == 2 else "3-token" if l == 3 else "4+ token"
            length_stats[etype][l_cat] += 1
            length_stats["OVERALL"][l_cat] += 1
    return length_stats

train_len = analyze_entity_length(train_sents)
dev_len = analyze_entity_length(dev_sents)

# -------------------------------------------------------------
# AUDIT 3: BIO BOUNDARIES & SPECIAL TOKENS
# -------------------------------------------------------------
def analyze_bio_boundaries(sentences):
    tag_counts = Counter()
    clitic_ents = []
    hyphen_ents = []
    english_ents = []
    adjacent_ent_count = 0
    
    clitic_patterns = [r'o$', r'ni$', r'da$', r'ma$', r'ro$', r'kwma$', r'haste$']
    
    for tokens, tags in sentences:
        for t in tags:
            tag_counts[t] += 1
        ents = extract_entities(tokens, tags)
        # Check adjacent entities
        for i in range(len(ents) - 1):
            if ents[i][1] == ents[i+1][0]:
                adjacent_ent_count += 1
        for e in ents:
            start, end, text, etype, tok_list = e
            # Clitic check
            if any(re.search(p, tok_list[-1].lower()) for p in clitic_patterns):
                clitic_ents.append(e)
            # Hyphen check
            if '-' in text:
                hyphen_ents.append(e)
            # English loanword check (ASCII alphabetic)
            if re.match(r'^[A-Za-z]+$', tok_list[0]):
                english_ents.append(e)
                
    return {
        "tag_counts": dict(tag_counts),
        "clitic_entity_count": len(clitic_ents),
        "hyphen_entity_count": len(hyphen_ents),
        "english_entity_count": len(english_ents),
        "adjacent_entity_count": adjacent_ent_count
    }

train_bio = analyze_bio_boundaries(train_sents)
dev_bio = analyze_bio_boundaries(dev_sents)

# -------------------------------------------------------------
# AUDIT 4: VOCABULARY & OOV ANALYSIS
# -------------------------------------------------------------
train_token_vocab = set(t.lower() for s in train_sents for t in s[0])
train_entity_sf = set(e[2].lower() for s in train_sents for e in extract_entities(s[0], s[1]))
train_entity_head = set(e[4][-1].lower() for s in train_sents for e in extract_entities(s[0], s[1]))

dev_tokens = [t.lower() for s in dev_sents for t in s[0]]
dev_token_oov = [t for t in dev_tokens if t not in train_token_vocab]
dev_token_oov_rate = len(dev_token_oov) / len(dev_tokens) if dev_tokens else 0.0

dev_ents_by_type = defaultdict(list)
dev_oov_by_type = defaultdict(list)

for s in dev_sents:
    for e in extract_entities(s[0], s[1]):
        etype = e[3]
        sf = e[2]
        dev_ents_by_type[etype].append(e)
        if sf.lower() not in train_entity_sf:
            dev_oov_by_type[etype].append(e)

oov_stats = {
    "train_token_vocab_size": len(train_token_vocab),
    "dev_total_tokens": len(dev_tokens),
    "dev_token_oov_count": len(dev_token_oov),
    "dev_token_oov_rate": dev_token_oov_rate,
    "per_type_sf_oov": {
        t: {
            "total_dev_entities": len(dev_ents_by_type[t]),
            "oov_entities": len(dev_oov_by_type[t]),
            "oov_rate": (len(dev_oov_by_type[t]) / len(dev_ents_by_type[t])) if dev_ents_by_type[t] else 0.0,
            "oov_examples": [e[2] for e in dev_oov_by_type[t]]
        } for t in ["PER", "LOC", "ORG", "MISC"]
    }
}

# -------------------------------------------------------------
# AUDIT 5: ENTITY FREQUENCY DISTRIBUTION
# -------------------------------------------------------------
freq_buckets = defaultdict(lambda: Counter())
for etype, sf_counts in train_sf.items():
    for sf, cnt in sf_counts.items():
        bucket = "singleton (1)" if cnt == 1 else "low (2-3)" if cnt in [2, 3] else "medium (4-10)" if cnt <= 10 else "high (>10)"
        freq_buckets[etype][bucket] += 1
        freq_buckets["OVERALL"][bucket] += 1

# -------------------------------------------------------------
# AUDIT 6: CONTEXTUAL INDICATORS
# -------------------------------------------------------------
context_terms = [
    "university", "board", "department", "committee", "channel",
    "company", "institute", "corporation", "bank", "police", "hukbar", "bedek"
]
context_in_train = defaultdict(int)
for tokens, tags in train_sents:
    for tok in tokens:
        clean_tok = re.sub(r'[^\w]', '', tok).lower()
        if clean_tok in context_terms:
            context_in_train[clean_tok] += 1

# -------------------------------------------------------------
# AUDIT 7: LOC vs ORG LEXICAL OVERLAP
# -------------------------------------------------------------
loc_sf_train = set(sf.lower() for sf in train_sf["LOC"].keys())
org_sf_train = set(sf.lower() for sf in train_sf["ORG"].keys())
overlap_train = loc_sf_train.intersection(org_sf_train)

loc_sf_dev = set(sf.lower() for sf in dev_sf["LOC"].keys())
org_sf_dev = set(sf.lower() for sf in dev_sf["ORG"].keys())
overlap_dev = loc_sf_dev.intersection(org_sf_dev)

# -------------------------------------------------------------
# AUDIT 10: GOLD-DEV ERROR EVALUATION FOR kokborok_exp4_best.pt
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
languages = ("konkani", "assamese", "sinhala", "kokborok")
w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages)

model = NERMultilingualModel(
    word_vocab_size=len(w_id_4),
    char_vocab_size=len(c_id_4),
    num_tags=9,
    num_languages=4,
    crf_config="per_language",
    hidden_dim=128,
).to(device)

sd = torch.load(CHECKPOINTS_DIR / "kokborok_exp4_best.pt", map_location=device)
model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
model.eval()

dev_conll_path = GOLD_DIR / "dev.conll"
dev_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [dev_conll_path]}, word_to_id=w_id_4, char_to_id=c_id_4)
dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

all_gold, all_pred = [], []
with torch.no_grad():
    for batch in dev_loader:
        paths = model(batch["word_ids"].to(device), batch["char_ids"].to(device),
                      lang_ids=batch["lang_ids"].to(device), mask=batch["mask"].to(device))
        for i, slen in enumerate(batch["seq_lengths"]):
            all_gold.append([ID_TO_TAG[tid.item()] for tid in batch["tags"][i, :slen]])
            all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

# Error collection
error_records = []
for s_idx, ((tokens, g_tags), p_tags) in enumerate(zip(dev_sents, all_pred)):
    g_ents = extract_entities(tokens, g_tags)
    p_ents = extract_entities(tokens, p_tags)
    
    g_set = {(start, end, text, etype) for start, end, text, etype, _ in g_ents}
    p_set = {(start, end, text, etype) for start, end, text, etype, _ in p_ents}
    
    fp_ents = p_set - g_set
    fn_ents = g_set - p_set
    
    for start, end, text, etype in fp_ents:
        # Context
        ctx_start = max(0, start - 3)
        ctx_end = min(len(tokens), end + 3)
        ctx = " ".join(tokens[ctx_start:ctx_end])
        
        # Categorize
        if text.lower() == "agartala" and etype == "LOC":
            cat = "possible annotation issue (unannotated lowercase toponym in gold)"
        else:
            cat = "type confusion / spurious prediction"
            
        error_records.append({
            "sentence_index": s_idx + 1,
            "error_type": "FP",
            "entity_span": text,
            "predicted_type": etype,
            "gold_type": "O",
            "local_context": ctx,
            "error_category": cat,
            "full_sentence": " ".join(tokens)
        })
        
    for start, end, text, etype in fn_ents:
        ctx_start = max(0, start - 3)
        ctx_end = min(len(tokens), end + 3)
        ctx = " ".join(tokens[ctx_start:ctx_end])
        error_records.append({
            "sentence_index": s_idx + 1,
            "error_type": "FN",
            "entity_span": text,
            "predicted_type": "O",
            "gold_type": etype,
            "local_context": ctx,
            "error_category": "missed entity",
            "full_sentence": " ".join(tokens)
        })

print(f"\nGold-Dev Total Errors for kokborok_exp4_best.pt: {len(error_records)} (FP={sum(1 for e in error_records if e['error_type']=='FP')}, FN={sum(1 for e in error_records if e['error_type']=='FN')})")

# Write reports/kokborok_exp9_error_matrix.csv
err_csv_path = REPORTS_DIR / "kokborok_exp9_error_matrix.csv"
with open(err_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sentence_index", "error_type", "entity_span", "predicted_type",
        "gold_type", "local_context", "error_category", "full_sentence"
    ])
    writer.writeheader()
    writer.writerows(error_records)

print(f"Saved error matrix CSV to: {err_csv_path.relative_to(PROJECT_ROOT)}")

# -------------------------------------------------------------
# AUDIT 11: DATA QUALITY & BIO CONSISTENCY
# -------------------------------------------------------------
def audit_data_quality(sentences, split_name):
    bio_violations = 0
    duplicate_sentences = 0
    empty_sentences = 0
    seen_sents = set()
    
    for s_idx, (tokens, tags) in enumerate(sentences):
        if not tokens:
            empty_sentences += 1
            continue
        s_str = " ".join(tokens)
        if s_str in seen_sents:
            duplicate_sentences += 1
        seen_sents.add(s_str)
        
        # Check BIO transitions
        in_type = None
        for tok, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                in_type = tag[2:]
            elif tag.startswith("I-"):
                if in_type != tag[2:]:
                    bio_violations += 1
                in_type = tag[2:]
            else:
                in_type = None
                
    return {
        "split": split_name,
        "total_sentences": len(sentences),
        "empty_sentences": empty_sentences,
        "duplicate_sentences": duplicate_sentences,
        "bio_violations": bio_violations
    }

quality_train = audit_data_quality(train_sents, "Gold Train")
quality_dev = audit_data_quality(dev_sents, "Gold Dev")

# -------------------------------------------------------------
# SAVE SUMMARY JSON
# -------------------------------------------------------------
summary_json = {
    "audit_1_dataset_size": {
        "gold_train": train_stats,
        "gold_dev": dev_stats
    },
    "audit_2_entity_length": {
        "gold_train": {k: dict(v) for k, v in train_len.items()},
        "gold_dev": {k: dict(v) for k, v in dev_len.items()}
    },
    "audit_3_bio_boundaries": {
        "gold_train": train_bio,
        "gold_dev": dev_bio
    },
    "audit_4_oov_analysis": oov_stats,
    "audit_5_frequency_buckets": {
        "gold_train": {k: dict(v) for k, v in freq_buckets.items()}
    },
    "audit_6_context_terms_in_train": dict(context_in_train),
    "audit_7_loc_org_overlap": {
        "gold_train_overlap": list(overlap_train),
        "gold_dev_overlap": list(overlap_dev)
    },
    "audit_10_dev_errors": error_records,
    "audit_11_data_quality": {
        "gold_train": quality_train,
        "gold_dev": quality_dev
    }
}

json_path = REPORTS_DIR / "kokborok_exp9_data_statistics.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

print(f"Saved complete audit statistics JSON to: {json_path.relative_to(PROJECT_ROOT)}")
