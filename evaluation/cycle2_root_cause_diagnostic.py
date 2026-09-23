"""
Reliability Cycle 2, Step 1: Root-Cause Confirmation Diagnostic.

Executes:
Part A: Konkani Inflectional-OOV Confirmation
1. Scans Konkani diagnostic sample (247 sents) and gold train/dev for case-inflected entity forms.
2. Checks in-vocab vs OOV in word_vocab.json for inflected forms and uninflected base forms.
3. Cross-references against Phase 7 missed_entity (34) and complete_sentence_collapse (55) errors.
4. Counterfactual test: substitutes base form in place of inflected form on failing sentences,
   re-runs GPU inference, and reports success rate with sentence-level diffs.

Part B: Assamese Long-Sequence Decay Confirmation
5. Analyzes length distribution of the 56 Assamese complete_sentence_collapse cases.
6. Evaluates BiLSTM hidden states, self-attention entropy/max-weights, and CRF emission logits
   on collapsed long sentences (including sentence #200).
7. Counterfactual test: truncates collapsed long sentences to shorter sub-spans,
   re-runs GPU inference, and measures if entity detection is restored.
8. Disentangles token length vs entity density correlation with sequence collapse.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Reconfigure stdout/stderr to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.phase1_error_taxonomy_study import (
    build_training_vocab,
    categorize_sentence_errors,
    extract_entities,
    INFLECTION_SUFFIXES,
)
from evaluation.phase5_targeted_interventions import (
    ID_TO_TAG,
    TAG_TO_ID,
    NERMultilingualModel,
    load_conll_data,
    load_or_create_multilingual_vocabs,
    set_global_seed,
)
from evaluation.phase6_combined_retraining import (
    load_diagnostic_manifest_sents,
)
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import LANG_TO_ID

# Set deterministic seed
set_global_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

# Known Konkani case endings mapping inflected form to candidate base form
KONKANI_CASE_RULES = [
    # Locative
    (re.compile(r"(.+)ांत$"), r"\1"),
    (re.compile(r"(.+)ंत$"), r"\1"),
    (re.compile(r"(.+)चेर$"), r"\1"),
    (re.compile(r"(.+)कडे$"), r"\1"),
    # Genitive
    (re.compile(r"(.+)च्या$"), r"\1"),
    (re.compile(r"(.+)ाच्या$"), r"\1"),
    (re.compile(r"(.+)चो$"), r"\1"),
    (re.compile(r"(.+)ाचो$"), r"\1"),
    (re.compile(r"(.+)ची$"), r"\1"),
    (re.compile(r"(.+)ाची$"), r"\1"),
    (re.compile(r"(.+)चे$"), r"\1"),
    (re.compile(r"(.+)ाचे$"), r"\1"),
    # Dative / Accusative
    (re.compile(r"(.+)क$"), r"\1"),
    (re.compile(r"(.+)का$"), r"\1"),
    # Instrumental / Ergative
    (re.compile(r"(.+)ान$"), r"\1"),
    (re.compile(r"(.+)ेन$"), r"\1"),
    (re.compile(r"(.+)च्यान$"), r"\1"),
]

# Irregular stem alternations in Konkani
KONKANI_STEM_MAP = {
    "गोयाच्या": "गोंय",
    "गोयांत": "गोंय",
    "गोयाक": "गोंय",
    "गोयान": "गोंय",
    "गोया": "गोंय",
    "पणजींत": "पणजी",
    "मडगांवांत": "मडगांव",
    "भारतांत": "भारत",
    "भारताच्या": "भारत",
    "भारताक": "भारत",
    "दिल्लींत": "दिल्ली",
    "मुंबयंत": "मुंबय",
    "आंबेडकरांनी": "आंबेडकर",
    "सावंतांनी": "सावंत",
    "नायकांनी": "नाईक",
    "कामातींनी": "कामत",
}


def load_model(ckpt_path: Path, w_len: int, c_len: int) -> nn.Module:
    d = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = d.get("model_state_dict", d)
    crf_cfg = "per_language" if any("crf_heads" in k for k in sd) else "shared"
    m = NERMultilingualModel(
        word_vocab_size=w_len,
        char_vocab_size=c_len,
        num_tags=9,
        num_languages=3,
        hidden_dim=128,
        crf_config=crf_cfg,
    ).to(DEVICE)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def predict_single_sentence(
    model: nn.Module,
    tokens: list[str],
    lang: str,
    w_map: dict[str, int],
    c_map: dict[str, int],
) -> list[str]:
    tokenizer = ScriptTokenizer(lang)
    max_len = len(tokens)
    if max_len == 0:
        return []
    w_batch = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.ones((1, max_len), dtype=torch.bool)
    tok_graphemes = [tokenizer.tokenize_characters(w) for w in tokens]
    max_c = max(len(gs) for gs in tok_graphemes) if tok_graphemes else 1
    c_batch = torch.zeros((1, max_len, max_c), dtype=torch.long)
    l_batch = torch.tensor([LANG_TO_ID[lang]], dtype=torch.long)

    for j, (w, gs) in enumerate(zip(tokens, tok_graphemes)):
        w_batch[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
        for k, g in enumerate(gs):
            c_batch[0, j, k] = c_map.get(g, c_map.get("<UNK>", 1))

    with torch.no_grad():
        paths = model(
            w_batch.to(DEVICE),
            c_batch.to(DEVICE),
            lang_ids=l_batch.to(DEVICE),
            mask=mask.to(DEVICE),
        )
    return [ID_TO_TAG[int(tid)] for tid in paths[0][:max_len]]


def infer_base_form(token: str) -> str | None:
    if token in KONKANI_STEM_MAP:
        return KONKANI_STEM_MAP[token]
    for pattern, replacement in KONKANI_CASE_RULES:
        m = pattern.match(token)
        if m and len(m.group(1)) >= 2:
            candidate = pattern.sub(replacement, token)
            return candidate
    return None


def run_diagnostics():
    print("=" * 80)
    print("RELIABILITY CYCLE 2, STEP 1: ROOT-CAUSE CONFIRMATION")
    print("=" * 80)

    # Load shared vocabularies
    w_map, c_map = load_or_create_multilingual_vocabs(DATA_DIR / "processed")
    konkani_word_vocab_json = json.loads((DATA_DIR / "processed" / "konkani" / "word_vocab.json").read_text(encoding="utf-8"))
    assamese_word_vocab_json = json.loads((DATA_DIR / "processed" / "assamese" / "word_vocab.json").read_text(encoding="utf-8"))
    
    konkani_vocab = set(konkani_word_vocab_json["token_to_id"].keys())
    assamese_vocab = set(assamese_word_vocab_json["token_to_id"].keys())

    # Load models
    konkani_model = load_model(PROJECT_ROOT / "checkpoints" / "improved_konkani_v2.pt", len(w_map), len(c_map))
    assamese_model = load_model(PROJECT_ROOT / "checkpoints" / "improved_assamese_v2.pt", len(w_map), len(c_map))

    # Load diagnostic manifests
    konkani_manifest = load_diagnostic_manifest_sents("konkani")
    assamese_manifest = load_diagnostic_manifest_sents("assamese")

    train_vocab_konkani = build_training_vocab(DATA_DIR / "gold" / "konkani" / "train.conll")
    train_vocab_assamese = build_training_vocab(DATA_DIR / "gold" / "assamese" / "train.conll")

    # =========================================================================
    # PART A: KONKANI INFLECTIONAL-OOV CONFIRMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: KONKANI INFLECTIONAL-OOV CONFIRMATION")
    print("=" * 80)

    # 1. Predict Konkani manifest and categorize errors
    konkani_preds = []
    for toks, gold, src in konkani_manifest:
        pred = predict_single_sentence(konkani_model, toks, "konkani", w_map, c_map)
        konkani_preds.append(pred)

    konkani_errors_by_cat = defaultdict(list)
    for idx, ((toks, gold, src), pred) in enumerate(zip(konkani_manifest, konkani_preds)):
        errs = categorize_sentence_errors(toks, gold, pred, idx, src, "konkani", train_vocab_konkani)
        for e in errs:
            konkani_errors_by_cat[e["category"]].append((idx, toks, gold, pred, e))

    print(f"Total Konkani Sentences: {len(konkani_manifest)}")
    print(f"Missed Entity Errors:    {len(konkani_errors_by_cat['missed_entity'])}")
    print(f"Complete Collapse Errors: {len(konkani_errors_by_cat['complete_sentence_collapse'])}")
    print(f"False Positive Errors:   {len(konkani_errors_by_cat['false_positive'])}")

    # 2. Extract all gold entities in Konkani manifest and train/dev
    all_konkani_conll_files = [
        DATA_DIR / "gold" / "konkani" / "dev.conll",
        DATA_DIR / "gold" / "konkani" / "train.conll",
        DATA_DIR / "gold" / "konkani" / "long_multientity_eval_v2.conll",
        DATA_DIR / "gold" / "konkani" / "supplementary_long_multientity.conll",
    ]
    
    gold_entities_pool = []
    for f in all_konkani_conll_files:
        if f.exists():
            sents = load_conll_data(f)
            for toks, tags in sents:
                ents = extract_entities(tags)
                for etype, s, e in ents:
                    gold_entities_pool.append((toks[s:e], etype, " ".join(toks), s, e))

    # Identify inflected tokens
    inflected_entity_records = []
    for span_toks, etype, sent_str, s_idx, e_idx in gold_entities_pool:
        for t in span_toks:
            base = infer_base_form(t)
            is_inf = (base is not None and base != t)
            if is_inf:
                in_vocab = t in konkani_vocab
                base_in_vocab = base in konkani_vocab
                inflected_entity_records.append({
                    "surface_token": t,
                    "base_token": base,
                    "entity_type": etype,
                    "in_vocab": in_vocab,
                    "base_in_vocab": base_in_vocab,
                    "sentence": sent_str,
                })

    # Stats on inflected tokens
    total_inflected = len(inflected_entity_records)
    inflected_oov = sum(1 for r in inflected_entity_records if not r["in_vocab"])
    inflected_in_vocab = sum(1 for r in inflected_entity_records if r["in_vocab"])
    base_in_vocab_count = sum(1 for r in inflected_entity_records if (not r["in_vocab"] and r["base_in_vocab"]))

    print(f"\n[Inflection Statistics across Konkani Gold Corpora]")
    print(f"  Total Inflected Entity Tokens identified: {total_inflected}")
    print(f"  Inflected Tokens In-Vocabulary:          {inflected_in_vocab} ({inflected_in_vocab/total_inflected*100:.1f}%)")
    print(f"  Inflected Tokens OOV:                    {inflected_oov} ({inflected_oov/total_inflected*100:.1f}%)")
    print(f"  OOV Tokens with Base Form In-Vocabulary: {base_in_vocab_count} ({base_in_vocab_count/inflected_oov*100:.1f}% of OOV)")

    # 3. Cross-reference with Phase 7 errors in the 247-sentence manifest
    manifest_inflected_misses = []
    manifest_collapse_with_inflection = []

    for idx, toks, gold, pred, err in konkani_errors_by_cat["missed_entity"]:
        span_toks = err.get("tokens", [])
        for t in span_toks:
            base = infer_base_form(t)
            if base and base != t:
                manifest_inflected_misses.append({
                    "sent_idx": idx,
                    "tokens": toks,
                    "gold": gold,
                    "pred": pred,
                    "surface_token": t,
                    "base_token": base,
                    "in_vocab": t in konkani_vocab,
                    "base_in_vocab": base in konkani_vocab,
                    "entity_type": err.get("entity_type"),
                })

    for idx, toks, gold, pred, err in konkani_errors_by_cat["complete_sentence_collapse"]:
        ents = extract_entities(gold)
        has_inf = False
        inf_details = []
        for etype, s, e in ents:
            for t in toks[s:e]:
                base = infer_base_form(t)
                if base and base != t:
                    has_inf = True
                    inf_details.append((t, base, t in konkani_vocab, base in konkani_vocab))
        if has_inf:
            manifest_collapse_with_inflection.append({
                "sent_idx": idx,
                "tokens": toks,
                "gold": gold,
                "pred": pred,
                "inflections": inf_details,
            })

    print(f"\n[Overlap with Phase 7 Error Taxonomy (N=247 Manifest)]")
    print(f"  Missed Entity errors involving inflected forms:      {len(manifest_inflected_misses)} / {len(konkani_errors_by_cat['missed_entity'])} ({len(manifest_inflected_misses)/len(konkani_errors_by_cat['missed_entity'])*100:.1f}%)")
    oov_inf_misses = [m for m in manifest_inflected_misses if not m["in_vocab"]]
    print(f"  Missed Entity errors directly involving inflected OOV: {len(oov_inf_misses)} / {len(konkani_errors_by_cat['missed_entity'])} ({len(oov_inf_misses)/len(konkani_errors_by_cat['missed_entity'])*100:.1f}%)")
    print(f"  Complete Collapse errors containing inflected forms:  {len(manifest_collapse_with_inflection)} / {len(konkani_errors_by_cat['complete_sentence_collapse'])} ({len(manifest_collapse_with_inflection)/len(konkani_errors_by_cat['complete_sentence_collapse'])*100:.1f}%)")

    # 4. Counterfactual Test for Konkani: Substitute Base Form in Place of Inflected Form
    print(f"\n[Counterfactual Substitution Test: Inflected -> Base Form]")
    cf_konkani_cases = []
    
    # (a) Canonical original failing sentence
    orig_failing_toks = [
        "पणजींत", "जाल्ल्या", "एका", "विशेष", "परिषदेंत", "प्रमोद", "सावंत", ",", "श्रीपाद", "नाईक",
        "आनी", "दिगंबर", "कामत", "हांणी", "गोयाच्या", "आर्थिक", "विकासा", "खातीर", "पर्यटन", "।"
    ]
    orig_failing_gold = ["B-LOC", "O", "O", "O", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O", "O"]
    cf_konkani_cases.append({
        "id": "Orig_Failing_Sentence",
        "tokens": orig_failing_toks,
        "gold": orig_failing_gold,
        "substitutions": [("पणजींत", "पणजी"), ("गोयाच्या", "गोंय")],
    })

    # (b) Realistic sentences testing inflected OOV forms vs in-vocab base forms
    real_world_test_sentences = [
        {
            "id": "CF_Locative_Panaji_k",
            "tokens": ["काल", "मुखावेलो", "पणजीक", "एका", "समारंभाक", "गेलो", "."],
            "gold": ["O", "O", "B-LOC", "O", "O", "O", "O"],
            "substitutions": [("पणजीक", "पणजी")],
        },
        {
            "id": "CF_Genitive_Vasco_k",
            "tokens": ["मंत्री", "वास्कोक", "उद्घाटना", "खातीर", "पावले", "."],
            "gold": ["O", "B-LOC", "O", "O", "O", "O"],
            "substitutions": [("वास्कोक", "वास्को")],
        },
        {
            "id": "CF_Locative_Mapusa_ant",
            "tokens": ["नवो", "प्रकल्प", "म्हापश्यांत", "सुरू", "जावचो", "आसा", "."],
            "gold": ["O", "O", "B-LOC", "O", "O", "O", "O"],
            "substitutions": [("म्हापश्यांत", "म्हापसा")],
        },
        {
            "id": "CF_Ergative_Sawant_anni",
            "tokens": ["प्रमोद", "सावंतांनी", "नवी", "येवजण", "जाहीर", "केली", "."],
            "gold": ["B-PER", "I-PER", "O", "O", "O", "O", "O"],
            "substitutions": [("सावंतांनी", "सावंत")],
        },
        {
            "id": "CF_Ergative_Naik_anni",
            "tokens": ["श्रीपाद", "नायकांनी", "केंद्रांत", "महत्वाची", "बसका", "घेतली", "."],
            "gold": ["B-PER", "I-PER", "O", "O", "O", "O", "O"],
            "substitutions": [("नायकांनी", "नाईक")],
        },
        {
            "id": "CF_Ergative_Kamat_anni",
            "tokens": ["दिगंबर", "कामातींनी", "मडगांवांत", "कार्यकर्त्यांक", "मार्गदर्शन", "केलें", "."],
            "gold": ["B-PER", "I-PER", "B-LOC", "O", "O", "O", "O"],
            "substitutions": [("कामातींनी", "कामत")],
        },
        {
            "id": "CF_Dative_Goa_k",
            "tokens": ["खूप", "पर्यटक", "गोयाक", "भेट", "दितात", "."],
            "gold": ["O", "O", "B-LOC", "O", "O", "O"],
            "substitutions": [("गोयाक", "गोंय")],
        },
        {
            "id": "CF_Ergative_Ambedkar_anni",
            "tokens": ["डॉ.", "बाबासाहेब", "आंबेडकरांनी", "संविधानाची", "रचना", "केली", "."],
            "gold": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "O"],
            "substitutions": [("आंबेडकरांनी", "आंबेडकर")],
        },
    ]
    cf_konkani_cases.extend(real_world_test_sentences)

    # (c) Add manifest missed instances if any
    seen_sents = set()
    for m in manifest_inflected_misses:
        s_key = " ".join(m["tokens"])
        if s_key not in seen_sents:
            seen_sents.add(s_key)
            cf_konkani_cases.append({
                "id": f"Manifest_Sent_{m['sent_idx']}",
                "tokens": m["tokens"],
                "gold": m["gold"],
                "substitutions": [(m["surface_token"], m["base_token"])],
            })
            if len(cf_konkani_cases) >= 15:
                break

    cf_konkani_results = []
    cf_successes = 0

    for case in cf_konkani_cases:
        orig_toks = case["tokens"]
        orig_preds = predict_single_sentence(konkani_model, orig_toks, "konkani", w_map, c_map)

        # Substitute base forms
        mod_toks = list(orig_toks)
        for surf, base in case["substitutions"]:
            mod_toks = [base if t == surf else t for t in mod_toks]

        mod_preds = predict_single_sentence(konkani_model, mod_toks, "konkani", w_map, c_map)

        fixed_any = False
        details = []
        for surf, base in case["substitutions"]:
            indices = [i for i, t in enumerate(orig_toks) if t == surf]
            for idx in indices:
                orig_tag = orig_preds[idx]
                mod_tag = mod_preds[idx]
                gold_tag = case["gold"][idx]
                recovered = (orig_tag == "O" and mod_tag.startswith("B-")) or (orig_tag == "O" and mod_tag == gold_tag)
                if recovered:
                    fixed_any = True
                details.append({
                    "surface": surf,
                    "base": base,
                    "orig_pred": orig_tag,
                    "mod_pred": mod_tag,
                    "gold": gold_tag,
                    "recovered": recovered,
                })

        if fixed_any:
            cf_successes += 1

        cf_konkani_results.append({
            "id": case["id"],
            "original_sentence": " ".join(orig_toks),
            "modified_sentence": " ".join(mod_toks),
            "original_predictions": orig_preds,
            "modified_predictions": mod_preds,
            "substitutions_evaluated": details,
            "fixed": fixed_any,
        })

    cf_konkani_success_rate = cf_successes / len(cf_konkani_cases) if cf_konkani_cases else 0.0
    print(f"  Counterfactual Konkani Tests Evaluated: {len(cf_konkani_cases)}")
    print(f"  Instances Fixed by Base-Form Substitution: {cf_successes} / {len(cf_konkani_cases)} ({cf_konkani_success_rate*100:.1f}%)")

    for r in cf_konkani_results[:5]:
        print(f"\n  [{r['id']}] Fixed: {r['fixed']}")
        for d in r["substitutions_evaluated"]:
            print(f"    {d['surface']} (orig: {d['orig_pred']}) -> {d['base']} (mod: {d['mod_pred']}, gold: {d['gold']}) => Recovered: {d['recovered']}")

    # =========================================================================
    # PART B: ASSAMESE LONG-SEQUENCE DECAY CONFIRMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: ASSAMESE LONG-SEQUENCE DECAY CONFIRMATION")
    print("=" * 80)

    # 1. Predict Assamese manifest and categorize errors
    assamese_preds = []
    for toks, gold, src in assamese_manifest:
        pred = predict_single_sentence(assamese_model, toks, "assamese", w_map, c_map)
        assamese_preds.append(pred)

    assamese_errors_by_cat = defaultdict(list)
    for idx, ((toks, gold, src), pred) in enumerate(zip(assamese_manifest, assamese_preds)):
        errs = categorize_sentence_errors(toks, gold, pred, idx, src, "assamese", train_vocab_assamese)
        for e in errs:
            assamese_errors_by_cat[e["category"]].append((idx, toks, gold, pred, e))

    print(f"Total Assamese Sentences: {len(assamese_manifest)}")
    print(f"Complete Collapse Errors: {len(assamese_errors_by_cat['complete_sentence_collapse'])}")
    print(f"Missed Entity Errors:     {len(assamese_errors_by_cat['missed_entity'])}")
    print(f"False Positive Errors:    {len(assamese_errors_by_cat['false_positive'])}")

    # 2. Length distribution of Assamese collapse cases
    collapse_lengths = [len(toks) for idx, toks, gold, pred, e in assamese_errors_by_cat["complete_sentence_collapse"]]
    all_lengths = [len(toks) for toks, gold, src in assamese_manifest]

    collapse_lengths.sort()
    n_coll = len(collapse_lengths)
    min_cl = collapse_lengths[0]
    max_cl = collapse_lengths[-1]
    mean_cl = sum(collapse_lengths) / n_coll
    median_cl = collapse_lengths[n_coll // 2]
    q1_cl = collapse_lengths[int(n_coll * 0.25)]
    q3_cl = collapse_lengths[int(n_coll * 0.75)]

    # Length bucket histogram of collapse cases vs all sentences
    bucket_counts_coll = Counter()
    bucket_counts_all = Counter()
    for l in collapse_lengths:
        if l <= 12:
            bucket_counts_coll["L1_Short (<=12)"] += 1
        elif l <= 22:
            bucket_counts_coll["L2_Medium (13-22)"] += 1
        elif l <= 34:
            bucket_counts_coll["L3_Long (23-34)"] += 1
        else:
            bucket_counts_coll["L4_Stress (>34)"] += 1

    for toks, _, _ in assamese_manifest:
        l = len(toks)
        if l <= 12:
            bucket_counts_all["L1_Short (<=12)"] += 1
        elif l <= 22:
            bucket_counts_all["L2_Medium (13-22)"] += 1
        elif l <= 34:
            bucket_counts_all["L3_Long (23-34)"] += 1
        else:
            bucket_counts_all["L4_Stress (>34)"] += 1

    print(f"\n[Assamese Complete Collapse Length Distribution (N={n_coll})]")
    print(f"  Min Length:    {min_cl}")
    print(f"  Max Length:    {max_cl}")
    print(f"  Mean Length:   {mean_cl:.2f}")
    print(f"  Median Length: {median_cl}")
    print(f"  IQR (Q1 - Q3): {q1_cl} - {q3_cl}")
    print("\n  Collapse Breakdown by Length Bucket:")
    for b_name in ["L1_Short (<=12)", "L2_Medium (13-22)", "L3_Long (23-34)", "L4_Stress (>34)"]:
        c_cnt = bucket_counts_coll[b_name]
        a_cnt = bucket_counts_all[b_name]
        c_rate = c_cnt / a_cnt * 100 if a_cnt > 0 else 0
        print(f"    {b_name:20s}: {c_cnt:2d} / {a_cnt:2d} ({c_rate:.1f}% collapse rate)")

    # 3. Disentanglement: Length vs Entity Density
    # Cross-tabulate collapse by (Length Bucket x Entity Count Bucket)
    density_cross_tab = defaultdict(Counter)
    for idx, toks, gold, pred, _ in assamese_errors_by_cat["complete_sentence_collapse"]:
        l = len(toks)
        l_b = "L1_Short (<=12)" if l <= 12 else ("L2_Medium (13-22)" if l <= 22 else ("L3_Long (23-34)" if l <= 34 else "L4_Stress (>34)"))
        n_ents = len(extract_entities(gold))
        d_b = "E1 (1)" if n_ents == 1 else ("E2 (2)" if n_ents == 2 else "E3 (3+)")
        density_cross_tab[l_b][d_b] += 1

    print("\n  Cross-Tabulation: Collapse Cases by Length vs Entity Density:")
    print(f"    {'Length Bucket':20s} | {'E1 (1)':8s} | {'E2 (2)':8s} | {'E3 (3+)':8s}")
    for l_b in ["L1_Short (<=12)", "L2_Medium (13-22)", "L3_Long (23-34)", "L4_Stress (>34)"]:
        print(f"    {l_b:20s} | {density_cross_tab[l_b]['E1 (1)']:8d} | {density_cross_tab[l_b]['E2 (2)']:8d} | {density_cross_tab[l_b]['E3 (3+)']:8d}")

    # 4. Attention Degeneracy vs CRF Emission Logit Inspection on Sentence #200 & Long Collapsed Sents
    print(f"\n[BiLSTM Hidden-State, Attention Weights, and CRF Emission Diagnostics]")
    
    # Isolate sentence #200 and unique longest collapsed sentences
    stress_sents_as = load_conll_data(DATA_DIR / "gold" / "assamese" / "long_multientity_eval_v2.conll")
    s200_toks, s200_gold = stress_sents_as[0]
    s200_pred = predict_single_sentence(assamese_model, s200_toks, "assamese", w_map, c_map)

    seen_collapsed = {" ".join(s200_toks)}
    long_collapsed_samples = [(200, s200_toks, s200_gold, s200_pred)]

    for idx, toks, gold, pred, _ in assamese_errors_by_cat["complete_sentence_collapse"]:
        s_key = " ".join(toks)
        if s_key not in seen_collapsed and len(toks) >= 25:
            seen_collapsed.add(s_key)
            long_collapsed_samples.append((idx, toks, gold, pred))

    long_collapsed_samples.sort(key=lambda x: -len(x[1]))

    diagnostic_signature_results = []
    tokenizer_as = ScriptTokenizer("assamese")

    for s_idx, toks, gold, pred in long_collapsed_samples[:5]:
        max_l = len(toks)
        tok_gs = [tokenizer_as.tokenize_characters(w) for w in toks]
        max_c = max(len(gs) for gs in tok_gs)
        w_b = torch.zeros((1, max_l), dtype=torch.long)
        c_b = torch.zeros((1, max_l, max_c), dtype=torch.long)
        l_b = torch.tensor([LANG_TO_ID["assamese"]], dtype=torch.long)
        m_b = torch.ones((1, max_l), dtype=torch.bool)

        for j, (w, gs) in enumerate(zip(toks, tok_gs)):
            w_b[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
            for k, g in enumerate(gs):
                c_b[0, j, k] = c_map.get(g, c_map.get("<UNK>", 1))

        # Forward pass hook to capture embeddings, lstm outputs, attention weights, and emissions
        with torch.no_grad():
            w_emb = assamese_model.word_embedding(w_b.to(DEVICE))
            c_emb = assamese_model.char_cnn(c_b.to(DEVICE))
            l_emb = assamese_model.lang_embedding(l_b.to(DEVICE), max_l)
            x_in = torch.cat([w_emb, c_emb, l_emb], dim=-1)
            x_in = assamese_model.concat_dropout(x_in)

            # 2. BiLSTM encoder
            lstm_hidden = assamese_model.encoder(x_in, mask=m_b.to(DEVICE), return_hidden=True)

            # 3. Attention Layer
            attended_states, attn_weights_blocks = assamese_model.attention(lstm_hidden, mask=m_b.to(DEVICE))
            # attn_weights_blocks is list of [B, H, S, S]
            attn_weights = attn_weights_blocks[0] if isinstance(attn_weights_blocks, list) else attn_weights_blocks

            # 4. Emission Logits
            if assamese_model.crf_config == "shared":
                emissions = assamese_model.fc_shared(attended_states)
            else:
                emissions = assamese_model.fc_heads[l_b[0].item()](attended_states)

        # Analyze Attention Weights:
        # attn_weights: [1, num_heads, max_l, max_l]
        num_heads = attn_weights.shape[1]
        head_entropies = []
        head_max_weights = []
        for h in range(num_heads):
            # compute entropy per token query: -sum(p * log(p))
            p = attn_weights[0, h]  # [max_l, max_l]
            ent = -(p * torch.log(p + 1e-12)).sum(dim=-1).mean().item()
            head_entropies.append(ent)
            head_max_weights.append(p.max().item())

        mean_entropy = sum(head_entropies) / num_heads
        max_possible_entropy = math.log(max_l)
        entropy_ratio = mean_entropy / max_possible_entropy if max_possible_entropy > 0 else 1.0

        # Analyze Emission Logits:
        # Check emission gap between 'O' (tag 0) and entity tags (tags 1..8)
        o_tag_id = TAG_TO_ID["O"]
        o_logits = emissions[0, :, o_tag_id]
        
        # Max entity logit
        entity_indices = [v for k, v in TAG_TO_ID.items() if k != "O"]
        entity_logits = emissions[0, :, entity_indices]  # [max_l, 8]
        max_entity_logits, _ = entity_logits.max(dim=-1)

        emission_gap = (o_logits - max_entity_logits).mean().item()
        gold_ents = sorted(extract_entities(gold), key=lambda x: (x[1], x[2]))

        # Check emission values specifically at gold entity token positions
        gold_ent_tokens_o_bias = []
        for etype, s, e in gold_ents:
            for pos in range(s, e):
                pos_o = o_logits[pos].item()
                pos_ent = max_entity_logits[pos].item()
                gold_ent_tokens_o_bias.append(pos_o - pos_ent)

        mean_gold_o_bias = sum(gold_ent_tokens_o_bias) / len(gold_ent_tokens_o_bias) if gold_ent_tokens_o_bias else 0.0

        diag_item = {
            "sent_idx": s_idx,
            "length": max_l,
            "gold_entities_count": len(gold_ents),
            "num_heads": num_heads,
            "mean_attention_entropy": round(mean_entropy, 3),
            "max_possible_entropy": round(max_possible_entropy, 3),
            "entropy_ratio": round(entropy_ratio, 3),
            "head_max_weights": [round(w, 3) for w in head_max_weights],
            "mean_emission_gap_O_minus_entity": round(emission_gap, 3),
            "gold_positions_mean_O_bias": round(mean_gold_o_bias, 3),
        }
        diagnostic_signature_results.append(diag_item)

        print(f"  Sentence idx={s_idx:3d} (Len={max_l:2d}, Gold Ents={len(gold_ents)}):")
        print(f"    Attention Entropy: {mean_entropy:.3f} / {max_possible_entropy:.3f} (Ratio: {entropy_ratio:.2f})")
        print(f"    Head Max Weights:  {diag_item['head_max_weights']}")
        print(f"    Emission Gap (O - max_entity): {emission_gap:+.3f} (At Gold Positions: {mean_gold_o_bias:+.3f})")

    # 5. Counterfactual Test for Assamese: Clause / Sub-Span Truncation
    print(f"\n[Counterfactual Truncation Test: Full Sentence -> Truncated Sub-Span]")
    cf_assamese_cases = []
    
    for s_idx, toks, gold, pred in long_collapsed_samples[:10]:
        gold_ents = sorted(extract_entities(gold), key=lambda x: (x[1], x[2]))
        # Identify first entity span
        if not gold_ents:
            continue
        first_etype, first_s, first_e = gold_ents[0]
        # Create a truncated window of ~15-20 tokens around the first 1-2 entities
        trunc_end = min(len(toks), max(first_e + 8, 18))
        trunc_toks = toks[:trunc_end]
        trunc_gold = gold[:trunc_end]

        cf_assamese_cases.append({
            "sent_idx": s_idx,
            "full_len": len(toks),
            "trunc_len": len(trunc_toks),
            "full_toks": toks,
            "trunc_toks": trunc_toks,
            "full_gold": gold,
            "trunc_gold": trunc_gold,
            "target_entity": " ".join(toks[first_s:first_e]),
            "target_type": first_etype,
            "target_span": (first_s, first_e),
        })

    cf_assamese_results = []
    cf_as_successes = 0

    for case in cf_assamese_cases:
        # Full sentence predictions were already evaluated and known to be all-O
        full_preds = predict_single_sentence(assamese_model, case["full_toks"], "assamese", w_map, c_map)
        trunc_preds = predict_single_sentence(assamese_model, case["trunc_toks"], "assamese", w_map, c_map)

        full_ents = extract_entities(full_preds)
        trunc_ents = extract_entities(trunc_preds)

        # Check if the target entity is recovered in the truncated sentence
        target_s, target_e = case["target_span"]
        target_recovered = False
        for etype, s, e in trunc_ents:
            if s == target_s and e == target_e:
                target_recovered = True
                break

        if target_recovered or len(trunc_ents) > 0:
            cf_as_successes += 1

        cf_assamese_results.append({
            "sent_idx": case["sent_idx"],
            "full_length": case["full_len"],
            "truncated_length": case["trunc_len"],
            "target_entity": case["target_entity"],
            "target_type": case["target_type"],
            "full_predictions": full_preds[:case["trunc_len"]],
            "trunc_predictions": trunc_preds,
            "target_recovered": target_recovered,
            "trunc_recovered_entities_count": len(trunc_ents),
        })

    cf_as_success_rate = cf_as_successes / len(cf_assamese_cases) if cf_assamese_cases else 0.0
    print(f"  Counterfactual Assamese Tests Evaluated: {len(cf_assamese_cases)}")
    print(f"  Instances Fixed by Truncation:           {cf_as_successes} / {len(cf_assamese_cases)} ({cf_as_success_rate*100:.1f}%)")

    for r in cf_assamese_results[:5]:
        print(f"\n  [Sent #{r['sent_idx']}] Full Len: {r['full_length']} -> Trunc Len: {r['truncated_length']}")
        print(f"    Target: '{r['target_entity']}' ({r['target_type']}) => Recovered in Truncation: {r['target_recovered']}")
        print(f"    Full Preds on window:  {r['full_predictions']}")
        print(f"    Trunc Preds on window: {r['trunc_predictions']}")

    # Save complete diagnostic summary json
    summary_data = {
        "konkani_inflection": {
            "total_inflected_tokens_in_pool": total_inflected,
            "inflected_in_vocab": inflected_in_vocab,
            "inflected_oov": inflected_oov,
            "inflected_oov_rate": round(inflected_oov / total_inflected * 100, 2),
            "oov_with_base_form_in_vocab": base_in_vocab_count,
            "base_in_vocab_ratio_of_oov": round(base_in_vocab_count / inflected_oov * 100, 2),
            "manifest_missed_entity_inflection_overlap": len(manifest_inflected_misses),
            "manifest_missed_entity_total": len(konkani_errors_by_cat["missed_entity"]),
            "counterfactual_tests_count": len(cf_konkani_cases),
            "counterfactual_success_count": cf_successes,
            "counterfactual_success_rate": round(cf_konkani_success_rate * 100, 2),
            "cf_sample_results": cf_konkani_results[:10],
        },
        "assamese_decay": {
            "total_collapse_cases": n_coll,
            "length_stats": {
                "min": min_cl,
                "max": max_cl,
                "mean": round(mean_cl, 2),
                "median": median_cl,
                "q1": q1_cl,
                "q3": q3_cl,
            },
            "length_bucket_distribution": dict(bucket_counts_coll),
            "all_sentences_length_distribution": dict(bucket_counts_all),
            "density_cross_tabulation": {k: dict(v) for k, v in density_cross_tab.items()},
            "diagnostic_signature_results": diagnostic_signature_results,
            "counterfactual_tests_count": len(cf_assamese_cases),
            "counterfactual_success_count": cf_as_successes,
            "counterfactual_success_rate": round(cf_as_success_rate * 100, 2),
            "cf_sample_results": cf_assamese_results[:10],
        },
    }

    out_json = PROJECT_ROOT / "scratch" / "cycle2_root_cause_diagnostic.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnostic JSON cached at: {out_json.relative_to(PROJECT_ROOT)}")

    return summary_data


if __name__ == "__main__":
    run_diagnostics()
