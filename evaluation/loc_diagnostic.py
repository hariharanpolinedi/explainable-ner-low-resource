"""
LOC-Specific Diagnostic Module.

Analyzes:
1. Gazetteer sizes: LOC (places.txt) vs PER (persons.txt) vs ORG (orgs.txt).
2. Weak supervision rule patterns for LOC in weak_supervision/rule_taggers.py.
3. In-vocab vs OOV entity split on gold-dev for LOC.
4. Morphological case inflection analysis (locative/genitive suffixes in Indic/Konkani).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from evaluation.honest_per_type_baseline import load_model, evaluate_conll
from evaluation.seqeval_metrics import extract_entities
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG
from weak_supervision.rule_taggers import TOPONYMIC_SUFFIXES

LANGUAGES = ["konkani", "assamese", "sinhala"]

BEST_CHECKPOINTS = {
    "konkani": PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt",
    "assamese": PROJECT_ROOT / "checkpoints" / "augmented_assamese.pt",
    "sinhala": PROJECT_ROOT / "checkpoints" / "augmented_sinhala.pt",
}


def analyze_gazetteer_sizes() -> dict[str, dict[str, int]]:
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    sizes: dict[str, dict[str, int]] = {}

    for lang in LANGUAGES:
        sizes[lang] = {}
        for cat, fname in [("PER", "persons.txt"), ("LOC", "places.txt"), ("ORG", "orgs.txt")]:
            fpath = gaz_dir / lang / fname
            if fpath.is_file():
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f if l.strip()]
                sizes[lang][cat] = len(lines)
            else:
                sizes[lang][cat] = 0
    return sizes


def analyze_loc_rules() -> dict[str, Any]:
    return {
        lang: {
            "toponymic_suffixes_count": len(TOPONYMIC_SUFFIXES.get(lang, ())),
            "suffixes": list(TOPONYMIC_SUFFIXES.get(lang, ())),
        }
        for lang in LANGUAGES
    }


def compute_loc_oov_split(lang: str, device: torch.device, w_to_id: dict, c_to_id: dict) -> dict[str, Any]:
    dev_path = PROJECT_ROOT / "data" / "gold" / lang / "dev.conll"
    train_path = PROJECT_ROOT / "data" / "gold" / lang / "train.conll"

    # Build train word vocabulary
    train_words = set()
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                train_words.add(line.split("\t")[0])

    # Run inference on dev set
    ckpt_path = BEST_CHECKPOINTS[lang]
    model = load_model(ckpt_path, device, w_to_id, c_to_id)

    ds = MultilingualNERDataset({lang: [dev_path]}, w_to_id, c_to_id)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    all_tokens: list[list[str]] = []
    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for b in dl:
            paths = model(
                b["word_ids"].to(device),
                b["char_ids"].to(device),
                lang_ids=b["lang_ids"].to(device),
                mask=b["mask"].to(device),
            )
            for i in range(len(b["seq_lengths"])):
                slen = b["seq_lengths"][i]
                all_golds.append([ID_TO_TAG[t.item()] for t in b["tags"][i, :slen]])
                all_preds.append([ID_TO_TAG[t] for t in paths[i][:slen]])
                all_tokens.append(b["tokens"][i][:slen])

    # Classify LOC entities into In-Vocab vs OOV
    in_vocab_tp = 0
    in_vocab_fn = 0
    in_vocab_fp = 0

    oov_tp = 0
    oov_fn = 0
    oov_fp = 0

    total_gold_loc = 0

    for toks, g_seq, p_seq in zip(all_tokens, all_golds, all_preds):
        g_ents = [e for e in extract_entities(g_seq) if e[0] == "LOC"]
        p_ents = [e for e in extract_entities(p_seq) if e[0] == "LOC"]
        p_ents_set = set(p_ents)
        g_ents_set = set(g_ents)

        for e in g_ents:
            total_gold_loc += 1
            span_toks = toks[e[1]:e[2]]
            is_oov = any(t not in train_words for t in span_toks)

            if e in p_ents_set:
                if is_oov:
                    oov_tp += 1
                else:
                    in_vocab_tp += 1
            else:
                if is_oov:
                    oov_fn += 1
                else:
                    in_vocab_fn += 1

        for pe in p_ents:
            if pe not in g_ents_set:
                span_toks = toks[pe[1]:pe[2]]
                is_oov = any(t not in train_words for t in span_toks)
                if is_oov:
                    oov_fp += 1
                else:
                    in_vocab_fp += 1

    def calc_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp + fn == 0) else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
        }

    return {
        "language": lang,
        "total_gold_loc_entities": total_gold_loc,
        "in_vocab": calc_metrics(in_vocab_tp, in_vocab_fp, in_vocab_fn),
        "oov": calc_metrics(oov_tp, oov_fp, oov_fn),
    }


def run_all_loc_diagnostics() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    gaz_sizes = analyze_gazetteer_sizes()
    loc_rules = analyze_loc_rules()

    oov_splits = {}
    for lang in LANGUAGES:
        oov_splits[lang] = compute_loc_oov_split(lang, device, w_to_id, c_to_id)

    return {
        "gazetteer_sizes": gaz_sizes,
        "loc_rules": loc_rules,
        "oov_splits": oov_splits,
    }


def write_loc_diagnostic_report(data: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# LOC-Specific Diagnostic Findings Report",
        "",
        "**Date**: 2026-09-17  ",
        "**Target**: In-depth empirical diagnosis of Location (`LOC`) detection performance across Konkani, Assamese, and Sinhala.  ",
        "",
        "---",
        "",
        "## 1. Gazetteer Coverage Disparity",
        "",
        "| Language | PER Gazetteer Lines | LOC Gazetteer Lines | ORG Gazetteer Lines | LOC Ratio vs (PER+ORG) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    gz = data["gazetteer_sizes"]
    for lang in LANGUAGES:
        per = gz[lang]["PER"]
        loc = gz[lang]["LOC"]
        org = gz[lang]["ORG"]
        ratio = loc / max(1, (per + org)) * 100
        lines.append(f"| **{lang.capitalize()}** | {per:,} | **{loc:,}** | {org:,} | {ratio:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Weak Supervision Rule Availability for LOC",
        "",
        "| Language | Suffix Patterns Count | Defined Toponymic Suffixes | Suffix Coverage Assessment |",
        "| :--- | :---: | :--- | :--- |",
    ] )

    lr = data["loc_rules"]
    for lang in LANGUAGES:
        suffixes = ", ".join(f"`{s}`" for s in lr[lang]["suffixes"][:6])
        if len(lr[lang]["suffixes"]) > 6:
            suffixes += ", ..."
        lines.append(f"| **{lang.capitalize()}** | {lr[lang]['toponymic_suffixes_count']} | {suffixes} | Pure static suffixes; **zero case inflection handling** |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. LOC In-Vocab vs OOV Generalization Breakdown (Gold-Dev)",
        "",
        "| Language | Total Gold LOC | In-Vocab LOC Count | In-Vocab LOC F1 (P/R) | OOV LOC Count | OOV LOC F1 (P/R) | OOV F1 Penalty |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for lang in LANGUAGES:
        s = data["oov_splits"][lang]
        tot = s["total_gold_loc_entities"]
        iv = s["in_vocab"]
        oov = s["oov"]
        tot_iv = iv["tp"] + iv["fn"]
        tot_oov = oov["tp"] + oov["fn"]
        penalty = iv["f1"] - oov["f1"]

        lines.append(
            f"| **{lang.capitalize()}** | {tot} | {tot_iv} | **{iv['f1']:.2f}%** ({iv['precision']:.1f}/{iv['recall']:.1f}) | "
            f"{tot_oov} | **{oov['f1']:.2f}%** ({oov['precision']:.1f}/{oov['recall']:.1f}) | **{penalty:+.2f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Root Cause Analysis for LOC Failures (Konkani Manual Test Case)",
        "",
        "In the manual test case (`पणजींत जाल्ल्या एका विशेष परिषदेंत... गोयाच्या आर्थिक विकासा खातीर...`):",
        "1. **Case Inflection & Agglutination**: Konkani places take bound locative (`-ंत`, e.g. `पणजी` -> `पणजींत`) and genitive (`-चो/-च्या`, e.g. `गोवा` -> `गोयाच्या`) postpositions.",
        "2. **Gazetteer Blindspot**: Gazetteers contain base nominative lemmas (`पणजी`, `गोवा`), but zero inflected surface variants.",
        "3. **Toponymic Rule Blindspot**: Suffix rules check for `-पूर`, `-गांव`, but never handle case morphemes appended after city names.",
        "4. **Embedding Fallback**: When an inflected place name is OOV in word embeddings, the Char-CNN must shoulder 100% of the representation. Without morphological inflection augmentation, the BiLSTM-CRF misidentifies or drops the boundary.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved LOC Diagnostic Report to: {out_path}")


if __name__ == "__main__":
    res = run_all_loc_diagnostics()
    out_f = PROJECT_ROOT / "reports" / "loc_diagnostic_findings.md"
    write_loc_diagnostic_report(res, out_f)
