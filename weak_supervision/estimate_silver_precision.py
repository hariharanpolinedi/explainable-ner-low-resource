"""
Precision, recall, and F1 evaluation engine for weak supervision labelers.

Evaluates token-level and span-level Named Entity Recognition metrics
against gold dev datasets (data/gold/{lang}/dev.conll).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_entities_from_bio(labels: list[str]) -> set[tuple[str, int, int]]:
    """
    Extract entity spans as (entity_type, start_idx, end_idx) from BIO tags.
    end_idx is exclusive.
    """
    entities: set[tuple[str, int, int]] = set()
    n = len(labels)
    i = 0
    while i < n:
        tag = labels[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < n and labels[i] == f"I-{etype}":
                i += 1
            entities.add((etype, start, i))
        else:
            i += 1
    return entities


def load_conll_data(conll_path: Path | str) -> list[tuple[list[str], list[str]]]:
    """Load sentences and labels from a CoNLL formatted file."""
    fpath = Path(conll_path)
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing CoNLL file: {fpath}")

    sentences: list[tuple[list[str], list[str]]] = []
    curr_tokens: list[str] = []
    curr_labels: list[str] = []

    with open(fpath, "r", encoding="utf-8") as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                if curr_tokens:
                    sentences.append((curr_tokens, curr_labels))
                    curr_tokens = []
                    curr_labels = []
                continue

            parts = line_str.split("\t")
            if len(parts) == 2:
                curr_tokens.append(parts[0])
                curr_labels.append(parts[1])

    if curr_tokens:
        sentences.append((curr_tokens, curr_labels))

    return sentences


def evaluate_predictions(
    gold_sentences: list[tuple[list[str], list[str]]],
    pred_labels: list[list[str]],
) -> dict[str, Any]:
    """
    Compute strict span-level Precision, Recall, and F1 per entity type and micro-average.
    """
    if len(gold_sentences) != len(pred_labels):
        raise ValueError(f"Sentence count mismatch: {len(gold_sentences)} vs {len(pred_labels)}")

    types = ["PER", "LOC", "ORG", "MISC"]
    tp_per_type: dict[str, int] = {t: 0 for t in types}
    fp_per_type: dict[str, int] = {t: 0 for t in types}
    fn_per_type: dict[str, int] = {t: 0 for t in types}

    for (tokens, gold_labs), pred_labs in zip(gold_sentences, pred_labels):
        gold_spans = extract_entities_from_bio(gold_labs)
        pred_spans = extract_entities_from_bio(pred_labs)

        for p in pred_spans:
            etype, _, _ = p
            if p in gold_spans:
                if etype in tp_per_type:
                    tp_per_type[etype] += 1
            else:
                if etype in fp_per_type:
                    fp_per_type[etype] += 1

        for g in gold_spans:
            etype, _, _ = g
            if g not in pred_spans:
                if etype in fn_per_type:
                    fn_per_type[etype] += 1

    metrics_per_type: dict[str, dict[str, float]] = {}
    for t in types:
        tp = tp_per_type[t]
        fp = fp_per_type[t]
        fn = fn_per_type[t]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if (tp + fn == 0) else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics_per_type[t] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    total_tp = sum(tp_per_type.values())
    total_fp = sum(fp_per_type.values())
    total_fn = sum(fn_per_type.values())

    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = (2 * total_prec * total_rec) / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0.0

    return {
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(total_prec, 4),
            "recall": round(total_rec, 4),
            "f1": round(total_f1, 4),
        },
        "by_type": metrics_per_type,
    }
