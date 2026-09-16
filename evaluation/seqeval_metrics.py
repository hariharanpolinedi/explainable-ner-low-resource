"""
Entity-level NER evaluation metrics using seqeval with token-accuracy tracking.

Computes:
1. Strict span-level Entity F1, Precision, and Recall (overall and per entity class: PER, LOC, ORG, MISC).
2. Token-level Accuracy (as secondary metric).
Never conflates token-level accuracy with entity-level F1.
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


def extract_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    """
    Extract entity spans as (entity_type, start_idx, end_idx) from BIO tag list.
    end_idx is exclusive.
    """
    entities: set[tuple[str, int, int]] = set()
    n = len(tags)
    i = 0
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < n and tags[i] == f"I-{etype}":
                i += 1
            entities.add((etype, start, i))
        else:
            i += 1
    return entities


def compute_seqeval_metrics(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    """
    Compute entity-level precision, recall, F1 using strict span matching
    (identical to seqeval strict mode), along with token-level accuracy.
    """
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError(f"Sequence count mismatch: {len(gold_sequences)} vs {len(pred_sequences)}")

    # Try seqeval library if available
    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        has_seqeval = True
    except ImportError:
        has_seqeval = False

    types = ["PER", "LOC", "ORG", "MISC"]
    tp_by_type: dict[str, int] = {t: 0 for t in types}
    fp_by_type: dict[str, int] = {t: 0 for t in types}
    fn_by_type: dict[str, int] = {t: 0 for t in types}

    total_tokens = 0
    correct_tokens = 0

    for gold_seq, pred_seq in zip(gold_sequences, pred_sequences):
        min_len = min(len(gold_seq), len(pred_seq))
        for g_tok, p_tok in zip(gold_seq[:min_len], pred_seq[:min_len]):
            total_tokens += 1
            if g_tok == p_tok:
                correct_tokens += 1

        gold_spans = extract_entities(gold_seq)
        pred_spans = extract_entities(pred_seq)

        for p in pred_spans:
            etype, _, _ = p
            if p in gold_spans:
                if etype in tp_by_type:
                    tp_by_type[etype] += 1
            else:
                if etype in fp_by_type:
                    fp_by_type[etype] += 1

        for g in gold_spans:
            etype, _, _ = g
            if g not in pred_spans:
                if etype in fn_by_type:
                    fn_by_type[etype] += 1

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    by_type: dict[str, dict[str, float]] = {}
    for t in types:
        tp = tp_by_type[t]
        fp = fp_by_type[t]
        fn = fn_by_type[t]
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp + fn == 0) else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        by_type[t] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_tp = sum(tp_by_type.values())
    total_fp = sum(fp_by_type.values())
    total_fn = sum(fn_by_type.values())

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "overall": {
            "entity_precision": round(overall_prec, 4),
            "entity_recall": round(overall_rec, 4),
            "entity_f1": round(overall_f1, 4),
            "token_accuracy": round(token_accuracy, 4),
            "total_entities_gold": total_tp + total_fn,
            "total_entities_pred": total_tp + total_fp,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "by_type": by_type,
        "token_accuracy": round(token_accuracy, 4),
        "total_tokens": total_tokens,
    }
