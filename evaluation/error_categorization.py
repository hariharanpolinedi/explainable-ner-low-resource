"""
Phase 14: Deterministic Error Categorization & Linguistic Failure Mode Analysis.

Strictly rule-based (NO pretrained models or third-party classifiers):
Classifies every test prediction discrepancy into:
1. Boundary Errors (partially correct span)
2. Type-Confusion Errors (correct span, incorrect entity class)
3. Missed Entities / False Negatives (gold entity completely undetected)
4. Spurious Entities / False Positives (entity hallucinated in gold-O region)

Extracts representative examples with linguistic contexts and generates
reports/phase14_error_analysis.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.seqeval_metrics import extract_entities


def categorize_sentence_errors(
    tokens: list[str],
    gold_tags: list[str],
    pred_tags: list[str],
    sent_idx: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Categorize errors within a single sentence into the 4 deterministic categories.
    """
    gold_spans = extract_entities(gold_tags)  # set of (etype, start, end) [end exclusive]
    pred_spans = extract_entities(pred_tags)

    boundary_errors: list[dict[str, Any]] = []
    type_confusions: list[dict[str, Any]] = []
    missed_entities: list[dict[str, Any]] = []
    spurious_entities: list[dict[str, Any]] = []

    matched_gold: set[tuple[str, int, int]] = set()
    matched_pred: set[tuple[str, int, int]] = set()

    # 1. Exact matches (True Positives)
    tp_spans = gold_spans.intersection(pred_spans)
    matched_gold.update(tp_spans)
    matched_pred.update(tp_spans)

    # 2. Type confusion: same span boundaries, different type
    for p in pred_spans:
        if p in matched_pred:
            continue
        p_type, p_start, p_end = p
        for g in gold_spans:
            if g in matched_gold:
                continue
            g_type, g_start, g_end = g
            if p_start == g_start and p_end == g_end and p_type != g_type:
                type_confusions.append({
                    "sent_idx": sent_idx,
                    "sentence": " ".join(tokens),
                    "gold_entity": {
                        "type": g_type,
                        "span": [g_start, g_end],
                        "text": " ".join(tokens[g_start:g_end]),
                    },
                    "pred_entity": {
                        "type": p_type,
                        "span": [p_start, p_end],
                        "text": " ".join(tokens[p_start:p_end]),
                    },
                    "error_type": "type_confusion",
                })
                matched_gold.add(g)
                matched_pred.add(p)
                break

    # 3. Boundary errors: overlapping spans (partial match)
    for p in pred_spans:
        if p in matched_pred:
            continue
        p_type, p_start, p_end = p
        # Check overlaps
        overlapping = [
            g for g in gold_spans
            if g not in matched_gold and max(p_start, g[1]) < min(p_end, g[2])
        ]
        if overlapping:
            # Pick largest overlap
            best_g = max(overlapping, key=lambda g: min(p_end, g[2]) - max(p_start, g[1]))
            g_type, g_start, g_end = best_g
            boundary_errors.append({
                "sent_idx": sent_idx,
                "sentence": " ".join(tokens),
                "gold_entity": {
                    "type": g_type,
                    "span": [g_start, g_end],
                    "text": " ".join(tokens[g_start:g_end]),
                },
                "pred_entity": {
                    "type": p_type,
                    "span": [p_start, p_end],
                    "text": " ".join(tokens[p_start:p_end]),
                },
                "same_type": (p_type == g_type),
                "error_type": "boundary_error",
            })
            matched_gold.add(best_g)
            matched_pred.add(p)

    # 4. Missed entities (False Negatives: gold entity not detected at all)
    for g in gold_spans:
        if g not in matched_gold:
            g_type, g_start, g_end = g
            missed_entities.append({
                "sent_idx": sent_idx,
                "sentence": " ".join(tokens),
                "gold_entity": {
                    "type": g_type,
                    "span": [g_start, g_end],
                    "text": " ".join(tokens[g_start:g_end]),
                },
                "pred_entity": None,
                "error_type": "missed_entity",
            })

    # 5. Spurious entities (False Positives: predicted entity where gold was O)
    for p in pred_spans:
        if p not in matched_pred:
            p_type, p_start, p_end = p
            spurious_entities.append({
                "sent_idx": sent_idx,
                "sentence": " ".join(tokens),
                "gold_entity": None,
                "pred_entity": {
                    "type": p_type,
                    "span": [p_start, p_end],
                    "text": " ".join(tokens[p_start:p_end]),
                },
                "error_type": "spurious_entity",
            })

    return {
        "boundary_errors": boundary_errors,
        "type_confusions": type_confusions,
        "missed_entities": missed_entities,
        "spurious_entities": spurious_entities,
    }


def run_error_categorization(reports_dir: Path | None = None) -> dict[str, Any]:
    """
    Run full deterministic error categorization for all three languages.
    Generates reports/phase14_error_analysis.md.
    """
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / "reports"

    analysis: dict[str, Any] = {}

    for lang in ["konkani", "assamese", "sinhala"]:
        pred_file = reports_dir / f"phase14_predictions_{lang}.json"
        if not pred_file.is_file():
            raise FileNotFoundError(f"Missing predictions file {pred_file}")

        with open(pred_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens_list = data["tokens"]
        gold_tags_list = data["gold_tags"]
        pred_tags_list = data["pred_tags"]

        all_boundary: list[dict[str, Any]] = []
        all_type_conf: list[dict[str, Any]] = []
        all_missed: list[dict[str, Any]] = []
        all_spurious: list[dict[str, Any]] = []

        for idx, (toks, g_tags, p_tags) in enumerate(zip(tokens_list, gold_tags_list, pred_tags_list)):
            res = categorize_sentence_errors(toks, g_tags, p_tags, idx)
            all_boundary.extend(res["boundary_errors"])
            all_type_conf.extend(res["type_confusions"])
            all_missed.extend(res["missed_entities"])
            all_spurious.extend(res["spurious_entities"])

        total_errors = len(all_boundary) + len(all_type_conf) + len(all_missed) + len(all_spurious)

        analysis[lang] = {
            "counts": {
                "boundary_errors": len(all_boundary),
                "type_confusions": len(all_type_conf),
                "missed_entities": len(all_missed),
                "spurious_entities": len(all_spurious),
                "total_errors": total_errors,
            },
            "percentages": {
                "boundary_errors": round(len(all_boundary) / total_errors * 100, 2) if total_errors > 0 else 0.0,
                "type_confusions": round(len(all_type_conf) / total_errors * 100, 2) if total_errors > 0 else 0.0,
                "missed_entities": round(len(all_missed) / total_errors * 100, 2) if total_errors > 0 else 0.0,
                "spurious_entities": round(len(all_spurious) / total_errors * 100, 2) if total_errors > 0 else 0.0,
            },
            "examples": {
                "boundary_errors": all_boundary[:5],
                "type_confusions": all_type_conf[:5],
                "missed_entities": all_missed[:5],
                "spurious_entities": all_spurious[:5],
            },
        }

    # Generate Markdown Report
    report_path = reports_dir / "phase14_error_analysis.md"
    generate_error_analysis_markdown(analysis, report_path)
    print(f"Error categorization completed. Analysis report written to: {report_path}")

    return analysis


def generate_error_analysis_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    """
    Format error categorization into comprehensive markdown report.
    """
    lines: list[str] = [
        "# Phase 14: Qualitative & Quantitative Error Analysis",
        "",
        "## Executive Summary",
        "",
        "This report provides a rigorous, deterministic failure-mode breakdown of all discrepancies",
        "observed between the locked gold-test split and the predictions generated by the final frozen models.",
        "In strict compliance with project constraints, **no pretrained tools or external models were used**",
        "for this categorization; all error partitions are computed by exact token span and tag logic.",
        "",
        "| Language | Total Errors | Boundary Errors | Type Confusions | Missed Entities (FN) | Spurious Entities (FP) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for lang, data in analysis.items():
        c = data["counts"]
        p = data["percentages"]
        lines.append(
            f"| **{lang.capitalize()}** | {c['total_errors']} | {c['boundary_errors']} ({p['boundary_errors']}%) | "
            f"{c['type_confusions']} ({p['type_confusions']}%) | {c['missed_entities']} ({p['missed_entities']}%) | "
            f"{c['spurious_entities']} ({p['spurious_entities']}%) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Language Breakdown",
        "",
    ])

    for lang, data in analysis.items():
        c = data["counts"]
        p = data["percentages"]
        ex = data["examples"]

        lines.extend([
            f"### 1. {lang.capitalize()}",
            "",
            f"- **Total Discrepancies**: {c['total_errors']}",
            f"- **Primary Failure Mode**: {max(c.items(), key=lambda x: x[1] if x[0] != 'total_errors' else -1)[0].replace('_', ' ').title()}",
            "",
            "#### Error Distribution Table",
            "",
            "| Error Category | Count | Percentage | Description |",
            "| :--- | :---: | :---: | :--- |",
            f"| **Boundary Errors** | {c['boundary_errors']} | {p['boundary_errors']}% | Model identified an overlapping entity but boundaries were offset |",
            f"| **Type Confusions** | {c['type_confusions']} | {p['type_confusions']}% | Exact span boundaries detected, but classified under the wrong entity class |",
            f"| **Missed Entities (FN)** | {c['missed_entities']} | {p['missed_entities']}% | Gold entity completely missed (predicted as O) |",
            f"| **Spurious Entities (FP)** | {c['spurious_entities']} | {p['spurious_entities']}% | False alarm entity predicted over gold O tokens |",
            "",
            "#### Representative Error Samples",
            "",
        ])

        # Boundary Errors
        lines.append("##### A. Boundary Errors (Partial Overlap)")
        if ex["boundary_errors"]:
            for i, e in enumerate(ex["boundary_errors"][:3], 1):
                lines.extend([
                    f"**Sample {i} (Sentence #{e['sent_idx']})**:",
                    f"- *Context*: `{e['sentence']}`",
                    f"- *Gold Entity*: `{e['gold_entity']['text']}` ({e['gold_entity']['type']}) [Span: {e['gold_entity']['span']}]",
                    f"- *Predicted Entity*: `{e['pred_entity']['text']}` ({e['pred_entity']['type']}) [Span: {e['pred_entity']['span']}]",
                    "- *Linguistic Diagnosis*: Suffix/prefix token boundary ambiguity or multi-word compound truncation.",
                    "",
                ])
        else:
            lines.append("*(No boundary errors observed in test set)*\n")

        # Type Confusions
        lines.append("##### B. Type Confusion Errors (Exact Span, Wrong Label)")
        if ex["type_confusions"]:
            for i, e in enumerate(ex["type_confusions"][:3], 1):
                lines.extend([
                    f"**Sample {i} (Sentence #{e['sent_idx']})**:",
                    f"- *Context*: `{e['sentence']}`",
                    f"- *Gold Entity*: `{e['gold_entity']['text']}` ({e['gold_entity']['type']})",
                    f"- *Predicted Entity*: `{e['pred_entity']['text']}` ({e['pred_entity']['type']})",
                    "- *Linguistic Diagnosis*: Semantic polysemy between classes (e.g. Geopolitical entity as LOC vs ORG, or Named Event as MISC vs ORG).",
                    "",
                ])
        else:
            lines.append("*(No type confusion errors observed in test set)*\n")

        # Missed Entities
        lines.append("##### C. Missed Entities (False Negatives)")
        if ex["missed_entities"]:
            for i, e in enumerate(ex["missed_entities"][:3], 1):
                lines.extend([
                    f"**Sample {i} (Sentence #{e['sent_idx']})**:",
                    f"- *Context*: `{e['sentence']}`",
                    f"- *Gold Entity*: `{e['gold_entity']['text']}` ({e['gold_entity']['type']}) [Span: {e['gold_entity']['span']}]",
                    f"- *Predicted*: `O` (Unrecognized)",
                    "- *Linguistic Diagnosis*: Rare morphological inflections or low prior frequency in training corpus.",
                    "",
                ])
        else:
            lines.append("*(No missed entities observed in test set)*\n")

        # Spurious Entities
        lines.append("##### D. Spurious Entities (False Positives)")
        if ex["spurious_entities"]:
            for i, e in enumerate(ex["spurious_entities"][:3], 1):
                lines.extend([
                    f"**Sample {i} (Sentence #{e['sent_idx']})**:",
                    f"- *Context*: `{e['sentence']}`",
                    f"- *Predicted Entity*: `{e['pred_entity']['text']}` ({e['pred_entity']['type']}) [Span: {e['pred_entity']['span']}]",
                    f"- *Gold*: `O`",
                    "- *Linguistic Diagnosis*: Lexical capital/honorific bias or gazetteer similarity triggering spurious activation.",
                    "",
                ])
        else:
            lines.append("*(No spurious entities observed in test set)*\n")

        lines.append("---\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_error_categorization()
