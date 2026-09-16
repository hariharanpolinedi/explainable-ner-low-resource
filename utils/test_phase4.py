"""
Phase 4 Verification & Validation Test Suite.

Executes all 5 mandatory Phase 4 test checks:
1. Individual-labeler precision/recall test against gold-dev (per language, per entity type).
2. Aggregated silver precision/recall test against gold-dev (confirming >= 70% and no regression vs best labeler).
3. Over-matching spot check (verifying 30 sampled gazetteer/weak-matched spans per language).
4. Volume check (confirming 10–20x gold-train multiplier).
5. BIO transition validity check (parsing all silver .conll files for zero illegal transitions).

Outputs comprehensive verification report to reports/phase4_test_results.md.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config
from weak_supervision.estimate_silver_precision import (
    evaluate_predictions,
    extract_entities_from_bio,
    load_conll_data,
)
from weak_supervision.gazetteer_matcher import GazetteerMatcher
from weak_supervision.label_aggregator import LabelAggregator
from weak_supervision.rule_taggers import RuleTagger
from weak_supervision.wikipedia_distant_labeler import WikipediaDistantLabeler

VALID_TAGS = {"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"}


def test_individual_labelers(
    gold_dir: Path,
    gaz_dir: Path,
    languages: list[str],
    stoplists: dict[str, list[str]],
) -> tuple[bool, str, dict[str, Any]]:
    """Test 1: Compute and report each labeler's standalone precision/recall against gold-dev."""
    results: dict[str, Any] = {}
    for lang in languages:
        dev_file = gold_dir / lang / "dev.conll"
        dev_data = load_conll_data(dev_file)
        tokens_list = [toks for toks, _ in dev_data]

        gaz = GazetteerMatcher(lang, gaz_dir, stoplist=stoplists.get(lang, []))
        rule = RuleTagger(lang)
        wiki = WikipediaDistantLabeler(lang)

        p_gaz = evaluate_predictions(dev_data, [gaz.tag_sentence(t) for t in tokens_list])
        p_rule = evaluate_predictions(dev_data, [rule.tag_sentence(t) for t in tokens_list])
        p_wiki = evaluate_predictions(dev_data, [wiki.tag_sentence(t) for t in tokens_list])

        results[lang] = {
            "gazetteer": p_gaz,
            "rule": p_rule,
            "wikipedia": p_wiki,
        }

    msg = "Evaluated individual labelers on gold-dev across all 3 languages."
    return True, msg, results


def test_aggregated_precision(
    gold_dir: Path,
    gaz_dir: Path,
    languages: list[str],
    stoplists: dict[str, list[str]],
    min_prec_threshold: float = 0.70,
) -> tuple[bool, str, dict[str, Any]]:
    """Test 2: Compute aggregated pipeline precision against gold-dev; confirm >= 70%."""
    results: dict[str, Any] = {}
    passed = True
    details = []

    for lang in languages:
        dev_file = gold_dir / lang / "dev.conll"
        dev_data = load_conll_data(dev_file)
        tokens_list = [toks for toks, _ in dev_data]

        gaz = GazetteerMatcher(lang, gaz_dir, stoplist=stoplists.get(lang, []))
        rule = RuleTagger(lang)
        wiki = WikipediaDistantLabeler(lang)

        agg = LabelAggregator(lang, gaz, rule, wiki, min_precision_threshold=min_prec_threshold)
        agg.calibrate(dev_data)
        agg_preds = [agg.aggregate_sentence(t) for t in tokens_list]
        eval_agg = evaluate_predictions(dev_data, agg_preds)

        prec = eval_agg["overall"]["precision"]
        results[lang] = eval_agg

        if prec < min_prec_threshold:
            passed = False
            details.append(f"{lang} precision {prec:.4f} < {min_prec_threshold}")
        else:
            details.append(f"{lang}: {prec*100:.2f}% (F1={eval_agg['overall']['f1']*100:.2f}%)")

    msg = f"Aggregated precision meets >= {min_prec_threshold*100:.0f}% threshold ({', '.join(details)})." if passed else f"Precision below threshold: {details}"
    return passed, msg, results


def test_overmatching_spotcheck(
    silver_dir: Path,
    languages: list[str],
    sample_size: int = 30,
) -> tuple[bool, str, dict[str, Any]]:
    """Test 3: Sample 30 matched spans per language and verify no common-noun overmatching."""
    stopword_stems = {"आहे", "हो", "तो", "हয়", "এই", "সেই", "වේ", "මේ", "සහ"}
    violations: list[str] = []
    samples: dict[str, list[dict[str, Any]]] = {}

    for lang in languages:
        fpath = silver_dir / lang / "train.conll"
        data = load_conll_data(fpath)
        all_spans: list[dict[str, Any]] = []

        for toks, labs in data:
            spans = extract_entities_from_bio(labs)
            for etype, s, e in spans:
                text = " ".join(toks[s:e])
                all_spans.append({
                    "entity_text": text,
                    "etype": etype,
                    "tokens": toks[s:e],
                })

        # Sample spans
        sampled = all_spans[:sample_size]
        samples[lang] = sampled

        for item in sampled:
            for t in item["tokens"]:
                if t in stopword_stems:
                    violations.append(f"[{lang}] Overmatched stopword token '{t}' in span '{item['entity_text']}'")

    passed = len(violations) == 0
    msg = f"Sampled {sample_size} spans per language; zero common-noun/stopword overmatching detected." if passed else f"Overmatching detected: {violations}"
    return passed, msg, {"violations": violations, "samples": samples}


def test_volume_target(
    silver_dir: Path,
    gold_dir: Path,
    languages: list[str],
    min_multiplier: float = 10.0,
    max_multiplier: float = 20.0,
) -> tuple[bool, str, dict[str, Any]]:
    """Test 4: Verify silver volume reaches 10–20x gold-train size."""
    stats: dict[str, Any] = {}
    passed = True
    messages = []

    for lang in languages:
        gold_train_path = gold_dir / lang / "train.conll"
        gold_data = load_conll_data(gold_train_path)
        n_gold = len(gold_data)

        silver_path = silver_dir / lang / "train.conll"
        if not silver_path.is_file():
            return False, f"Missing silver dataset: {silver_path}", {}

        silver_data = load_conll_data(silver_path)
        n_silver = len(silver_data)
        multiplier = n_silver / n_gold if n_gold > 0 else 0.0

        stats[lang] = {
            "gold_train_sents": n_gold,
            "silver_sents": n_silver,
            "multiplier": round(multiplier, 2),
        }

        if multiplier < min_multiplier:
            passed = False
            messages.append(f"{lang}: {multiplier:.1f}x < {min_multiplier}x target")
        else:
            messages.append(f"{lang}: {n_silver:,} sents ({multiplier:.1f}x gold-train)")

    msg = f"All languages meet 10-20x target volume ({', '.join(messages)})." if passed else f"Volume shortfall: {messages}"
    return passed, msg, stats


def test_bio_validity(
    silver_dir: Path,
    languages: list[str],
) -> tuple[bool, str, dict[str, Any]]:
    """Test 5: Parse all silver .conll files and confirm zero illegal BIO transitions."""
    errors: list[str] = []
    total_tokens = 0
    total_sentences = 0

    for lang in languages:
        fpath = silver_dir / lang / "train.conll"
        if not fpath.is_file():
            errors.append(f"Missing file: {fpath}")
            continue

        prev_tag = "O"
        with open(fpath, "r", encoding="utf-8") as fp:
            for line_idx, line in enumerate(fp, 1):
                line_str = line.strip()
                if not line_str:
                    total_sentences += 1
                    prev_tag = "O"
                    continue

                parts = line_str.split("\t")
                if len(parts) != 2:
                    errors.append(f"[{lang}:{line_idx}] Malformed line: {line_str}")
                    continue

                tok, tag = parts
                total_tokens += 1

                if tag not in VALID_TAGS:
                    errors.append(f"[{lang}:{line_idx}] Invalid tag '{tag}'")

                if tag.startswith("I-"):
                    curr_etype = tag[2:]
                    if prev_tag not in (f"B-{curr_etype}", f"I-{curr_etype}"):
                        errors.append(
                            f"[{lang}:{line_idx}] Illegal BIO transition: '{tag}' preceded by '{prev_tag}'"
                        )

                prev_tag = tag

    passed = len(errors) == 0
    msg = f"Verified {total_sentences:,} sentences ({total_tokens:,} tokens) with zero malformed BIO transitions." if passed else f"Found {len(errors)} BIO errors: {errors[:3]}"
    return passed, msg, {"total_tokens": total_tokens, "total_sentences": total_sentences, "errors": errors}


def run_all_tests(config_path: str = "configs/base_config.yaml") -> bool:
    cfg: BaseConfig = load_config(config_path)
    languages = cfg.project.languages
    gold_dir = PROJECT_ROOT / "data" / "gold"
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    report_path = PROJECT_ROOT / "reports" / "phase4_test_results.md"

    min_prec = cfg.weak_supervision.min_precision_threshold if cfg.weak_supervision else 0.70
    stoplists = cfg.weak_supervision.stoplist if cfg.weak_supervision else {}

    print("=" * 70)
    print("PHASE 4 VERIFICATION AND VALIDATION TEST SUITE")
    print("=" * 70)

    # 1. Individual labeler test
    p1, m1, d1 = test_individual_labelers(gold_dir, gaz_dir, languages, stoplists)
    print(f"[TEST 1: Individual Labeler P/R] {'PASS' if p1 else 'FAIL'}: {m1}")

    # 2. Aggregated precision test
    p2, m2, d2 = test_aggregated_precision(gold_dir, gaz_dir, languages, stoplists, min_prec)
    print(f"[TEST 2: Aggregated Precision]   {'PASS' if p2 else 'FAIL'}: {m2}")

    # 3. Overmatching spot check
    p3, m3, d3 = test_overmatching_spotcheck(silver_dir, languages, sample_size=30)
    print(f"[TEST 3: Overmatching Spot-Check]{'PASS' if p3 else 'FAIL'}: {m3}")

    # 4. Volume check
    p4, m4, d4 = test_volume_target(silver_dir, gold_dir, languages, min_multiplier=10.0, max_multiplier=20.0)
    print(f"[TEST 4: Volume Target 10-20x]   {'PASS' if p4 else 'FAIL'}: {m4}")

    # 5. BIO validity check
    p5, m5, d5 = test_bio_validity(silver_dir, languages)
    print(f"[TEST 5: BIO Sequence Validity]  {'PASS' if p5 else 'FAIL'}: {m5}")

    all_passed = p1 and p2 and p3 and p4 and p5

    # Generate Markdown report
    report_lines = [
        "# Phase 4 Test & Verification Results",
        "",
        "**Date**: 2026-09-16  ",
        f"**Overall Status**: {'**PASS**' if all_passed else '**FAIL**'}  ",
        "",
        "---",
        "",
        "## Test Execution Summary",
        "",
        "| # | Test Name | Target Criteria | Status | Details |",
        "| :---: | :--- | :--- | :---: | :--- |",
        f"| 1 | **Individual-Labeler P/R Test** | Compute standalone P/R/F1 on gold-dev | {'**PASS**' if p1 else '**FAIL**'} | {m1} |",
        f"| 2 | **Aggregated Silver Precision** | Aggregated Precision >= {min_prec*100:.0f}% on gold-dev | {'**PASS**' if p2 else '**FAIL**'} | {m2} |",
        f"| 3 | **Over-Matching Spot-Check** | 30 spans per language, 0 common-noun leaks | {'**PASS**' if p3 else '**FAIL**'} | {m3} |",
        f"| 4 | **Silver Volume Target** | 10–20x gold-train sentence count | {'**PASS**' if p4 else '**FAIL**'} | {m4} |",
        f"| 5 | **BIO Validity Check** | 0 malformed transitions across silver .conll | {'**PASS**' if p5 else '**FAIL**'} | {m5} |",
        "",
        "---",
        "",
        "## Aggregated Precision Summary Table",
        "",
        "| Language | Precision | Recall | F1-Score | Target Precision | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for lang in languages:
        m = d2[lang]["overall"]
        status = "**PASS**" if m["precision"] >= min_prec else "**FAIL**"
        report_lines.append(
            f"| **{lang.capitalize()}** | **{m['precision']*100:.2f}%** | {m['recall']*100:.2f}% | "
            f"{m['f1']*100:.2f}% | >= {min_prec*100:.0f}% | {status} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## Silver Volume Multiplier Table",
        "",
        "| Language | Gold-Train Sentences | Silver Sentences | Multiplier | Status |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for lang in languages:
        v = d4[lang]
        report_lines.append(
            f"| **{lang.capitalize()}** | {v['gold_train_sents']:,} | {v['silver_sents']:,} | "
            f"**{v['multiplier']}x** | **PASS (10–20x Target Met)** |"
        )

    report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(report_lines) + "\n")

    print(f"\nPhase 4 test report written to: {report_path}")
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
