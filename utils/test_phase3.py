"""
Phase 3 Verification & Validation Suite.

Executes all 4 mandatory testing checks:
1. Tagset and BIO Sequence Invariant Check
2. Inter-Annotator Agreement Check (Cohen's kappa >= 0.75)
3. Split-Balance Check (No missing entity categories in any split)
4. Test-Lock Cryptographic Checksum Verification

Outputs comprehensive verification report to reports/phase3_test_results.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.agreement_calculator import compute_language_agreement
from utils.config_loader import BaseConfig, load_config

VALID_TAGS = {"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"}


def test_tagset_and_bio_invariants(gold_dir: Path, languages: list[str]) -> tuple[bool, str, dict[str, Any]]:
    """Test 1: Check that all tags are valid BIO tags and no illegal transitions exist."""
    errors: list[str] = []
    file_stats: dict[str, Any] = {}

    for lang in languages:
        for split in ["train", "dev", "test"]:
            fpath = gold_dir / lang / f"{split}.conll"
            if not fpath.is_file():
                errors.append(f"Missing file: {fpath}")
                continue

            sent_count = 0
            tok_count = 0
            prev_tag = "O"

            with open(fpath, "r", encoding="utf-8") as fp:
                for line_idx, line in enumerate(fp, 1):
                    line_str = line.strip()
                    if not line_str:
                        sent_count += 1
                        prev_tag = "O"
                        continue

                    parts = line_str.split("\t")
                    if len(parts) != 2:
                        errors.append(f"[{fpath.name}:{line_idx}] Malformed line: {line_str}")
                        continue

                    tok, tag = parts
                    tok_count += 1

                    if tag not in VALID_TAGS:
                        errors.append(f"[{fpath.name}:{line_idx}] Invalid tag '{tag}'")

                    if tag.startswith("I-"):
                        curr_etype = tag[2:]
                        if prev_tag not in (f"B-{curr_etype}", f"I-{curr_etype}"):
                            errors.append(
                                f"[{fpath.name}:{line_idx}] Illegal transition: '{tag}' preceded by '{prev_tag}'"
                            )

                    prev_tag = tag

            file_stats[f"{lang}_{split}"] = {
                "sentences": sent_count,
                "tokens": tok_count,
            }

    passed = len(errors) == 0
    msg = "All 9 CoNLL files strictly conform to the 9-class BIO tagset with 0 illegal transitions." if passed else f"Found {len(errors)} errors: {errors[:3]}"
    return passed, msg, {"file_stats": file_stats, "errors": errors}


def test_agreement_threshold(annotation_dir: Path, languages: list[str], threshold: float = 0.75) -> tuple[bool, str, dict[str, Any]]:
    """Test 2: Cohen's kappa >= 0.75 on 100-sentence overlap sample."""
    results: dict[str, Any] = {}
    passed = True

    for lang in languages:
        m = compute_language_agreement(lang, annotation_dir)
        results[lang] = m
        if m["kappa"] < threshold:
            passed = False

    msg = f"All 3 languages achieved Cohen's kappa >= {threshold} (Konkani: {results['konkani']['kappa']}, Assamese: {results['assamese']['kappa']}, Sinhala: {results['sinhala']['kappa']})." if passed else "At least one language failed agreement threshold."
    return passed, msg, results


def test_split_balance(gold_dir: Path, languages: list[str]) -> tuple[bool, str, dict[str, Any]]:
    """Test 3: Confirm all 4 entity types (PER, LOC, ORG, MISC) are present in every split."""
    table: dict[str, dict[str, dict[str, int]]] = {}
    missing_classes: list[str] = []

    for lang in languages:
        table[lang] = {}
        for split in ["train", "dev", "test"]:
            fpath = gold_dir / lang / f"{split}.conll"
            counts = Counter()
            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    line_str = line.strip()
                    if line_str:
                        _, tag = line_str.split("\t")
                        if tag.startswith("B-"):
                            counts[tag[2:]] += 1

            table[lang][split] = dict(counts)
            for etype in ["PER", "LOC", "ORG", "MISC"]:
                if counts[etype] == 0:
                    missing_classes.append(f"{lang}_{split} missing {etype}")

    passed = len(missing_classes) == 0
    msg = "All splits across all 3 languages contain non-zero counts of PER, LOC, ORG, and MISC." if passed else f"Missing classes detected: {missing_classes}"
    return passed, msg, {"table": table, "missing": missing_classes}


def test_test_lock_checksums(gold_dir: Path, lockfile_path: Path, languages: list[str]) -> tuple[bool, str, dict[str, Any]]:
    """Test 4: Verify that test.conll actual SHA-256 matches the recorded checksums in lockfile."""
    if not lockfile_path.is_file():
        return False, f"Missing lockfile: {lockfile_path}", {}

    with open(lockfile_path, "r", encoding="utf-8") as fp:
        lockfile_content = fp.read()

    checksum_results: dict[str, Any] = {}
    passed = True

    for lang in languages:
        test_file = gold_dir / lang / "test.conll"
        if not test_file.is_file():
            return False, f"Missing test file: {test_file}", {}

        # Compute actual hash
        h = hashlib.sha256()
        with open(test_file, "rb") as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()

        # Check if actual_sha exists in lockfile
        if actual_sha not in lockfile_content:
            passed = False
            match = False
        else:
            match = True

        checksum_results[lang] = {
            "actual_sha256": actual_sha,
            "matched_in_lockfile": match,
        }

    msg = "All gold test set checksums match reports/gold_test_lockfile.md exactly." if passed else "Checksum mismatch against lockfile!"
    return passed, msg, checksum_results


def run_all_tests(config_path: str = "configs/base_config.yaml") -> bool:
    cfg: BaseConfig = load_config(config_path)
    languages = cfg.project.languages
    gold_dir = PROJECT_ROOT / "data" / "gold"
    annotation_dir = PROJECT_ROOT / "annotation"
    lockfile_path = PROJECT_ROOT / "reports" / "gold_test_lockfile.md"
    report_path = PROJECT_ROOT / "reports" / "phase3_test_results.md"

    threshold = cfg.annotation.kappa_threshold if cfg.annotation else 0.75

    print("=" * 70)
    print("PHASE 3 VERIFICATION AND VALIDATION SUITE")
    print("=" * 70)

    # 1. Tagset & BIO Transitions
    p1, m1, d1 = test_tagset_and_bio_invariants(gold_dir, languages)
    print(f"[TEST 1: Tagset & BIO Invariants] {'PASS' if p1 else 'FAIL'}: {m1}")

    # 2. Inter-Annotator Agreement
    p2, m2, d2 = test_agreement_threshold(annotation_dir, languages, threshold)
    print(f"[TEST 2: Agreement Threshold]    {'PASS' if p2 else 'FAIL'}: {m2}")

    # 3. Split Balance
    p3, m3, d3 = test_split_balance(gold_dir, languages)
    print(f"[TEST 3: Split Balance]          {'PASS' if p3 else 'FAIL'}: {m3}")

    # 4. Checksum Verification
    p4, m4, d4 = test_test_lock_checksums(gold_dir, lockfile_path, languages)
    print(f"[TEST 4: Test Lock Checksums]    {'PASS' if p4 else 'FAIL'}: {m4}")

    all_passed = p1 and p2 and p3 and p4

    # Generate Markdown Report
    report_lines = [
        "# Phase 3 Test & Verification Results",
        "",
        "**Date**: 2026-09-16  ",
        f"**Overall Status**: {'**PASS**' if all_passed else '**FAIL**'}  ",
        "",
        "---",
        "",
        "## Test Execution Summary",
        "",
        "| # | Test Check | Target Criteria | Status | Details |",
        "| :---: | :--- | :--- | :---: | :--- |",
        f"| 1 | **Tagset & BIO Transitions** | 9-class BIO, 0 illegal transitions | {'**PASS**' if p1 else '**FAIL**'} | {m1} |",
        f"| 2 | **Inter-Annotator Agreement** | Cohen's κ >= {threshold:.2f} | {'**PASS**' if p2 else '**FAIL**'} | {m2} |",
        f"| 3 | **Split-Balance Check** | Non-zero PER/LOC/ORG/MISC in all splits | {'**PASS**' if p3 else '**FAIL**'} | {m3} |",
        f"| 4 | **Test-Lock Checksum Check** | SHA-256 matches lockfile exactly | {'**PASS**' if p4 else '**FAIL**'} | {m4} |",
        "",
        "---",
        "",
        "## Detailed Split Balance Table",
        "",
        "| Language | Split | PER | LOC | ORG | MISC | Total Entities |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for lang in languages:
        for split in ["train", "dev", "test"]:
            c = d3["table"][lang][split]
            report_lines.append(
                f"| **{lang.capitalize()}** | `{split}` | {c.get('PER', 0)} | {c.get('LOC', 0)} | "
                f"{c.get('ORG', 0)} | {c.get('MISC', 0)} | **{sum(c.values())}** |"
            )

    report_lines.extend([
        "",
        "---",
        "",
        "## Cryptographic Checksum Lock Verification",
        "",
        "| Language | File | SHA-256 Checksum | Verified in Lockfile |",
        "| :--- | :--- | :--- | :---: |",
    ])

    for lang in languages:
        cinfo = d4[lang]
        report_lines.append(
            f"| **{lang.capitalize()}** | `data/gold/{lang}/test.conll` | `{cinfo['actual_sha256']}` | {'**VERIFIED**' if cinfo['matched_in_lockfile'] else '**MISMATCH**'} |"
        )

    report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(report_lines) + "\n")

    print(f"\nPhase 3 test report written to: {report_path}")
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
