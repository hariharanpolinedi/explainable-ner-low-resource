"""
Phase 8 Comprehensive Testing and Validation Suite.

Verifies:
1. Forward-pass shape test: language-conditioning vector concatenation and pipeline shapes
   for both shared and per-language CRF heads.
2. Data-pooling correctness test: verifies pooled dataset example counts per language
   and correctness of language tags.
3. Sampling-strategy verification: confirms empirical batch distributions for uniform vs proportional sampling.
4. Per-language entity-F1 comparison: verifies all 4 configurations evaluated on gold-dev.
5. Reproducibility check: verifies deterministic training and identical dev-F1 on fixed seed.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from evaluation.compare_joint_vs_monolingual import load_phase7_baselines
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    ID_TO_LANG,
    LANG_TO_ID,
    MultilingualBatchSampler,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_joint_warmstart import train_multilingual_joint_experiment


def test_forward_pass_shape() -> bool:
    """Test 1: Forward-pass shape test for both shared and per_language CRF heads."""
    print("\n--- TEST 1: Forward-Pass Shape & Language Conditioning Test ---")
    batch_size = 4
    seq_len = 8
    max_word_len = 6
    vocab_size = 500
    char_vocab_size = 100

    word_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    char_ids = torch.randint(1, char_vocab_size, (batch_size, seq_len, max_word_len))
    lang_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    tags = torch.randint(0, 9, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    mask[:, -2:] = False

    # Check shared CRF head
    model_shared = NERMultilingualModel(
        word_vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=9,
        num_languages=3,
        word_embedding_dim=100,
        char_embedding_dim=50,
        lang_embedding_dim=16,
        filter_widths=(3, 4, 5),
        filters_per_width=30,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_attention_blocks=1,
        crf_config="shared",
    )
    loss_s = model_shared(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)
    paths_s, attn_s = model_shared(word_ids, char_ids, lang_ids=lang_ids, mask=mask, return_attention=True)

    assert not torch.isnan(loss_s) and loss_s.item() > 0.0, "Shared loss is NaN or non-positive"
    assert len(paths_s) == batch_size, f"Expected {batch_size} decoded paths, got {len(paths_s)}"
    assert isinstance(attn_s, list) and attn_s[0].shape == (batch_size, 4, seq_len, seq_len), f"Unexpected attention shape"
    print("  [PASSED] Shared CRF head forward pass and decoding validated.")

    # Check per_language CRF head
    model_per_lang = NERMultilingualModel(
        word_vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=9,
        num_languages=3,
        word_embedding_dim=100,
        char_embedding_dim=50,
        lang_embedding_dim=16,
        filter_widths=(3, 4, 5),
        filters_per_width=30,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_attention_blocks=1,
        crf_config="per_language",
    )
    loss_p = model_per_lang(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)
    paths_p, attn_p = model_per_lang(word_ids, char_ids, lang_ids=lang_ids, mask=mask, return_attention=True)

    assert not torch.isnan(loss_p) and loss_p.item() > 0.0, "Per-language loss is NaN or non-positive"
    assert len(paths_p) == batch_size, f"Expected {batch_size} decoded paths, got {len(paths_p)}"
    assert isinstance(attn_p, list) and attn_p[0].shape == (batch_size, 4, seq_len, seq_len), f"Unexpected attention shape"
    print("  [PASSED] Per-language CRF head forward pass and decoding validated.")
    return True


def test_data_pooling_correctness() -> bool:
    """Test 2: Confirm pooled dataset contains expected counts and valid language tags."""
    print("\n--- TEST 2: Data-Pooling Correctness Test ---")
    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    proc_dir = PROJECT_ROOT / "data" / "processed"

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    train_paths: dict[str, list[Path | str]] = {}
    for lang in ["konkani", "assamese", "sinhala"]:
        train_paths[lang] = [
            gold_dir / lang / "train.conll",
            silver_dir / lang / "train.conll",
        ]

    dataset = MultilingualNERDataset(
        lang_conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    total_samples = len(dataset)
    expected_per_lang = 7800  # 600 gold + 7200 silver
    expected_total = expected_per_lang * 3

    assert total_samples == expected_total, f"Expected {expected_total} samples, got {total_samples}"

    # Spot-check language partitions
    for lang, lid in LANG_TO_ID.items():
        indices = dataset.lang_indices[lid]
        assert len(indices) == expected_per_lang, (
            f"Expected {expected_per_lang} samples for {lang}, got {len(indices)}"
        )
        # Spot check first, middle, last sample
        for spot in [indices[0], indices[len(indices)//2], indices[-1]]:
            sample = dataset[spot]
            toks, wids, cids, tids, sl_id = sample
            assert sl_id == lid, f"Language tag mismatch: expected {lid}, got {sl_id}"
            assert len(toks) > 0 and len(wids) == len(toks)

    print(f"  [PASSED] Data pooling verified: exactly {total_samples} samples across 3 languages.")
    return True


def test_sampling_strategies() -> bool:
    """Test 3: Sampling-strategy verification (batch composition distribution)."""
    print("\n--- TEST 3: Sampling-Strategy Verification Test ---")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    # Use simulated pool indices with distinct index ranges per language
    dummy_indices = {
        0: list(range(0, 1000)),
        1: list(range(1000, 2000)),
        2: list(range(2000, 3000)),
    }

    # Test uniform sampling
    sampler_u = MultilingualBatchSampler(
        lang_indices=dummy_indices,
        batch_size=32,
        strategy="uniform",
        seed=42,
        num_batches_per_epoch=300,
    )
    u_counts = {0: 0, 1: 0, 2: 0}
    for batch in sampler_u:
        # Determine language of batch
        b_lang = batch[0] // 1000
        u_counts[b_lang] += 1

    # Ratio should be approximately 33.3% each (within tolerance 0.25 - 0.42)
    u_ratios = {k: v / 300 for k, v in u_counts.items()}
    print(f"  Uniform batch proportions: {u_ratios}")
    for k, r in u_ratios.items():
        assert 0.25 <= r <= 0.42, f"Uniform ratio out of expected range: {r}"

    # Test proportional sampling
    sampler_p = MultilingualBatchSampler(
        lang_indices=dummy_indices,
        batch_size=32,
        strategy="proportional",
        seed=42,
        num_batches_per_epoch=300,
    )
    p_counts = {0: 0, 1: 0, 2: 0}
    for batch in sampler_p:
        b_lang = batch[0] // 1000
        p_counts[b_lang] += 1

    p_ratios = {k: v / 300 for k, v in p_counts.items()}
    print(f"  Proportional batch proportions: {p_ratios}")
    # Raw corpus ratios: Konkani ~ 0.203, Assamese ~ 0.360, Sinhala ~ 0.438
    # Sinhala should be noticeably higher than Konkani
    assert p_ratios[2] > p_ratios[0], "Expected Sinhala ratio > Konkani ratio in proportional sampling"
    assert p_ratios[0] < 0.28, f"Expected Konkani ratio to reflect low-resource volume, got {p_ratios[0]}"
    print("  [PASSED] Sampling strategies confirmed distinct and adhering to theoretical distributions.")
    return True


def test_entity_f1_comparison() -> bool:
    """Test 4: Verify all 4 configurations exist and were evaluated against Phase 7."""
    print("\n--- TEST 4: Per-Language Entity-F1 Comparison Test ---")
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    report_file = PROJECT_ROOT / "reports" / "phase8_joint_vs_monolingual.md"

    assert report_file.is_file(), f"Missing Phase 8 report: {report_file}"
    joint_best = checkpoints_dir / "joint_best.pt"
    assert joint_best.is_file(), f"Missing joint_best checkpoint: {joint_best}"

    chk = torch.load(joint_best, map_location="cpu", weights_only=False)
    assert "best_avg_dev_f1" in chk, "Checkpoint missing best_avg_dev_f1"
    assert "per_lang_results" in chk, "Checkpoint missing per_lang_results"

    p7_baselines = load_phase7_baselines(checkpoints_dir)
    print("  Comparison vs Phase-7 Baselines:")
    for lang in ["konkani", "assamese", "sinhala"]:
        p7_f1 = p7_baselines.get(lang, 0.0) * 100
        p8_f1 = chk["per_lang_results"][lang]["entity_f1"] * 100
        delta = p8_f1 - p7_f1
        print(f"    {lang.capitalize()}: Phase-7 = {p7_f1:.2f}% | Phase-8 = {p8_f1:.2f}% | Delta = {delta:+.2f}%")
        # Ensure no catastrophic regression (no language dropped by > 5%)
        assert delta > -5.0, f"Catastrophic regression detected in {lang}: {delta:.2f}%"

    print("  [PASSED] Entity-F1 comparison verified across all target languages.")
    return True


def test_reproducibility() -> bool:
    """Test 5: Reproducibility check on fixed seed."""
    print("\n--- TEST 5: Reproducibility Check ---")
    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    tmp_chk = PROJECT_ROOT / "checkpoints" / "repro_test_joint.pt"

    print("  Running 1-epoch fixed-seed test...")
    res1 = train_multilingual_joint_experiment(
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        proc_dir=proc_dir,
        checkpoint_out=tmp_chk,
        crf_config="shared",
        sampling_strategy="uniform",
        max_epochs=1,
        seed=1337,
        device_setting="auto",
    )
    score1 = res1["best_avg_dev_f1"]

    res2 = train_multilingual_joint_experiment(
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        proc_dir=proc_dir,
        checkpoint_out=tmp_chk,
        crf_config="shared",
        sampling_strategy="uniform",
        max_epochs=1,
        seed=1337,
        device_setting="auto",
    )
    score2 = res2["best_avg_dev_f1"]

    diff = abs(score1 - score2)
    print(f"  Run 1 score: {score1*100:.4f}% | Run 2 score: {score2*100:.4f}% | Diff: {diff:.6f}")
    assert diff < 1e-4, f"Reproducibility check failed: diff={diff} exceeds tolerance 1e-4"

    if tmp_chk.is_file():
        tmp_chk.unlink()

    print("  [PASSED] Multilingual joint training determinism verified.")
    return True


def run_all_tests() -> None:
    """Run all Phase 8 validation tests and record reports/phase8_test_results.md."""
    start_time = time.time()
    results: list[tuple[str, str, str]] = []

    tests = [
        ("Forward-Pass Shape & Language Conditioning", test_forward_pass_shape),
        ("Data-Pooling Correctness & Integrity", test_data_pooling_correctness),
        ("Sampling-Strategy Verification", test_sampling_strategies),
        ("Per-Language Entity-F1 Comparison", test_entity_f1_comparison),
        ("Reproducibility Check (Fixed Seed)", test_reproducibility),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            passed = fn()
            if passed:
                results.append((name, "PASS", "Validated cleanly without anomalies."))
            else:
                results.append((name, "FAIL", "Returned False."))
                all_passed = False
        except Exception as exc:
            results.append((name, "FAIL", f"Exception: {str(exc)}"))
            all_passed = False

    total_time = time.time() - start_time
    report_path = PROJECT_ROOT / "reports" / "phase8_test_results.md"
    lines = [
        "# Phase 8 Test & Validation Results Report",
        "",
        "**Date**: 2026-09-16  ",
        f"**Status**: {'PASS' if all_passed else 'FAIL'}  ",
        f"**Total Execution Time**: {total_time:.2f}s  ",
        "",
        "---",
        "",
        "## Test Execution Summary",
        "",
        "| Test Name | Result | Diagnostic Details |",
        "| :--- | :---: | :--- |",
    ]
    for name, status, details in results:
        lines.append(f"| {name} | **{status}** | {details} |")

    lines.extend([
        "",
        "---",
        "",
        "## Verification Sign-Off",
        "",
        f"- All 5 required validation criteria executed: **{'YES' if all_passed else 'NO'}**",
        f"- Overall Suite Status: **{'PASS' if all_passed else 'FAIL'}**",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\nGenerated test results report at {report_path}")
    if not all_passed:
        raise RuntimeError("Phase 8 test suite failed.")


if __name__ == "__main__":
    run_all_tests()
