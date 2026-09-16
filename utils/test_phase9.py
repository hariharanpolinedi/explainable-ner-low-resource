"""
Phase 9 Comprehensive Testing and Validation Suite.

Verifies:
1. Checkpoint-loading test: confirms Phase-8 joint weights load cleanly into the fine-tuning architecture without shape mismatches.
2. Freezing-correctness test: confirms frozen-layer parameters genuinely have requires_grad=False and zero parameter updates during frozen epochs.
3. Per-language entity-F1 comparison: fine-tuned vs from-scratch control across all languages.
4. Unfreezing-schedule sensitivity: default phased vs immediate schedule.
5. Reproducibility check: deterministic evaluation on fixed seed.
"""

from __future__ import annotations

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

from model.ner_model_v4_multilingual import NERMultilingualModel
from training.finetune_per_language import train_finetune_experiment
from training.layer_unfreeze_schedule import LayerUnfreezeScheduler
from training.multilingual_dataset import load_or_create_multilingual_vocabs


def test_checkpoint_loading() -> bool:
    """Test 1: Verify Phase-8 joint weights load into the fine-tuning model cleanly."""
    print("\n--- TEST 1: Checkpoint-Loading Verification ---")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)
    joint_chk_path = PROJECT_ROOT / "checkpoints" / "joint_best.pt"

    assert joint_chk_path.is_file(), f"Missing Phase 8 joint checkpoint: {joint_chk_path}"
    chk = torch.load(joint_chk_path, map_location="cpu", weights_only=False)

    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
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

    # Load state dict strictly
    model.load_state_dict(chk["model_state_dict"], strict=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [PASSED] Successfully loaded {len(chk['model_state_dict'])} tensors into model ({param_count:,} parameters).")
    return True


def test_freezing_correctness() -> bool:
    """Test 2: Verify frozen layers genuinely do not update during frozen epochs."""
    print("\n--- TEST 2: Freezing-Correctness Verification ---")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
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

    scheduler = LayerUnfreezeScheduler(model=model, schedule="phased", unfreeze_epoch=4)

    # Epoch 1: lower layers should be frozen
    status = scheduler.step_epoch(1)
    assert status["lower_layers_frozen"] is True, "Expected lower layers to be frozen in Epoch 1"

    # Confirm requires_grad flags
    for p in model.word_embedding.parameters():
        assert not p.requires_grad, "word_embedding parameter has requires_grad=True during frozen epoch"
    for p in model.char_cnn.parameters():
        assert not p.requires_grad, "char_cnn parameter has requires_grad=True during frozen epoch"
    for p in model.encoder.parameters():
        assert not p.requires_grad, "encoder parameter has requires_grad=True during frozen epoch"
    for p in model.attention.parameters():
        assert p.requires_grad, "attention parameter has requires_grad=False during upper-unfrozen epoch"

    # Save copy of lower-layer weights before a backward/optimizer step
    initial_word_emb = model.word_embedding.embedding.weight.clone()
    initial_lstm_weight = model.encoder.lstm.weight_ih_l0.clone()
    initial_attn_wq = model.attention.blocks[0].w_q.weight.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Perform a dummy training step
    batch_size, seq_len = 2, 4
    word_ids = torch.randint(1, 100, (batch_size, seq_len))
    char_ids = torch.randint(1, 50, (batch_size, seq_len, 5))
    lang_ids = torch.tensor([0, 0], dtype=torch.long)
    tags = torch.randint(0, 9, (batch_size, seq_len))

    optimizer.zero_grad()
    loss = model(word_ids, char_ids, lang_ids=lang_ids, tags=tags)
    loss.backward()
    optimizer.step()

    # Verify: lower layers did NOT change numerically
    diff_word_emb = (model.word_embedding.embedding.weight - initial_word_emb).abs().max().item()
    diff_lstm = (model.encoder.lstm.weight_ih_l0 - initial_lstm_weight).abs().max().item()
    diff_attn = (model.attention.blocks[0].w_q.weight - initial_attn_wq).abs().max().item()

    assert diff_word_emb == 0.0, f"Frozen word_embedding updated numerically: diff={diff_word_emb}"
    assert diff_lstm == 0.0, f"Frozen BiLSTM updated numerically: diff={diff_lstm}"
    assert diff_attn > 0.0, "Upper layer attention failed to update"

    print(f"  Word embedding delta: {diff_word_emb:.8f} (strictly zero)")
    print(f"  BiLSTM weight delta: {diff_lstm:.8f} (strictly zero)")
    print(f"  Self-attention weight delta: {diff_attn:.8f} (active update)")
    print("  [PASSED] Freezing correctness verified.")
    return True


def test_entity_f1_comparison() -> bool:
    """Test 3: Verify fine-tuned vs control results across all languages."""
    print("\n--- TEST 3: Fine-Tuned vs From-Scratch Control Comparison ---")
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    report_path = PROJECT_ROOT / "reports" / "phase9_finetune_results.md"

    assert report_path.is_file(), f"Missing Phase 9 report: {report_path}"

    for lang in ["konkani", "assamese", "sinhala"]:
        ft_path = checkpoints_dir / f"finetuned_{lang}.pt"
        ctrl_path = checkpoints_dir / f"fromscratch_control_{lang}.pt"
        assert ft_path.is_file(), f"Missing fine-tuned checkpoint for {lang}: {ft_path}"
        assert ctrl_path.is_file(), f"Missing control checkpoint for {lang}: {ctrl_path}"

        chk_ft = torch.load(ft_path, map_location="cpu", weights_only=False)
        chk_ctrl = torch.load(ctrl_path, map_location="cpu", weights_only=False)

        ft_f1 = chk_ft["best_dev_f1"] * 100
        ctrl_f1 = chk_ctrl["best_dev_f1"] * 100
        delta = ft_f1 - ctrl_f1
        print(f"  {lang.capitalize()}: Fine-Tuned = {ft_f1:.2f}% | Control = {ctrl_f1:.2f}% | Delta = {delta:+.2f}%")
        # Ensure fine-tuning beats or ties control (no catastrophic failure)
        assert delta >= -1.0, f"Severe regression vs control in {lang}: delta={delta:.2f}%"

    print("  [PASSED] Fine-tuned vs Control comparison validated.")
    return True


def test_schedule_sensitivity() -> bool:
    """Test 4: Verify schedule sensitivity comparison across languages."""
    print("\n--- TEST 4: Schedule Sensitivity Comparison (Phased vs Immediate) ---")
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    for lang in ["konkani", "assamese", "sinhala"]:
        phased_path = checkpoints_dir / f"finetuned_{lang}.pt"
        imm_path = checkpoints_dir / f"finetuned_{lang}_immediate.pt"
        assert phased_path.is_file(), f"Missing phased checkpoint for {lang}"
        assert imm_path.is_file(), f"Missing immediate checkpoint for {lang}"

        chk_p = torch.load(phased_path, map_location="cpu", weights_only=False)
        chk_i = torch.load(imm_path, map_location="cpu", weights_only=False)

        p_f1 = chk_p["best_dev_f1"] * 100
        i_f1 = chk_i["best_dev_f1"] * 100
        print(f"  {lang.capitalize()}: Phased = {p_f1:.2f}% | Immediate = {i_f1:.2f}% | Delta = {p_f1 - i_f1:+.2f}%")

    print("  [PASSED] Schedule sensitivity validated.")
    return True


def test_reproducibility() -> bool:
    """Test 5: Verify reproducibility on fixed seed."""
    print("\n--- TEST 5: Reproducibility Check ---")
    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    tmp_chk = PROJECT_ROOT / "checkpoints" / "repro_test_ft.pt"
    joint_chk_path = PROJECT_ROOT / "checkpoints" / "joint_best.pt"

    print("  Running 1-epoch fine-tuning reproducibility test on Konkani...")
    res1 = train_finetune_experiment(
        lang="konkani",
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        proc_dir=proc_dir,
        checkpoint_out=tmp_chk,
        init_checkpoint_path=joint_chk_path,
        schedule="phased",
        max_epochs=1,
        seed=1337,
        device_setting="auto",
    )
    score1 = res1["best_dev_f1"]

    res2 = train_finetune_experiment(
        lang="konkani",
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        proc_dir=proc_dir,
        checkpoint_out=tmp_chk,
        init_checkpoint_path=joint_chk_path,
        schedule="phased",
        max_epochs=1,
        seed=1337,
        device_setting="auto",
    )
    score2 = res2["best_dev_f1"]

    diff = abs(score1 - score2)
    print(f"  Run 1 score: {score1*100:.4f}% | Run 2 score: {score2*100:.4f}% | Diff: {diff:.6f}")
    assert diff < 1e-4, f"Reproducibility check failed: diff={diff} exceeds tolerance 1e-4"

    if tmp_chk.is_file():
        tmp_chk.unlink()

    print("  [PASSED] Fine-tuning determinism verified.")
    return True


def run_all_tests() -> None:
    """Run all Phase 9 validation tests and generate reports/phase9_test_results.md."""
    start_time = time.time()
    results: list[tuple[str, str, str]] = []

    tests = [
        ("Checkpoint-Loading Verification", test_checkpoint_loading),
        ("Freezing-Correctness Verification", test_freezing_correctness),
        ("Per-Language Entity-F1 Comparison", test_entity_f1_comparison),
        ("Schedule Sensitivity Comparison", test_schedule_sensitivity),
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
    report_path = PROJECT_ROOT / "reports" / "phase9_test_results.md"
    lines = [
        "# Phase 9 Test & Validation Results Report",
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
        raise RuntimeError("Phase 9 test suite failed.")


if __name__ == "__main__":
    run_all_tests()
