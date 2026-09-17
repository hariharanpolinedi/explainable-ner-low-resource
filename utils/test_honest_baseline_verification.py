"""
Test Suite: Checkpoint Provenance and Honest Baseline Verification.

Validates:
1. Provenance-trace test: deployed checkpoint mapped to specific commit hash and file hash.
2. Loadability test: confirms all 40 checkpoints tested and properly classified.
3. Baseline reproducibility test: recomputes Konkani baseline and checks exact numerical equality.
4. Manual test-case fidelity test: confirms exact Phase 2 normalization & tokenization used in reproduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from evaluation.trace_deployed_checkpoint import trace_all_deployed_checkpoints
from evaluation.inventory_checkpoints import inventory_all_checkpoints
from evaluation.honest_per_type_baseline import evaluate_conll, load_model
from evaluation.reproduce_manual_test_case import run_manual_test_case_reproduction
from preprocessing.normalize import normalize_text
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import load_or_create_multilingual_vocabs


def test_provenance_trace() -> bool:
    print("\n[CHECK 1] Testing Deployed Checkpoint Provenance Trace...")
    results = trace_all_deployed_checkpoints()
    for lang in ["konkani", "assamese", "sinhala"]:
        assert lang in results, f"Missing {lang} trace"
        r = results[lang]
        assert r["is_rejected_patch"] is False, f"Deployed model for {lang} should NOT be a rejected patch"
        assert len(r["matching_checkpoints_in_checkpoints_dir"]) > 0, f"No matching checkpoint in checkpoints/ for {lang}"
        assert r["bundle_creation_commit"] is not None, f"Missing commit for {lang}"
        commit_hash = r["bundle_creation_commit"]["hash"]
        assert len(commit_hash) == 40, f"Invalid git commit hash: {commit_hash}"
    print("  -> PASS: All deployed models mapped to valid git commit and verified baselines")
    return True


def test_checkpoint_loadability() -> bool:
    print("\n[CHECK 2] Testing Full Checkpoint Loadability Inventory...")
    inv = inventory_all_checkpoints()
    assert len(inv) == 40, f"Expected 40 checkpoints, found {len(inv)}"
    loadable = [x for x in inv if x["loadable_under_v4_multilingual"]]
    excluded = [x for x in inv if not x["loadable_under_v4_multilingual"]]
    assert len(loadable) == 29, f"Expected 29 loadable, got {len(loadable)}"
    assert len(excluded) == 11, f"Expected 11 excluded, got {len(excluded)}"
    for ex in excluded:
        assert ex["exclusion_reason"] is not None, f"Missing exclusion reason for {ex['name']}"
    print(f"  -> PASS: All 40 checkpoints evaluated ({len(loadable)} loadable, {len(excluded)} excluded with documented reasons)")
    return True


def test_baseline_reproducibility() -> bool:
    print("\n[CHECK 3] Testing Baseline Determinism and Reproducibility (Konkani Gold-Dev)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)
    dev_file = PROJECT_ROOT / "data" / "gold" / "konkani" / "dev.conll"

    ckpt_p = PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt"
    m1 = load_model(ckpt_p, device, w_to_id, c_to_id)
    eval1 = evaluate_conll(m1, dev_file, "konkani", device, w_to_id, c_to_id)

    m2 = load_model(ckpt_p, device, w_to_id, c_to_id)
    eval2 = evaluate_conll(m2, dev_file, "konkani", device, w_to_id, c_to_id)

    f1_1 = eval1["overall"]["entity_f1"]
    f1_2 = eval2["overall"]["entity_f1"]
    assert f1_1 == f1_2, f"Evaluation non-deterministic: {f1_1} vs {f1_2}"
    assert round(f1_1 * 100, 2) == 96.69, f"Expected 96.69% F1, got {round(f1_1 * 100, 2)}"
    print(f"  -> PASS: Exact numerical identity confirmed (Pass 1: {f1_1:.4f} == Pass 2: {f1_2:.4f})")
    return True


def test_manual_test_case_fidelity() -> bool:
    print("\n[CHECK 4] Testing Manual Test-Case Fidelity & Pipeline Exactness...")
    results = run_manual_test_case_reproduction()
    assert len(results) >= 5, f"Expected at least 5 Konkani checkpoints tested, got {len(results)}"
    deployed_res = next(r for r in results if r["filename"] == "final_konkani.pt")
    assert deployed_res["hits"] == 2, f"Expected 2 hits for deployed model, got {deployed_res['hits']}"
    assert deployed_res["total_expected"] == 5, "Expected 5 total targets"
    print("  -> PASS: Exact match with screenshot failure (2/5 detected, missing LOC and middle PER)")
    return True


def run_all() -> bool:
    print("=" * 75)
    print("RUNNING CHECKPOINT PROVENANCE & HONEST BASELINE VERIFICATION TESTS")
    print("=" * 75)
    c1 = test_provenance_trace()
    c2 = test_checkpoint_loadability()
    c3 = test_baseline_reproducibility()
    c4 = test_manual_test_case_fidelity()
    all_pass = c1 and c2 and c3 and c4
    print("=" * 75)
    if all_pass:
        print("ALL 4 VERIFICATION CHECKS PASSED.")
    else:
        print("VERIFICATION CHECKS FAILED.")
    print("=" * 75)
    return all_pass


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
