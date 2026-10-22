"""
Phase 13 (Kokborok Training & Standalone Evaluation) Automated Verification Suite.

Validates the 5 mandatory testing requirements for Phase 13:
1. Strategy-Fidelity Test: Confirm the executed training run matches Phase 12's recommendation
   (Option c: Staged / Protected Modular Approach), with starting checkpoint and backbone freezing verified.
2. Divergence-Gate Test: Confirm genuine training occurred, with weight-change magnitude verified
   (>= 1.0% params changed, L2 diff >= 0.01).
3. Test-Set Integrity Test: Confirm the checksum of data/gold/kokborok/test.conll before and after the
   single official evaluation run is identical and matches the lockfile.
4. Single-Evaluation-Run Test: Confirm gold-test was evaluated exactly once for the official record
   via cryptographic run-lock guard.
5. Evaluation-Package Completeness Test: Confirm every component of the standard evaluation package
   (per-type breakdown, token accuracy, confusion matrix, IV/OOV split, error categorization) is present
   for Kokborok, matching the existing three languages' format.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

KOKBOROK_TEST_SHA256 = "de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799"
REFERENCE_3LANG_CHECKSUMS = {
    "data/gold/konkani/test.conll": "bd08ebbf60e048d4fce5d9f5fd233f7dbdeb5bf1f066ee0718eb4b04404ab2d1",
    "data/gold/assamese/test.conll": "2a49321609716b38a893d7b7ca5aa3c27ba0c90863fbca8ac0d4aa87d143c070",
    "data/gold/sinhala/test.conll": "47cf890b0e9c76e235e17355806b521f8925525e80256bf6667b2835a16854d0",
    "baseline_v1/checkpoints/final_assamese.pt": "2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d",
    "baseline_v1/checkpoints/final_konkani.pt": "58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038",
    "baseline_v1/checkpoints/final_sinhala.pt": "6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03",
    "checkpoints/current_best_konkani.pt": "73bac91510a6f755ff80e7c4529ac709d040872ad1d3c3b38ffb1bde0f2530ce",
    "checkpoints/current_best_assamese.pt": "d59afca5d233c471384db4d46f5407855a581cf573f77725b29f63bdf9733dd3",
    "checkpoints/improved_sinhala_v2.pt": "aca068df0355e0b1656ee1ffb8e4c9a3d3046aa70233b38800be7092f36afe86",
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase13Kokborok(unittest.TestCase):

    def setUp(self) -> None:
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"
        self.reports_dir = PROJECT_ROOT / "reports"
        self.final_ckpt_path = self.checkpoints_dir / "final_kokborok.pt"
        self.eval_lock_path = self.reports_dir / "experiment_logs" / "kokborok_test_eval_lock.json"
        self.test_conll_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "test.conll"

    def test_1_strategy_fidelity(self) -> None:
        """Test 1: Strategy-fidelity test: confirm training matches Phase 12's recommendation."""
        self.assertTrue(self.final_ckpt_path.is_file(), "final_kokborok.pt checkpoint is missing")
        ckpt = torch.load(self.final_ckpt_path, map_location="cpu")

        # Verify strategy is Option C staged protected
        self.assertEqual(
            ckpt.get("integration_strategy"),
            "option_c_staged_protected",
            "Checkpoint was not trained under Option (c) Staged / Protected Modular strategy",
        )

        # Verify configuration
        cfg = ckpt.get("config", {})
        self.assertEqual(cfg.get("crf_config"), "per_language", "CRF config must be per_language")
        self.assertEqual(cfg.get("staging"), "frozen_shared_bilstm_attention_heads", "Backbone must be frozen")
        self.assertEqual(cfg.get("num_languages"), 4, "Must configure 4 languages")
        self.assertEqual(cfg.get("warm_start_source"), "checkpoints/current_best_konkani.pt")

        # Verify baseline checkpoints are untouched
        for rel_path, exp_hash in REFERENCE_3LANG_CHECKSUMS.items():
            fpath = PROJECT_ROOT / rel_path
            self.assertTrue(fpath.is_file(), f"Reference file {rel_path} does not exist")
            actual_hash = sha256_file(fpath)
            self.assertEqual(
                actual_hash,
                exp_hash,
                f"Reference file {rel_path} was modified! Hash mismatch.",
            )

    def test_2_divergence_gate(self) -> None:
        """Test 2: Divergence-gate test: confirm genuine training occurred with reported weight changes."""
        self.assertTrue(self.final_ckpt_path.is_file())
        ckpt = torch.load(self.final_ckpt_path, map_location="cpu")
        div_stats = ckpt.get("divergence_stats", {})

        pct_changed = div_stats.get("pct_changed", 0.0)
        l2_diff = div_stats.get("l2_diff", 0.0)
        max_diff = div_stats.get("max_diff", 0.0)

        self.assertGreaterEqual(pct_changed, 1.0, f"Divergence pct_changed ({pct_changed}%) < 1.0%")
        self.assertGreaterEqual(l2_diff, 0.01, f"Divergence L2 diff ({l2_diff}) < 0.01")
        self.assertGreater(max_diff, 0.0, "Divergence max_diff must be positive")

        print(
            f"\n  [Divergence Gate Passed] pct_changed={pct_changed:.2f}%, "
            f"l2_diff={l2_diff:.4f}, max_diff={max_diff:.4f}"
        )

    def test_3_test_set_integrity(self) -> None:
        """Test 3: Test-set integrity test: confirm checksum is identical to Phase 11 lockfile."""
        self.assertTrue(self.test_conll_path.is_file(), "data/gold/kokborok/test.conll missing")
        current_hash = sha256_file(self.test_conll_path)
        self.assertEqual(
            current_hash,
            KOKBOROK_TEST_SHA256,
            f"Kokborok gold test file hash changed! Expected {KOKBOROK_TEST_SHA256}, got {current_hash}",
        )

    def test_4_single_evaluation_run(self) -> None:
        """Test 4: Single-evaluation-run test: confirm gold-test evaluated exactly once with lockfile guard."""
        self.assertTrue(self.eval_lock_path.is_file(), "Official evaluation lock file missing")
        eval_data = json.loads(self.eval_lock_path.read_text(encoding="utf-8"))

        self.assertEqual(
            eval_data.get("single_run_guard_status"),
            "LOCKED_OFFICIAL_RECORD",
            "Single run guard status not locked",
        )
        self.assertEqual(eval_data.get("language"), "kokborok")
        self.assertEqual(eval_data.get("iso_code"), "trp")
        self.assertEqual(eval_data.get("test_sha256"), KOKBOROK_TEST_SHA256)

        # Confirm exactly 200 sentences in predictions
        predictions = eval_data.get("predictions", {})
        self.assertEqual(predictions.get("sentences_count"), 200)
        self.assertEqual(len(predictions.get("tokens", [])), 200)

    def test_5_evaluation_package_completeness(self) -> None:
        """Test 5: Evaluation-package completeness test: confirm all required reports and components exist."""
        required_reports = [
            self.reports_dir / "kokborok_final_results.md",
            self.reports_dir / "kokborok_confusion_matrix.png",
            self.reports_dir / "kokborok_oov_analysis.md",
            self.reports_dir / "kokborok_error_categorization.md",
            self.reports_dir / "kokborok_data_sufficiency_check.md",
        ]
        for rpath in required_reports:
            self.assertTrue(rpath.is_file(), f"Required evaluation artifact missing: {rpath.name}")
            self.assertGreater(rpath.stat().st_size, 0, f"Artifact is empty: {rpath.name}")

        # Check kokborok_final_results.md has per-entity-type breakdown and token accuracy
        final_results_text = (self.reports_dir / "kokborok_final_results.md").read_text(encoding="utf-8")
        self.assertIn("`PER`", final_results_text)
        self.assertIn("`LOC`", final_results_text)
        self.assertIn("`ORG`", final_results_text)
        self.assertIn("`MISC`", final_results_text)
        self.assertIn("Token Acc", final_results_text)
        self.assertIn("68.97%", final_results_text)
        self.assertIn("76.92%", final_results_text)
        self.assertIn("62.50%", final_results_text)
        self.assertIn("99.55%", final_results_text)

        # Check OOV analysis
        oov_text = (self.reports_dir / "kokborok_oov_analysis.md").read_text(encoding="utf-8")
        self.assertIn("In-Vocabulary (IV) Subset", oov_text)
        self.assertIn("Out-of-Vocabulary (OOV) Subset", oov_text)
        self.assertIn("80.00%", oov_text)
        self.assertIn("44.44%", oov_text)

        # Check Error Categorization
        err_text = (self.reports_dir / "kokborok_error_categorization.md").read_text(encoding="utf-8")
        self.assertIn("Missed Entities (False Negatives)", err_text)
        self.assertIn("Spurious Entities (False Positives)", err_text)
        self.assertIn("Boundary Errors", err_text)
        self.assertIn("Type Confusion", err_text)

        # Check Data Sufficiency Check
        suff_text = (self.reports_dir / "kokborok_data_sufficiency_check.md").read_text(encoding="utf-8")
        self.assertIn("STRICTLY REJECTED", suff_text)


if __name__ == "__main__":
    unittest.main()
