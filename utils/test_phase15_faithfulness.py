"""
Phase 15 (Unified Explainability & Faithfulness Re-Validation) Automated Verification Suite.

Validates the 5 mandatory testing requirements for Phase 15:
1. Checkpoint-Identity Test: Confirms the exact four checkpoints evaluated match Phase 14's
   final-system determination (current_best_konkani, current_best_assamese, improved_sinhala_v2, final_kokborok).
2. Symmetric-Design Verification Test: Confirms, from the code of explainability/erasure_faithfulness_test_v2.py,
   that both top-k and random-k candidate pools strictly exclude the target entity query token.
3. Sample-Size Test: Confirms >=100 predictions per language for Konkani, Assamese, Sinhala,
   and verifies an honest explicit shortfall report for Kokborok (evaluating all 17 available instances).
4. Winning-Method-Determination Test: Confirms the method selection for each language is independently
   derived from that language's own faithfulness gap numbers (higher/less-negative gap wins).
5. Cycle-2 Comparison Test: Confirms Konkani and Assamese's new faithfulness numbers are explicitly
   compared against their pre-Cycle-2 numbers, with changes reported.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase15Faithfulness(unittest.TestCase):

    def setUp(self) -> None:
        self.reports_dir = PROJECT_ROOT / "reports"
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"
        self.raw_json_path = self.reports_dir / "experiment_logs" / "phase15_faithfulness_four_language_raw.json"
        self.master_report_path = self.reports_dir / "phase15_faithfulness_results_four_language.md"
        self.cycle2_report_path = self.reports_dir / "phase15_cycle2_faithfulness_comparison.md"
        self.head_report_path = self.reports_dir / "phase15_head_wise_analysis_four_language.md"
        self.gallery_dir = self.reports_dir / "phase15_heatmap_gallery"
        self.erasure_v2_code_path = PROJECT_ROOT / "explainability" / "erasure_faithfulness_test_v2.py"

    def test_1_checkpoint_identity(self) -> None:
        """Test 1: Confirm the exact four checkpoints match Phase 14's final system determination."""
        expected_checkpoints = {
            "konkani": "current_best_konkani.pt",
            "assamese": "current_best_assamese.pt",
            "sinhala": "improved_sinhala_v2.pt",
            "kokborok": "final_kokborok.pt",
        }
        for lang, ckpt_name in expected_checkpoints.items():
            ckpt_path = self.checkpoints_dir / ckpt_name
            self.assertTrue(ckpt_path.is_file(), f"Missing checkpoint for {lang}: {ckpt_name}")

        master_text = self.master_report_path.read_text(encoding="utf-8")
        for ckpt_name in expected_checkpoints.values():
            self.assertIn(ckpt_name, master_text, f"Checkpoint {ckpt_name} not cited in master report")

    def test_2_symmetric_design_verification(self) -> None:
        """Test 2: Confirm, from code, that both top-k and random-k pools exclude target query token."""
        self.assertTrue(self.erasure_v2_code_path.is_file())
        code = self.erasure_v2_code_path.read_text(encoding="utf-8")

        # Verify candidate pool exclusion: valid_candidates = [i for i in range(seq_len) if i != q_idx]
        self.assertIn("valid_candidates = [i for i in range(seq_len) if i != q_idx]", code)

        # Verify top-k drawn strictly from valid_candidates
        self.assertIn("grad_ranked_indices = sorted(valid_candidates", code)
        self.assertIn("raw_ranked_indices = sorted(valid_candidates", code)

        # Verify random-k drawn from valid candidates non-top pool
        self.assertIn("non_top_pool = [i for i in valid_candidates", code)

    def test_3_sample_size_and_shortfall(self) -> None:
        """Test 3: Confirm >=100 predictions per language, or explicit documented shortfall for Kokborok."""
        self.assertTrue(self.raw_json_path.is_file())
        raw_data = json.loads(self.raw_json_path.read_text(encoding="utf-8"))
        f_data = raw_data["faithfulness"]

        for lang in ["konkani", "assamese", "sinhala"]:
            n_eval = f_data[lang]["sample_size"]
            self.assertGreaterEqual(n_eval, 100, f"{lang} evaluated {n_eval} < 100 samples")

        # Kokborok explicit shortfall check
        kb_eval = f_data["kokborok"]["sample_size"]
        kb_avail = f_data["kokborok"]["total_dev_entities_available"]
        self.assertEqual(kb_eval, kb_avail, "Kokborok must evaluate 100% of available dev entities")
        self.assertEqual(kb_eval, 17, "Kokborok dev entities must equal 17")
        self.assertGreater(f_data["kokborok"]["sample_shortfall"], 0, "Shortfall must be recorded")

        # Verify report documents the shortfall
        master_text = self.master_report_path.read_text(encoding="utf-8")
        self.assertIn("Documented Sample Size Limitation for Kokborok", master_text)
        self.assertIn("17", master_text)

    def test_4_winning_method_determination(self) -> None:
        """Test 4: Confirm winning method for each language is independently derived from its numbers."""
        self.assertTrue(self.raw_json_path.is_file())
        raw_data = json.loads(self.raw_json_path.read_text(encoding="utf-8"))
        f_data = raw_data["faithfulness"]

        for lang in ["konkani", "assamese", "sinhala", "kokborok"]:
            d = f_data[lang]
            raw_gap = sum(d["raw_attention"][f"k={k}"]["faithfulness_gap"] for k in [1, 2, 3]) / 3.0
            grad_gap = sum(d["grad_x_attn"][f"k={k}"]["faithfulness_gap"] for k in [1, 2, 3]) / 3.0

            expected_winner = "Raw Attention" if raw_gap >= grad_gap else "Gradient x Attention"
            self.assertEqual(d["winning_method"], expected_winner, f"Method mismatch for {lang}")

    def test_5_cycle2_comparison(self) -> None:
        """Test 5: Confirm Konkani/Assamese new numbers are compared against pre-Cycle-2 numbers."""
        self.assertTrue(self.cycle2_report_path.is_file())
        c2_text = self.cycle2_report_path.read_text(encoding="utf-8")

        self.assertIn("Pre-Cycle-2 vs. Post-Cycle-2", c2_text)
        self.assertIn("Konkani Faithfulness Comparison", c2_text)
        self.assertIn("Assamese Faithfulness Comparison", c2_text)
        self.assertIn("Gap Delta", c2_text)
        self.assertIn("Win Rate Delta", c2_text)

        # Check that gallery cards exist for all 4 languages
        for lang in ["konkani", "assamese", "sinhala", "kokborok"]:
            html_cards = list(self.gallery_dir.glob(f"{lang}_*.html"))
            self.assertGreaterEqual(len(html_cards), 3, f"Expected >= 3 heatmap cards for {lang}, found {len(html_cards)}")


if __name__ == "__main__":
    unittest.main()
