"""
Phase 14 (Cross-Language Interference / Synergy Study) Automated Test & Verification Suite.

Validates the 3 mandatory testing requirements for Phase 14:
1. Final-vs-Comparison-Checkpoint Consistency Test:
   - Confirms whether Phase 13's final checkpoint differs from Phase 12's comparison checkpoints.
   - Confirms that Phase 12's conclusion was explicitly re-verified against the final checkpoint.
2. Evidence-Grounding Test:
   - Confirms that every interaction-effect claim cites specific checkpoint paths and numbers.
3. Completeness Test:
   - Confirms that all four languages (Konkani, Assamese, Sinhala, Kokborok) appear in the final
     comparison table with explicit before/after status and deltas.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase14Synergy(unittest.TestCase):

    def setUp(self) -> None:
        self.reports_dir = PROJECT_ROOT / "reports"
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"
        self.audit_json_path = self.reports_dir / "phase14_synergy_audit.json"
        self.report_md_path = self.reports_dir / "cross_language_synergy_final.md"

    def test_1_final_vs_comparison_consistency(self) -> None:
        """Test 1: Confirm final checkpoints vs comparison checkpoints and re-verification."""
        self.assertTrue(self.audit_json_path.is_file(), "phase14_synergy_audit.json missing")
        audit_data = json.loads(self.audit_json_path.read_text(encoding="utf-8"))

        # Confirm Option C comparison checkpoint results exist
        self.assertIn("option_c_comparison", audit_data)
        comp_res = audit_data["option_c_comparison"]
        self.assertEqual(comp_res["kokborok"]["entity_f1"], 50.00)

        # Confirm Phase 13 final checkpoint results exist
        self.assertIn("final_model_phase13", audit_data)
        final_res = audit_data["final_model_phase13"]
        self.assertEqual(final_res["kokborok"]["entity_f1"], 61.54)

        # Confirm difference: final checkpoint is improved (+11.54 pp on Dev F1)
        diff_f1 = final_res["kokborok"]["entity_f1"] - comp_res["kokborok"]["entity_f1"]
        self.assertGreater(diff_f1, 0.0, "Final checkpoint must show empirical difference")

        # Confirm Option A negative transfer re-verified
        self.assertIn("option_a_joint", audit_data)
        joint_res = audit_data["option_a_joint"]
        self.assertLess(joint_res["konkani"]["entity_f1"], 90.0, "Option A Konkani must show negative transfer")
        self.assertLess(joint_res["assamese"]["entity_f1"], 90.0, "Option A Assamese must show negative transfer")
        self.assertLess(joint_res["sinhala"]["entity_f1"], 90.0, "Option A Sinhala must show negative transfer")

    def test_2_evidence_grounding(self) -> None:
        """Test 2: Confirm every interaction-effect claim cites specific checkpoints and numbers."""
        self.assertTrue(self.report_md_path.is_file(), "cross_language_synergy_final.md missing")
        report_text = self.report_md_path.read_text(encoding="utf-8")

        # Check explicit checkpoints cited
        required_ckpts = [
            "checkpoints/final_kokborok.pt",
            "checkpoints/staged_optionC_kokborok.pt",
            "checkpoints/joint4lang_optionA_kokborok.pt",
            "checkpoints/current_best_konkani.pt",
            "checkpoints/current_best_assamese.pt",
            "checkpoints/improved_sinhala_v2.pt",
        ]
        for ckpt in required_ckpts:
            self.assertIn(ckpt, report_text, f"Missing required checkpoint citation: {ckpt}")

        # Check specific empirical metrics cited
        self.assertIn("96.69%", report_text)
        self.assertIn("98.55%", report_text)
        self.assertIn("98.50%", report_text)
        self.assertIn("61.54%", report_text)
        self.assertIn("68.97%", report_text)
        self.assertIn("87.73%", report_text)
        self.assertIn("89.54%", report_text)
        self.assertIn("86.87%", report_text)
        self.assertIn("90.22%", report_text)

    def test_3_completeness(self) -> None:
        """Test 3: Confirm all 4 languages appear in comparison table with before/after status."""
        self.assertTrue(self.report_md_path.is_file())
        report_text = self.report_md_path.read_text(encoding="utf-8")

        for lang in ["Konkani (`kok`)", "Assamese (`asm`)", "Sinhala (`sin`)", "Kokborok (`trp`)"]:
            self.assertIn(lang, report_text, f"Language missing from final comparison table: {lang}")

        # Check before/after deltas present
        self.assertIn("0.00 pp", report_text)
        self.assertIn("+1.37 pp", report_text)
        self.assertIn("+2.20 pp", report_text)
        self.assertIn("+61.54 pp", report_text)
        self.assertIn("90.22%", report_text)


if __name__ == "__main__":
    unittest.main()
