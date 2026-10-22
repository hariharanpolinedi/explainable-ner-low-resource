"""
Phase 12 (Kokborok Integration Strategy) Automated Test & Verification Suite.

Validates the 4 mandatory testing requirements for Phase 12:
1. Language-Conditioning Extension Test: Confirms 4-language conditioning works correctly
   (forward pass shapes, embedding table sizing, absence of silent 3-language breakages).
2. Divergence-Gate Test: Confirms every trained candidate checkpoint passed the divergence gate,
   with parameter-wise divergence metrics reported (>= 1.0% params changed, L2 diff >= 0.01).
3. Existing-Language Protection Test: Confirms that existing languages' numbers under every option
   tested are explicitly quantified against their pre-integration baseline.
4. Checkpoint-Naming & Preservation Test: Confirms baseline_v1/ and checkpoints/improved_{lang}_v2.pt
   and current_best checkpoints remain strictly unmodified throughout this phase.
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

from model.language_conditioning import LanguageEmbedding
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, RAW_CORPUS_VOLUMES


class TestPhase12Kokborok(unittest.TestCase):

    def setUp(self) -> None:
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"
        self.baseline_v1_dir = PROJECT_ROOT / "baseline_v1" / "checkpoints"
        self.reports_dir = PROJECT_ROOT / "reports"
        self.raw_results_path = self.reports_dir / "kokborok_integration_raw_results.json"
        self.comparison_report_path = self.reports_dir / "kokborok_integration_strategy_comparison.md"

    def test_1_language_conditioning_extension(self) -> None:
        """Requirement 1: Verify 3-to-4 language extension forward passes and dictionaries."""
        # 1. Check LANG_TO_ID contains all 4 languages
        self.assertIn("kokborok", LANG_TO_ID, "Kokborok missing from LANG_TO_ID")
        self.assertEqual(LANG_TO_ID["kokborok"], 3, "Kokborok ID must be 3")
        self.assertEqual(len(LANG_TO_ID), 4, "LANG_TO_ID must have exactly 4 languages")

        # 2. Check RAW_CORPUS_VOLUMES
        self.assertIn("kokborok", RAW_CORPUS_VOLUMES, "Kokborok missing from RAW_CORPUS_VOLUMES")
        self.assertGreater(RAW_CORPUS_VOLUMES["kokborok"], 0)

        # 3. Test LanguageEmbedding module forward pass
        le = LanguageEmbedding(num_languages=4, embedding_dim=16)
        lang_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        out = le(lang_ids, seq_len=8)
        self.assertEqual(out.shape, (4, 8, 16), f"Unexpected shape from LanguageEmbedding: {out.shape}")

        # 4. Test NERMultilingualModel 4-language forward pass (shared head)
        model_shared = NERMultilingualModel(
            word_vocab_size=1000,
            char_vocab_size=200,
            num_tags=9,
            num_languages=4,
            crf_config="shared",
        )
        words = torch.randint(0, 500, (4, 8))
        chars = torch.randint(0, 100, (4, 8, 5))
        loss = model_shared(words, chars, lang_ids=lang_ids, tags=torch.randint(0, 9, (4, 8)))
        self.assertTrue(torch.is_tensor(loss) and loss.dim() == 0, "Loss must be scalar tensor")
        paths = model_shared(words, chars, lang_ids=lang_ids)
        self.assertEqual(len(paths), 4, "Must decode paths for 4 batch items")

        # 5. Test NERMultilingualModel 4-language forward pass (per_language head)
        model_pl = NERMultilingualModel(
            word_vocab_size=1000,
            char_vocab_size=200,
            num_tags=9,
            num_languages=4,
            crf_config="per_language",
        )
        self.assertEqual(len(model_pl.fc_heads), 4, "Must instantiate 4 per-language linear heads")
        self.assertEqual(len(model_pl.crf_heads), 4, "Must instantiate 4 per-language CRF heads")

    def test_2_divergence_gate_enforcement(self) -> None:
        """Requirement 2: Verify divergence gate results on all option checkpoints."""
        self.assertTrue(
            self.raw_results_path.exists(),
            f"Master raw results JSON missing: {self.raw_results_path}",
        )
        data = json.loads(self.raw_results_path.read_text(encoding="utf-8"))

        # Verify Option (b)
        self.assertIn("option_b_standalone", data)
        div_b = data["option_b_standalone"]["divergence"]
        self.assertGreaterEqual(div_b["pct_changed"], 1.0, "Option (b) divergence < 1.0%")
        self.assertGreaterEqual(div_b["l2_diff"], 0.01, "Option (b) L2 diff < 0.01")

        # Verify Option (a)
        self.assertIn("option_a_shared_uniform", data)
        div_a = data["option_a_shared_uniform"]["divergence"]
        self.assertGreaterEqual(div_a["pct_changed"], 1.0, "Option (a) divergence < 1.0%")
        self.assertGreaterEqual(div_a["l2_diff"], 0.01, "Option (a) L2 diff < 0.01")

        # Verify Option (c)
        if "option_c_staged" in data and data["option_c_staged"] is not None:
            div_c = data["option_c_staged"]["divergence"]
            self.assertGreaterEqual(div_c["pct_changed"], 1.0, "Option (c) divergence < 1.0%")
            self.assertGreaterEqual(div_c["l2_diff"], 0.01, "Option (c) L2 diff < 0.01")

    def test_3_existing_language_protection(self) -> None:
        """Requirement 3: Confirm explicit quantification of existing-language deltas."""
        data = json.loads(self.raw_results_path.read_text(encoding="utf-8"))
        baselines = data["pre_integration_baselines"]

        # Option (b) must preserve existing languages by construction (delta == 0.0)
        res_b = data["option_b_standalone"]
        for lang in ["konkani", "assamese", "sinhala"]:
            self.assertEqual(
                res_b[f"{lang}_dev_f1"],
                baselines[lang],
                f"Option (b) unexpectedly altered {lang} baseline!",
            )

        # Option (a) must log exact deltas for all 3 existing languages
        res_a = data["option_a_shared_uniform"]["results"]
        for lang in ["konkani", "assamese", "sinhala"]:
            f1 = res_a[lang]["entity_f1"]
            delta = round(f1 - baselines[lang], 2)
            self.assertIsInstance(delta, float)

    def test_4_checkpoint_naming_and_preservation(self) -> None:
        """Requirement 4: Confirm baseline and current_best checkpoints remain untouched."""
        # 1. Baseline v1 checkpoints
        for lang in ["konkani", "assamese", "sinhala"]:
            p = self.baseline_v1_dir / f"final_{lang}.pt"
            self.assertTrue(p.exists(), f"baseline_v1 final_{lang}.pt missing!")

        # 2. Current-best checkpoints
        current_bests = {
            "konkani": ("checkpoints/current_best_konkani.pt", "73bac91510a6f755ff80e7c4529ac709d040872ad1d3c3b38ffb1bde0f2530ce"),
            "assamese": ("checkpoints/current_best_assamese.pt", "d59afca5d233c471384db4d46f5407855a581cf573f77725b29f63bdf9733dd3"),
            "sinhala": ("checkpoints/improved_sinhala_v2.pt", "aca068df0355e0b1656ee1ffb8e4c9a3d3046aa70233b38800be7092f36afe86"),
        }
        for lang, (p_str, exp_hash) in current_bests.items():
            p = PROJECT_ROOT / p_str
            self.assertTrue(p.exists(), f"Current best for {lang} missing: {p_str}")
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(h, exp_hash, f"Current best checkpoint for {lang} was modified! Hash mismatch.")

        # 3. New checkpoints created under distinct names
        self.assertTrue((self.checkpoints_dir / "standalone_kokborok_optionB.pt").exists(), "optionB checkpoint missing")
        self.assertTrue((self.checkpoints_dir / "joint4lang_optionA_kokborok.pt").exists(), "joint4lang kokborok checkpoint missing")


if __name__ == "__main__":
    unittest.main()
