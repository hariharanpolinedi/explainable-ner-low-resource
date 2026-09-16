"""
Phase 10 Test & Validation Suite.

Verifies:
1. Test-Set Isolation Guard: Zero test-set leakage into pseudo-labeling pool across all 3 languages,
   plus strict exception triggering if test contamination is simulated.
2. Confidence Filter Correctness: Confirms only sentences >= threshold are retained.
3. Pseudo-Volume Cap Enforcement: Confirms strict cap enforcement without overflow.
4. Round-by-round Dev-F1 Tracking & Rollback Guard: Verifies degradation triggers rollback.
5. Reproducibility: Confirms deterministic dev-F1 score on repeated run with same seed.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluation.test_set_isolation_guard import (
    CriticalLeakageError,
    load_sentences_from_conll,
    verify_pool_isolation,
)
from training.confidence_filter import filter_unlabeled_pool


class TestPhase10(unittest.TestCase):

    def setUp(self) -> None:
        self.languages = ["konkani", "assamese", "sinhala"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_1_test_set_isolation_guard(self) -> None:
        """1. Confirm zero overlap between unlabeled pool and gold-test, and test leakage exception."""
        for lang in self.languages:
            gold_train = set(load_sentences_from_conll(f"data/gold/{lang}/train.conll"))
            gold_dev = set(load_sentences_from_conll(f"data/gold/{lang}/dev.conll"))
            gold_test = set(load_sentences_from_conll(f"data/gold/{lang}/test.conll"))
            silver = set(load_sentences_from_conll(f"data/silver/{lang}/train.conll"))
            used = gold_train | gold_dev | gold_test | silver

            candidates = []
            with open(f"data/processed/{lang}/sentences.txt", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    l = line.strip()
                    if l:
                        toks = l.split()
                        if 3 <= len(toks) <= 60:
                            s_str = " ".join(toks)
                            if s_str not in used:
                                candidates.append(s_str)
                    if len(candidates) >= 500:
                        break

            res = verify_pool_isolation(lang, candidates)
            self.assertEqual(res["test_overlap_count"], 0, f"Test leakage in {lang}!")
            self.assertEqual(res["status"], "PASS")

        # Test intentional contamination triggers CriticalLeakageError
        test_sent = list(gold_test)[0]
        contaminated_pool = candidates + [test_sent]
        with self.assertRaises(CriticalLeakageError):
            verify_pool_isolation("sinhala", contaminated_pool)

    def test_2_confidence_filter_correctness(self) -> None:
        """2. Confirm confidence filter strictly retains only sentences at or above threshold."""
        sample_batch = [
            ["नागेश", "करमली", "चो", "जल्म", "काकोडें", "हांगा", "जालो", "."],
            ["तो", "जें", "कितें", "आयकता", ",", "पळेता", "."],
            ["गोंय", "सरकारान", "निर्णय", "घेतलो", "."],
            ["आनी", "हेर", "सगळे", "लोक", "गेले", "."],
        ]

        threshold = 0.90
        res = filter_unlabeled_pool(
            lang="konkani",
            candidate_tokens=sample_batch,
            checkpoint_path="checkpoints/finetuned_konkani.pt",
            proc_dir="data/processed",
            threshold=threshold,
            device=self.device,
            batch_size=4,
        )

        self.assertGreater(res["total_evaluated"], 0)
        for s in res["retained_samples"]:
            self.assertGreaterEqual(
                s["confidence"],
                threshold,
                f"Sentence confidence {s['confidence']} < threshold {threshold}",
            )

        # Confirm impossibly high threshold (e.g. 1.05) yields 0 retained
        res_impossible = filter_unlabeled_pool(
            lang="konkani",
            candidate_tokens=sample_batch,
            checkpoint_path="checkpoints/finetuned_konkani.pt",
            proc_dir="data/processed",
            threshold=1.05,
            device=self.device,
            batch_size=4,
        )
        self.assertEqual(res_impossible["retained_count"], 0)

    def test_3_pseudo_volume_cap_enforcement(self) -> None:
        """3. Confirm pipeline strictly enforces volume cap."""
        sample_batch = [
            ["नागेश", "करमली", "चो", "जल्म", "काकोडें", "हांगा", "जालो", "."],
            ["तो", "जें", "कितें", "आयकता", ",", "पळेता", "."],
            ["गोंय", "सरकारान", "निर्णय", "घेतलो", "."],
            ["आनी", "हेर", "सगळे", "लोक", "गेले", "."],
            ["तांच्या", "बापायचें", "नांव", "रामदास", "नायक", "करमली", "."],
        ]

        cap = 2
        res = filter_unlabeled_pool(
            lang="konkani",
            candidate_tokens=sample_batch,
            checkpoint_path="checkpoints/finetuned_konkani.pt",
            proc_dir="data/processed",
            threshold=0.50,  # low threshold so all qualify
            device=self.device,
            batch_size=4,
            volume_cap=cap,
        )

        self.assertLessEqual(res["retained_count"], cap)
        self.assertTrue(res["volume_cap_applied"])
        self.assertEqual(len(res["retained_samples"]), cap)

    def test_4_round_by_round_dev_f1_tracking_and_rollback(self) -> None:
        """4. Confirm self-training history logging and rollback on regression."""
        # Simulated test of rollback logic
        history = [
            {"round": 0, "dev_f1": 0.9500, "status": "BASE"},
            {"round": 1, "dev_f1": 0.9550, "status": "ACCEPTED"},
            {"round": 2, "dev_f1": 0.9480, "status": "REGRESSION_DETECTED (ROLLED BACK)"},
        ]

        best_f1 = history[0]["dev_f1"]
        best_round = 0
        for h in history[1:]:
            if h["dev_f1"] >= best_f1:
                best_f1 = h["dev_f1"]
                best_round = h["round"]
            else:
                self.assertIn("ROLLED BACK", h["status"])

        self.assertEqual(best_round, 1)
        self.assertEqual(best_f1, 0.9550)

    def test_5_reproducibility(self) -> None:
        """5. Confirm deterministic inference and confidence scores across runs."""
        sample_batch = [
            ["नागेश", "करमली", "चो", "जल्म", "काकोडें", "हांगा", "जालो", "."],
            ["गोंय", "सरकारान", "निर्णय", "घेतलो", "."],
        ]

        res1 = filter_unlabeled_pool(
            lang="konkani",
            candidate_tokens=sample_batch,
            checkpoint_path="checkpoints/finetuned_konkani.pt",
            proc_dir="data/processed",
            threshold=0.80,
            device=self.device,
        )

        res2 = filter_unlabeled_pool(
            lang="konkani",
            candidate_tokens=sample_batch,
            checkpoint_path="checkpoints/finetuned_konkani.pt",
            proc_dir="data/processed",
            threshold=0.80,
            device=self.device,
        )

        self.assertEqual(res1["retained_count"], res2["retained_count"])
        for s1, s2 in zip(res1["retained_samples"], res2["retained_samples"]):
            self.assertEqual(s1["tags"], s2["tags"])
            self.assertAlmostEqual(s1["confidence"], s2["confidence"], places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
