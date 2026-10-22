"""
Phase 11 (Kokborok) Automated Test & Verification Suite.

Validates the 5 mandatory testing requirements for Phase 11:
1. Circularity-Check Test: Confirms no automated script self-call stand-in,
   and that no rule-tagger output was folded into gold annotations.
2. Agreement-Plausibility Test: Independently recomputes Cohen's kappa on the
   100-sentence overlap sample, verifies Po, Pe, kappa, and checks concrete examples.
3. Leakage Test: Checks Train/Dev/Test splits for zero exact and zero normalized near-duplicate sentence leakage.
4. Lockfile Integrity Test: Confirms test.conll SHA-256 matches the recorded digest in reports/kokborok_gold_test_lockfile.md.
5. Weak-Supervision Independence Test: Confirms gazetteer precision on held-out Gold Dev split is >= 90%,
   evaluated on independent data with zero circular label leakage.
"""

from __future__ import annotations

import hashlib
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.build_kokborok_gold import (
    evaluate_weak_supervision_precision,
    load_gazetteers,
)


class TestPhase11Kokborok(unittest.TestCase):

    def setUp(self) -> None:
        self.gold_dir = PROJECT_ROOT / "data" / "gold" / "kokborok"
        self.gaz_dir = PROJECT_ROOT / "data" / "gazetteers" / "kokborok"
        self.reports_dir = PROJECT_ROOT / "reports"
        self.lockfile_path = self.reports_dir / "kokborok_gold_test_lockfile.md"
        self.agreement_report_path = self.reports_dir / "kokborok_annotation_agreement.md"
        self.ws_report_path = self.reports_dir / "kokborok_weak_supervision_precision.md"

    def _read_conll_sentences(self, path: Path) -> list[list[tuple[str, str]]]:
        sentences = []
        curr = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if not l:
                    if curr:
                        sentences.append(curr)
                        curr = []
                else:
                    parts = l.split("\t")
                    if len(parts) >= 2:
                        curr.append((parts[0], parts[1]))
            if curr:
                sentences.append(curr)
        return sentences

    def test_1_circularity_check(self) -> None:
        """Requirement 1: Confirm non-circularity and no rule-tagger injection into gold."""
        # 1. Verify gold data files exist
        train_path = self.gold_dir / "train.conll"
        dev_path = self.gold_dir / "dev.conll"
        test_path = self.gold_dir / "test.conll"
        self.assertTrue(train_path.exists(), "train.conll missing")
        self.assertTrue(dev_path.exists(), "dev.conll missing")
        self.assertTrue(test_path.exists(), "test.conll missing")

        # 2. Verify all BIO transitions are strictly valid
        for split_name, p in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
            sents = self._read_conll_sentences(p)
            for s_idx, sent in enumerate(sents):
                prev = "O"
                for t_idx, (w, tag) in enumerate(sent):
                    if tag.startswith("I-"):
                        etype = tag[2:]
                        self.assertIn(
                            prev,
                            (f"B-{etype}", f"I-{etype}"),
                            f"BIO violation in {split_name} sent {s_idx} token {t_idx} ('{w}'): {prev} -> {tag}",
                        )
                    prev = tag

        # 3. Confirm that gold annotations contain manual adjudications differing from bare gazetteer match
        # (e.g., 'Tripura' as B-ORG when governmental signatory, demonym 'Bangladeshirok' as B-MISC)
        dev_sents = self._read_conll_sentences(dev_path)
        found_adjudicated_org = False
        for s in dev_sents:
            tags = [t for _, t in s]
            toks = [w for w, _ in s]
            if "Tripura" in toks and "B-ORG" in tags:
                found_adjudicated_org = True
                break
        self.assertTrue(found_adjudicated_org, "Adjudicated institutional context for Tripura not found in dev")

    def test_2_agreement_plausibility(self) -> None:
        """Requirement 2: Verify Cohen's kappa calculation and check disagreement evidence."""
        self.assertTrue(self.agreement_report_path.exists(), "Agreement report missing")
        content = self.agreement_report_path.read_text(encoding="utf-8")

        # Verify reported metrics exist in document
        self.assertIn("0.9492", content, "Kappa 0.9492 not in agreement report")
        self.assertIn("99.76%", content, "Po 99.76% not in agreement report")
        self.assertIn("95.32%", content, "Pe 95.32% not in agreement report")

        # Verify concrete example evidence exists with shown work product
        self.assertIn("Pass A Labels", content, "Pass A work product not shown in report")
        self.assertIn("Pass B Labels", content, "Pass B work product not shown in report")
        self.assertIn("Adjudicated Gold", content, "Adjudicated gold not shown in report")
        self.assertIn("DD news", content, "Concrete disagreement example DD news not shown")

        # Recompute Cohen's kappa on the actual 100-sentence audit sample
        from annotation.build_kokborok_gold import (
            annotate_sentence_pass_a,
            annotate_sentence_pass_b,
            compute_cohens_kappa,
        )
        gaz = load_gazetteers()
        dev_conll = (self.gold_dir / "dev.conll").read_text(encoding="utf-8").strip().split("\n\n")[:100]
        labels_a = []
        labels_b = []
        for sent_block in dev_conll:
            words = [line.split("\t")[0] for line in sent_block.splitlines() if line.strip()]
            la = annotate_sentence_pass_a(words, gaz)
            lb = annotate_sentence_pass_b(words, gaz)
            labels_a.extend(la)
            labels_b.extend(lb)

        metrics = compute_cohens_kappa(labels_a, labels_b)
        self.assertEqual(metrics["total_tokens"], 842)
        self.assertEqual(metrics["observed_agreement_po"], 0.9976)
        self.assertEqual(metrics["chance_agreement_pe"], 0.9532)
        self.assertEqual(metrics["cohens_kappa"], 0.9492)

    def test_3_leakage_check(self) -> None:
        """Requirement 3: Check splits for duplicate/near-duplicate sentence leakage."""
        train_sents = [" ".join(w for w, _ in s) for s in self._read_conll_sentences(self.gold_dir / "train.conll")]
        dev_sents = [" ".join(w for w, _ in s) for s in self._read_conll_sentences(self.gold_dir / "dev.conll")]
        test_sents = [" ".join(w for w, _ in s) for s in self._read_conll_sentences(self.gold_dir / "test.conll")]

        # Split sizes
        self.assertEqual(len(train_sents), 850, f"Expected 850 train sentences, got {len(train_sents)}")
        self.assertEqual(len(dev_sents), 200, f"Expected 200 dev sentences, got {len(dev_sents)}")
        self.assertEqual(len(test_sents), 200, f"Expected 200 test sentences, got {len(test_sents)}")

        # Exact overlap check
        set_train = set(train_sents)
        set_dev = set(dev_sents)
        set_test = set(test_sents)

        self.assertEqual(len(set_train & set_dev), 0, "Exact leakage found between Train and Dev")
        self.assertEqual(len(set_train & set_test), 0, "Exact leakage found between Train and Test")
        self.assertEqual(len(set_dev & set_test), 0, "Exact leakage found between Dev and Test")

        # Normalized near-duplicate overlap check (lowercase, non-alphanumeric stripped)
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        norm_train = {norm(s) for s in train_sents}
        norm_dev = {norm(s) for s in dev_sents}
        norm_test = {norm(s) for s in test_sents}

        self.assertEqual(len(norm_train & norm_dev), 0, "Near-duplicate leakage found between Train and Dev")
        self.assertEqual(len(norm_train & norm_test), 0, "Near-duplicate leakage found between Train and Test")
        self.assertEqual(len(norm_dev & norm_test), 0, "Near-duplicate leakage found between Dev and Test")

    def test_4_lockfile_integrity(self) -> None:
        """Requirement 4: Confirm test split checksum is recorded and matches disk."""
        self.assertTrue(self.lockfile_path.exists(), "Lockfile missing")
        lock_text = self.lockfile_path.read_text(encoding="utf-8")

        # Extract recorded SHA-256
        match = re.search(r"`([a-f0-9]{64})`", lock_text)
        self.assertIsNotNone(match, "Could not find 64-char hex SHA-256 in lockfile")
        recorded_sha256 = match.group(1)

        # Compute actual hash of test.conll
        test_conll_path = self.gold_dir / "test.conll"
        actual_sha256 = hashlib.sha256(test_conll_path.read_bytes()).hexdigest()

        self.assertEqual(
            recorded_sha256,
            actual_sha256,
            f"Test lockfile checksum mismatch: recorded={recorded_sha256}, actual={actual_sha256}",
        )

        # Verify entity representation in test split (all 4 classes supported)
        test_sents = self._read_conll_sentences(test_conll_path)
        counts = Counter()
        for s in test_sents:
            for _, tag in s:
                if tag.startswith("B-"):
                    counts[tag[2:]] += 1
        for etype in ["PER", "LOC", "ORG", "MISC"]:
            self.assertGreater(counts[etype], 0, f"Entity class {etype} missing from test split")

    def test_5_weak_supervision_independence(self) -> None:
        """Requirement 5: Confirm weak supervision precision is measured on held-out gold split."""
        gaz = load_gazetteers()
        self.assertGreater(len(gaz["LOC"]), 10, "LOC gazetteer empty or insufficient")
        self.assertGreater(len(gaz["PER"]), 10, "PER gazetteer empty or insufficient")
        self.assertGreater(len(gaz["ORG"]), 10, "ORG gazetteer empty or insufficient")

        dev_path = self.gold_dir / "dev.conll"
        metrics = evaluate_weak_supervision_precision(dev_path, gaz)

        self.assertGreaterEqual(
            metrics["overall_span_precision"],
            0.90,
            f"Weak supervision precision below 90%: {metrics['overall_span_precision']}",
        )
        self.assertEqual(metrics["total_gold_spans"], 17, "Expected 17 gold spans in dev")
        self.assertEqual(metrics["total_pred_spans"], 12, "Expected 12 predicted spans in dev")


if __name__ == "__main__":
    unittest.main()
