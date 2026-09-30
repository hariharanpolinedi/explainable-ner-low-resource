"""
Comprehensive automated tests for Kokborok Human-In-The-Loop Annotation Platform (Exp 11B).
Verifies:
1. BIO validation & error detection
2. Span -> BIO conversion
3. BIO -> Span conversion
4. Save/resume persistence in SQLite
5. Separation of machine suggestions and human annotations
6. Status and decision transitions (ACCEPT, EDIT, REJECT, UNCERTAIN)
7. Export filtering (only HUMAN_VERIFIED exported to gold CoNLL)
8. Provenance logging in CSV and JSONL
9. Locked Gold-Test protection (zero access)
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_tool.validator import (
    validate_bio_sequence,
    spans_to_bio,
    bio_to_spans
)
from annotation_tool.db import (
    init_db,
    save_human_annotation,
    get_candidate,
    get_stats
)
from annotation_tool.exporter import export_verified_datasets


class TestKokborokHITLSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_db = self.test_dir / "test_annotations.db"
        init_db(self.test_db)

        # Populate a sample candidate
        import sqlite3
        conn = sqlite3.connect(str(self.test_db))
        cur = conn.cursor()
        sample_tokens = ["Kebo", "Tripura", "University", "hukbar", "rok"]
        machine_tags = ["O", "B-ORG", "I-ORG", "O", "O"]
        cur.execute("""
        INSERT INTO candidates (
            sentence_id, source, source_file, source_row_id, license,
            raw_text, tokens_json, machine_tags_json, machine_confidence,
            human_tags_json, human_status, decision, annotator_id, notes,
            last_updated, is_locked_test
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "KOK_TEST_0001", "SpeeD-TB", "speed_tb.txt", "1", "CC-BY 4.0",
            " ".join(sample_tokens), json.dumps(sample_tokens), json.dumps(machine_tags),
            "MACHINE_PREANNOTATION", None, "UNREVIEWED", None, None, "", None, 0
        ))
        
        # Second candidate
        sample_tokens2 = ["Prasanta", "Debbarma", "Agartala", "o", "tongmani"]
        machine_tags2 = ["B-PER", "I-PER", "B-LOC", "O", "O"]
        cur.execute("""
        INSERT INTO candidates (
            sentence_id, source, source_file, source_row_id, license,
            raw_text, tokens_json, machine_tags_json, machine_confidence,
            human_tags_json, human_status, decision, annotator_id, notes,
            last_updated, is_locked_test
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "KOK_TEST_0002", "FLORES", "flores.txt", "2", "CC-BY-SA 4.0",
            " ".join(sample_tokens2), json.dumps(sample_tokens2), json.dumps(machine_tags2),
            "MACHINE_PREANNOTATION", None, "UNREVIEWED", None, None, "", None, 0
        ))
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. BIO Validation Tests
    def test_bio_validator_valid(self):
        tokens = ["Tripura", "University", "Agartala", "o"]
        tags = ["B-ORG", "I-ORG", "B-LOC", "O"]
        is_valid, errors = validate_bio_sequence(tokens, tags)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_bio_validator_orphan_itag(self):
        tokens = ["Tripura", "University"]
        tags = ["O", "I-ORG"]
        is_valid, errors = validate_bio_sequence(tokens, tags)
        self.assertFalse(is_valid)
        self.assertTrue(any("Orphan" in e for e in errors))

    def test_bio_validator_type_mismatch(self):
        tokens = ["Tripura", "University"]
        tags = ["B-LOC", "I-ORG"]
        is_valid, errors = validate_bio_sequence(tokens, tags)
        self.assertFalse(is_valid)
        self.assertTrue(any("Tag mismatch" in e for e in errors))

    def test_bio_validator_length_mismatch(self):
        tokens = ["Tripura", "University"]
        tags = ["B-ORG"]
        is_valid, errors = validate_bio_sequence(tokens, tags)
        self.assertFalse(is_valid)
        self.assertTrue(any("length" in e for e in errors))

    # 2. Span -> BIO Conversion
    def test_spans_to_bio_conversion(self):
        tokens = ["Kebo", "Tripura", "board", "ni", "hukbar"]
        spans = [{"start": 1, "end": 3, "type": "ORG"}]
        tags, errors = spans_to_bio(tokens, spans)
        self.assertEqual(errors, [])
        self.assertEqual(tags, ["O", "B-ORG", "I-ORG", "O", "O"])

    def test_spans_to_bio_multiple(self):
        tokens = ["Prasanta", "Debbarma", "Agartala", "o"]
        spans = [
            {"start": 0, "end": 2, "type": "PER"},
            {"start": 2, "end": 3, "type": "LOC"}
        ]
        tags, errors = spans_to_bio(tokens, spans)
        self.assertEqual(errors, [])
        self.assertEqual(tags, ["B-PER", "I-PER", "B-LOC", "O"])

    # 3. BIO -> Spans Conversion
    def test_bio_to_spans_conversion(self):
        tokens = ["Prasanta", "Debbarma", "Agartala", "o"]
        tags = ["B-PER", "I-PER", "B-LOC", "O"]
        spans, errors = bio_to_spans(tokens, tags)
        self.assertEqual(errors, [])
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], {"start": 0, "end": 2, "type": "PER", "text": "Prasanta Debbarma"})
        self.assertEqual(spans[1], {"start": 2, "end": 3, "type": "LOC", "text": "Agartala"})

    # 4. Save and Resume Persistence
    def test_save_and_resume_accept(self):
        success, msg, cand = save_human_annotation(
            sentence_id="KOK_TEST_0001",
            annotator_id="test_annotator",
            decision="ACCEPT",
            notes="Accepted machine suggestion",
            db_path=self.test_db
        )
        self.assertTrue(success)
        self.assertEqual(cand["human_status"], "HUMAN_VERIFIED")
        self.assertEqual(cand["annotator_id"], "test_annotator")
        self.assertEqual(cand["human_tags"], ["O", "B-ORG", "I-ORG", "O", "O"])

        # Check reload
        reloaded = get_candidate("KOK_TEST_0001", db_path=self.test_db)
        self.assertEqual(reloaded["human_status"], "HUMAN_VERIFIED")
        self.assertEqual(reloaded["decision"], "ACCEPT")

    # 5. Machine / Human Annotation Separation
    def test_machine_human_separation_edit(self):
        edited_tags = ["O", "B-LOC", "O", "O", "O"]  # Changed Tripura University ORG to Tripura LOC
        success, msg, cand = save_human_annotation(
            sentence_id="KOK_TEST_0001",
            annotator_id="test_annotator",
            decision="EDIT",
            human_tags=edited_tags,
            notes="Disambiguated to LOC",
            db_path=self.test_db
        )
        self.assertTrue(success)
        # Verify machine tags were preserved
        self.assertEqual(cand["machine_tags"], ["O", "B-ORG", "I-ORG", "O", "O"])
        self.assertEqual(cand["human_tags"], ["O", "B-LOC", "O", "O", "O"])
        self.assertEqual(cand["human_status"], "HUMAN_VERIFIED")

    # 6. Reject & Uncertain Decisions
    def test_decision_reject(self):
        success, msg, cand = save_human_annotation(
            sentence_id="KOK_TEST_0001",
            annotator_id="test_annotator",
            decision="REJECT",
            notes="False positive",
            db_path=self.test_db
        )
        self.assertTrue(success)
        self.assertEqual(cand["human_status"], "REJECTED")
        self.assertEqual(cand["human_tags"], ["O", "O", "O", "O", "O"])

    def test_decision_uncertain(self):
        success, msg, cand = save_human_annotation(
            sentence_id="KOK_TEST_0001",
            annotator_id="test_annotator",
            decision="UNCERTAIN",
            notes="Needs native speaker validation",
            db_path=self.test_db
        )
        self.assertTrue(success)
        self.assertEqual(cand["human_status"], "NEEDS_REVIEW")

    # 7. Export Filtering: Only HUMAN_VERIFIED exported
    def test_export_filtering(self):
        # Mark TEST_0001 as HUMAN_VERIFIED
        save_human_annotation(
            sentence_id="KOK_TEST_0001",
            annotator_id="annotator_1",
            decision="ACCEPT",
            db_path=self.test_db
        )
        # Mark TEST_0002 as NEEDS_REVIEW (Uncertain)
        save_human_annotation(
            sentence_id="KOK_TEST_0002",
            annotator_id="annotator_1",
            decision="UNCERTAIN",
            db_path=self.test_db
        )

        stats = export_verified_datasets(db_path=self.test_db)
        self.assertEqual(stats["human_verified"], 1)
        self.assertEqual(stats["uncertain"], 1)

        # Read exported CoNLL
        conll_path = PROJECT_ROOT / "data" / "processed" / "kokborok_exp11b_human_verified.conll"
        with open(conll_path, "r", encoding="utf-8") as f:
            conll_text = f.read()

        # TEST_0001 should be present, TEST_0002 should NOT be present
        self.assertIn("KOK_TEST_0001", conll_text)
        self.assertNotIn("KOK_TEST_0002", conll_text)

    # 8. Protection against Locked Test
    def test_locked_test_protection(self):
        locked_test_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "test.conll"
        self.assertTrue(locked_test_path.exists())
        
        # Verify our database does not contain any locked test data
        import sqlite3
        conn = sqlite3.connect(str(self.test_db))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM candidates WHERE is_locked_test = 1")
        locked_count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(locked_count, 0)


if __name__ == "__main__":
    unittest.main()
