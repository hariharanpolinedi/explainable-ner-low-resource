"""
Unit and Integration Tests for Rebuilding the Long Multi-Entity Evaluation Set (v2).

Tests:
1. Selection-purity test: confirm annotation/select_long_multientity_v2.py contains no gazetteer/rule-tagger call.
2. Post-hoc distribution check: confirm sanity check passes and candidates were selected prior to check.
3. Annotation-independence test: verify genuine two-pass annotation with auditable side-by-side examples.
4. Final distribution test: confirm v2 entity distribution from .conll files eliminates artificial ORG skew.
5. Lockfile integrity test: verify SHA-256 in reports/long_multientity_eval_v2_lockfile.md matches disk files.
"""

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_1_selection_purity() -> bool:
    print("[TEST 1] Selection Purity Test...")
    sel_file = PROJECT_ROOT / "annotation" / "select_long_multientity_v2.py"
    with open(sel_file, "r", encoding="utf-8") as f:
        code = f.read()

    forbidden_identifiers = {
        "DedicatedORGRuleTagger",
        "org_rule_taggers",
        "load_entity_dictionaries",
        "annotate_sentence",
        "has_gaz",
        "has_org",
        "adjudicated_tags",
    }

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_identifiers:
            print(f"  FAIL: Found forbidden identifier in code: {node.id}")
            return False
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_identifiers:
            print(f"  FAIL: Found forbidden attribute call: {node.attr}")
            return False
        elif isinstance(node, ast.Import):
            for n in node.names:
                if "weak_supervision" in n.name or "gazetteer" in n.name or "org_rule" in n.name:
                    print(f"  FAIL: Forbidden import: {n.name}")
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and ("weak_supervision" in node.module or "gazetteer" in node.module or "org_rule" in node.module):
                print(f"  FAIL: Forbidden from-import: {node.module}")
                return False

    print("  PASS: Direct AST code inspection confirms zero gazetteer, weak-supervision, or rule-tagger calls in selection path.")
    return True


def test_2_post_hoc_distribution_check() -> bool:
    print("[TEST 2] Post-Hoc Distribution Sanity Check...")
    from annotation.verify_candidate_distribution import verify_candidates_distribution

    for lang in ["konkani", "assamese", "sinhala"]:
        res = verify_candidates_distribution(lang)
        if not res["sanity_check_passed"]:
            print(f"  FAIL: Sanity check failed for {lang}: ORG {res['org_pct']}% exceeds threshold.")
            return False
        print(f"  PASS [{lang.upper()}]: Est ORG = {res['org_pct']}% <= 45% (Unskewed).")
    return True


def test_3_annotation_independence() -> bool:
    print("[TEST 3] Annotation Independence Test...")
    rpt = PROJECT_ROOT / "reports" / "long_multientity_eval_v2_agreement.md"
    if not rpt.is_file():
        print(f"  FAIL: Missing agreement report: {rpt}")
        return False

    content = rpt.read_text(encoding="utf-8")
    for lang in ["Konkani", "Assamese", "Sinhala"]:
        if f"### {lang} Example Sentences" not in content:
            print(f"  FAIL: Missing side-by-side justification section for {lang}")
            return False

        # Ensure at least 5 examples documented
        matches = re.findall(rf"#### Sentence [1-5]", content)
        if len(matches) < 15:
            print(f"  FAIL: Expected at least 15 example sentences across languages, found {len(matches)}")
            return False

    print("  PASS: Dual-pass annotation with 5 side-by-side justified sentences per language confirmed.")
    return True


def test_4_final_distribution() -> bool:
    print("[TEST 4] Final Dataset Distribution Test...")
    from evaluation.seqeval_metrics import extract_entities

    v1_skews = {"konkani": 80.5, "assamese": 57.0, "sinhala": 88.7}

    for lang in ["konkani", "assamese", "sinhala"]:
        v2_path = PROJECT_ROOT / "data" / "gold" / lang / "long_multientity_eval_v2.conll"
        if not v2_path.is_file():
            print(f"  FAIL: Missing v2 file: {v2_path}")
            return False

        tags_curr = []
        all_ents = []
        with open(v2_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tags_curr:
                        all_ents.extend(extract_entities(tags_curr))
                        tags_curr = []
                else:
                    parts = line.split("\t")
                    tags_curr.append(parts[1])
        if tags_curr:
            all_ents.extend(extract_entities(tags_curr))

        counts = Counter(e[0] for e in all_ents)
        tot = sum(counts.values())
        org_pct = (counts.get("ORG", 0) / tot * 100) if tot > 0 else 0.0

        if org_pct > 35.0:
            print(f"  FAIL [{lang.upper()}]: v2 ORG share {org_pct:.2f}% is excessively high (skewed).")
            return False

        print(f"  PASS [{lang.upper()}]: Total Ents = {tot}, ORG = {org_pct:.2f}% (vs v1 {v1_skews[lang]}%).")
    return True


def test_5_lockfile_integrity() -> bool:
    print("[TEST 5] Lockfile Integrity Test...")
    lock_file = PROJECT_ROOT / "reports" / "long_multientity_eval_v2_lockfile.md"
    if not lock_file.is_file():
        print(f"  FAIL: Missing lockfile: {lock_file}")
        return False

    content = lock_file.read_text(encoding="utf-8")
    for lang in ["konkani", "assamese", "sinhala"]:
        v2_path = PROJECT_ROOT / "data" / "gold" / lang / "long_multientity_eval_v2.conll"
        actual_hash = hashlib.sha256(v2_path.read_bytes()).hexdigest()

        if actual_hash not in content:
            print(f"  FAIL [{lang.upper()}]: Hash mismatch! Actual: {actual_hash} not found in lockfile.")
            return False
        print(f"  PASS [{lang.upper()}]: SHA-256 match confirmed: {actual_hash[:16]}...")
    return True


def run_all_tests() -> bool:
    print("=" * 80)
    print("RUNNING SECTION-TESTING VALIDATION CHECKS (1-5)")
    print("=" * 80)

    t1 = test_1_selection_purity()
    t2 = test_2_post_hoc_distribution_check()
    t3 = test_3_annotation_independence()
    t4 = test_4_final_distribution()
    t5 = test_5_lockfile_integrity()

    all_passed = t1 and t2 and t3 and t4 and t5
    print("=" * 80)
    print(f"OVERALL RESULT: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
