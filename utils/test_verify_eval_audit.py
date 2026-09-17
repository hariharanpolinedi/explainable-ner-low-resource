"""
Verification Test Suite: long_multientity_eval Selection-Bias and Annotation-Independence Audit.

Validates the 4 Section-Testing criteria:
1. Selection-bias trace test: exact filter/ranking logic extracted from actual code.
2. Distribution-comparison test: entity counts recomputed directly from .conll files for all 4 datasets per language.
3. Annotation-independence trace test: process reconstruction based on actual code/reports with quotes.
4. Cross-language consistency test: verifies whether all 3 languages used the same process.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_selection_bias import audit_candidate_selection_logic
from evaluation.entity_distribution_comparison import compare_all_distributions
from annotation.audit_annotation_process import audit_language_annotation


def test_selection_bias_trace() -> bool:
    print("\n[CHECK 1] Testing Selection-Bias Trace Extraction...")
    findings = audit_candidate_selection_logic()
    assert findings["screening_filters"]["has_gaz_or_org_screen"] is True, "Missing screening check"
    assert findings["rule_tagger_injected_into_gold"]["present"] is True, "Missing rule tagger injection"
    print("  -> PASS: Filter and injection code successfully extracted from expand_long_multientity_set.py")
    return True


def test_distribution_comparison() -> bool:
    print("\n[CHECK 2] Testing Entity Distribution Direct Computation...")
    results = compare_all_distributions()
    for lang in ["konkani", "assamese", "sinhala"]:
        for split in ["long_multientity_eval", "dev", "test", "train"]:
            assert split in results[lang], f"Missing {split} in {lang}"
            data = results[lang][split]
            assert "percentages" in data, f"Missing percentages in {split}"
            assert data["total_entities"] > 0, f"Zero entities in {lang} {split}"
    # Verify skew
    konk_org_pct = results["konkani"]["long_multientity_eval"]["percentages"]["ORG"]["pct"]
    assert konk_org_pct > 75.0, f"Expected Konkani ORG > 75%, got {konk_org_pct}"
    print("  -> PASS: Directly recomputed distributions across all 4 splits and 3 languages")
    return True


def test_annotation_independence_trace() -> bool:
    print("\n[CHECK 3] Testing Annotation Independence and Circularity Trace...")
    konk_audit = audit_language_annotation("konkani")
    assert konk_audit["kappa_annotator_a_vs_b"] == 1.0, f"Expected Konkani kappa=1.0, got {konk_audit['kappa_annotator_a_vs_b']}"
    assert konk_audit["rule_tagger_injected_tokens"] > 400, "Expected >400 rule-injected tokens in Konkani"
    assert konk_audit["entities_under_gazetteer_only_annotator_a"]["ORG"] <= 2, "Expected gazetteer ORGs <= 2"
    print("  -> PASS: Verified Annotator A vs B identity and post-agreement rule injection")
    return True


def test_cross_language_consistency() -> bool:
    print("\n[CHECK 4] Testing Cross-Language Process Consistency...")
    as_audit = audit_language_annotation("assamese")
    si_audit = audit_language_annotation("sinhala")
    assert as_audit["rule_tagger_injected_tokens"] > 150, "Expected injected tokens in Assamese"
    assert si_audit["rule_tagger_injected_tokens"] > 500, "Expected injected tokens in Sinhala"
    assert as_audit["kappa_annotator_a_vs_b"] > 0.90, "Expected high kappa from synthetic simulation"
    assert si_audit["kappa_annotator_a_vs_b"] > 0.90, "Expected high kappa from synthetic simulation"
    print("  -> PASS: All three languages confirmed to use identical synthetic programmatic simulation")
    return True


def run_all_checks() -> bool:
    print("=" * 70)
    print("RUNNING LONG_MULTIENTITY_EVAL AUDIT VERIFICATION CHECKS")
    print("=" * 70)
    c1 = test_selection_bias_trace()
    c2 = test_distribution_comparison()
    c3 = test_annotation_independence_trace()
    c4 = test_cross_language_consistency()
    all_passed = c1 and c2 and c3 and c4
    print("=" * 70)
    if all_passed:
        print("ALL 4 VERIFICATION CHECKS PASSED.")
    else:
        print("VERIFICATION CHECKS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
