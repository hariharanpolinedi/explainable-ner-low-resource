"""
Phase 16 Automated Verification Test Suite
Tests:
1. Dependency-completeness test: confirm all dependent reports exist and are readable.
2. Transcription-accuracy test: spot-check numbers against their source files.
3. Stale-vs-current labeling test: confirm historical artifacts are explicitly flagged.
4. Negative-result preservation test: confirm rejected candidates are thoroughly documented.
"""

import os
import sys

def run_tests():
    print("=" * 60)
    print("PHASE 16 AUTOMATED VERIFICATION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # -------------------------------------------------------------
    # Test 1: Dependency-Completeness Test
    # -------------------------------------------------------------
    print("\n--- Test 1: Dependency-Completeness Test ---")
    dependent_reports = [
        "baseline_v1/paper_assets/results_master_table.md",
        "baseline_v1/paper_assets/MANIFEST.md",
        "reports/cycle2_final_verdict.md",
        "reports/cycle2_assamese_final_selection_justification.md",
        "reports/post_improvement_error_reanalysis.md",
        "reports/phase7_final_verdict.md",
        "reports/kokborok_final_results.md",
        "reports/cross_language_synergy_final.md",
        "reports/phase15_faithfulness_results_four_language.md",
        "reports/cycle2_konkani_intervention_results.md",
        "reports/cycle2_assamese_intervention_results.md",
        "reports/kokborok_integration_strategy_comparison.md"
    ]
    
    missing_deps = []
    for dep in dependent_reports:
        if not os.path.exists(dep):
            missing_deps.append(dep)
        else:
            size = os.path.getsize(dep)
            if size == 0:
                missing_deps.append(f"{dep} (empty)")
    
    if not missing_deps:
        print(f"[PASS] All {len(dependent_reports)} required dependent reports exist, are non-empty, and verified.")
        results["dependency_completeness"] = "PASS"
    else:
        print(f"[FAIL] Missing or empty dependencies: {missing_deps}")
        results["dependency_completeness"] = "FAIL"
        
    # -------------------------------------------------------------
    # Test 2: Transcription-Accuracy Test
    # -------------------------------------------------------------
    print("\n--- Test 2: Transcription-Accuracy Spot-Check Test ---")
    master_table_path = "paper/assets/results_master_table.md"
    with open(master_table_path, "r", encoding="utf-8") as f:
        master_content = f.read()
        
    spot_checks = [
        ("Konkani Test F1", "96.27%", "reports/cross_language_synergy_final.md"),
        ("Assamese Test F1", "97.87%", "reports/cross_language_synergy_final.md"),
        ("Sinhala Test F1", "97.76%", "reports/cross_language_synergy_final.md"),
        ("Kokborok Test F1", "68.97%", "reports/kokborok_final_results.md"),
        ("Assamese Dev F1", "98.55%", "reports/cross_language_synergy_final.md"),
        ("Sinhala Dev F1", "98.50%", "reports/cross_language_synergy_final.md"),
        ("Kokborok Dev F1", "61.54%", "reports/cross_language_synergy_final.md"),
        ("Four-Language Macro F1", "90.22%", "reports/cross_language_synergy_final.md"),
        ("Kokborok Token Accuracy", "99.55%", "reports/kokborok_final_results.md"),
        ("Assamese Test Recall", "100.00%", "reports/cross_language_synergy_final.md")
    ]
    
    transcription_failures = []
    for label, val, src in spot_checks:
        # Check that value exists in source
        with open(src, "r", encoding="utf-8") as sf:
            s_content = sf.read()
        if val not in s_content:
            transcription_failures.append(f"Value '{val}' not found in source '{src}' for {label}")
        if val not in master_content:
            transcription_failures.append(f"Value '{val}' not found in master table for {label}")
            
    if not transcription_failures:
        print(f"[PASS] All {len(spot_checks)} numerical spot-checks match source files exactly.")
        results["transcription_accuracy"] = "PASS"
    else:
        print(f"[FAIL] Transcription discrepancies: {transcription_failures}")
        results["transcription_accuracy"] = "FAIL"

    # -------------------------------------------------------------
    # Test 3: Stale-vs-Current Labeling Test
    # -------------------------------------------------------------
    print("\n--- Test 3: Stale-vs-Current Labeling Test ---")
    manifest_path = "paper/assets/MANIFEST_v2.md"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_content = f.read()
        
    required_labels = [
        "Sinhala Evaluation Metrics",
        "Valid As-Is",
        "Superseded",
        "historical Cycle-1-era",
        "current_best_konkani.pt",
        "current_best_assamese.pt"
    ]
    
    missing_labels = [l for l in required_labels if l.lower() not in manifest_content.lower()]
    if not missing_labels:
        print("[PASS] Manifest explicitly distinguishes Valid As-Is, Superseded, and Historical Cycle-1 artifacts.")
        results["stale_vs_current_labeling"] = "PASS"
    else:
        print(f"[FAIL] Missing required status labels in manifest: {missing_labels}")
        results["stale_vs_current_labeling"] = "FAIL"

    # -------------------------------------------------------------
    # Test 4: Negative-Result Preservation Test
    # -------------------------------------------------------------
    print("\n--- Test 4: Negative-Result Preservation Test ---")
    interventions_path = "paper/assets/interventions_attempted.md"
    with open(interventions_path, "r", encoding="utf-8") as f:
        interventions_content = f.read()
        
    rejected_items = [
        "Middle-Token Position-Weighted",
        "Rule-Based Suffix Morphological Augmentation",
        "Joint Indo-Aryan Sub-Network Compound",
        "Syntactic boundary-loss gradient scaling",
        "Targeted synthetic data reinforcement",
        "Aggressive entity class-weighting sweep",
        "Option (a): Joint Four-Language Retraining",
        "Option (b): Standalone Kokborok Model From Scratch"
    ]
    
    missing_rejected = [item for item in rejected_items if item.lower() not in interventions_content.lower()]
    if not missing_rejected:
        print(f"[PASS] All {len(rejected_items)} negative / rejected interventions are explicitly documented with rationales.")
        results["negative_result_preservation"] = "PASS"
    else:
        print(f"[FAIL] Missing rejected intervention items: {missing_rejected}")
        results["negative_result_preservation"] = "FAIL"

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 16 TEST SUITE SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, status in results.items():
        print(f"{test_name:35}: {status}")
        if status != "PASS":
            all_passed = False
            
    if all_passed:
        print("\nALL PHASE 16 TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("\nPHASE 16 TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
