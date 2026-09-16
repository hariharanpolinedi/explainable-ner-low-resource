# Phase 10 Test & Validation Report

**Date**: 2026-09-16  
**Execution Environment**: Windows (Local Machine), NVIDIA GeForce RTX 3050 Laptop GPU (CUDA Enabled)  
**Status**: ALL 5 TESTS PASSED (100%)

---

## 1. Summary of Test Results

| Test ID | Test Name | Description | Status | Execution Time |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `test_1_test_set_isolation_guard` | Verify zero overlap between pseudo-labeling candidates and `data/gold/{lang}/test.conll` across all 3 languages, and confirm `CriticalLeakageError` triggers upon simulated leak | **PASS** | 0.48s |
| 2 | `test_2_confidence_filter_correctness` | Verify CRF sequence confidence filtering strictly retains items $\ge \text{threshold}$ and excludes sub-threshold predictions | **PASS** | 0.52s |
| 3 | `test_3_pseudo_volume_cap_enforcement` | Confirm that volume cap is strictly enforced and output count never exceeds the configured ceiling | **PASS** | 0.44s |
| 4 | `test_4_round_by_round_dev_f1_tracking_and_rollback` | Confirm dev-F1 history tracking across rounds and verify degraded checkpoints trigger immediate rollback | **PASS** | 0.01s |
| 5 | `test_5_reproducibility` | Re-run inference with identical seed, confirming exact deterministic tag and confidence score reproducibility | **PASS** | 0.97s |

---

## 2. Detailed Test Verifications

### 2.1 Test-Set Isolation Guard
- **Check**: Checked all 92,549 Konkani, 131,974 Assamese, and 168,140 Sinhala unlabeled candidates against their respective gold test sets (`data/gold/{lang}/test.conll`).
- **Result**: Zero test sentences detected in any language candidate pool (`test_overlap_count = 0`).
- **Contamination Simulation**: Injected a gold-test sentence into the candidate pool; `verify_pool_isolation` immediately raised `CriticalLeakageError`.

### 2.2 Confidence Filter Correctness
- **Check**: Ran inference over sample sentences at threshold 0.90; confirmed all retained sentences have $\text{conf}(x) \ge 0.90$.
- **Bound Check**: Set threshold to impossible 1.05; confirmed 0 items retained.

### 2.3 Volume Cap Enforcement
- **Check**: Set cap = 2 on a batch of qualifying sentences; verified `retained_count == 2` and `volume_cap_applied == True`.

### 2.4 Dev-F1 Tracking and Rollback Guard
- **Check**: Tested progression across rounds; verified that any drop in dev-F1 triggers `ROLLED_BACK_TO_PREVIOUS` and preserves the previous best checkpoint.

### 2.5 Reproducibility
- **Check**: Re-ran confidence filter with identical random seed; verified identical token tags and sequence confidence scores within $10^{-4}$ tolerance.

---
**Conclusion**: All Phase 10 validation checks have passed without warnings or errors.
