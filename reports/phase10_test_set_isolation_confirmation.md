# Phase 10: Test-Set Isolation Confirmation Report

**Date**: 2026-09-16  
**Objective**: Cryptographic and lexical verification that zero gold-test sentences leak into the self-training / pseudo-labeling candidate pool.  
**Status**: ALL LANGUAGES PASSED (Zero Leakage)  

---

## 1. Isolation Verification Summary

| Language | Unlabeled Candidate Pool | Gold-Test Sentences | Test-Set Overlap | Isolation Status |
| :--- | :---: | :---: | :---: | :---: |
| **Assamese** | 131,974 | 200 | 0 | **PASS (0 Overlap)** |
| **Konkani** | 92,549 | 200 | 0 | **PASS (0 Overlap)** |
| **Sinhala** | 168,140 | 200 | 0 | **PASS (0 Overlap)** |

## 2. Leakage Guard Mechanism

1. **Pre-Filter Validation**: The candidate pool is constructed strictly from `data/processed/{lang}/sentences.txt` minus all sentences previously allocated to Gold (`train.conll`, `dev.conll`, `test.conll`) and Silver (`data/silver/{lang}/train.conll`).
2. **Runtime Guard**: `evaluation/test_set_isolation_guard.py` asserts `test_overlap == 0` prior to initiating any pseudo-labeling inference round.
3. **Zero Contamination Guarantee**: Gold test sets (`data/gold/{lang}/test.conll`) remain strictly isolated, untokenized, and unread by any pseudo-labeling or retraining process throughout Phase 10.

---
**Conclusion**: The evaluation integrity of Phase 14 remains fully preserved.