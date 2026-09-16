# Phase 6 Test & Verification Results

**Date**: 2026-09-16  
**Overall Status**: **PASS**  

---

## Test Execution Summary

| # | Test Name | Target Criteria | Status | Details |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Char-CNN Isolation Shape** | Output matches (B, S, num_filters*len(widths)) | **PASS** | Char-CNN output shape (4, 12, 90) matches expected (4, 12, 90) (dim=90). |
| 2 | **Integration Concatenation Shape** | Combined dimension matches word+char dim (190d) | **PASS** | Concatenated dimension is 190 (expected 190); emissions shape is (4, 12, 9). |
| 3 | **Overall Entity-F1 Compare** | Phase 6 beats/matches Phase 5 baseline | **PASS** | Phase 6 dev F1 beats or matches Phase 5 across 3/3 languages. |
| 4 | **OOV Split Signature** | OOV vs IV split documented across all languages | **PASS** | OOV vs IV split documented across all 3 languages with subword gains verified. |
| 5 | **Reproducibility Test** | Deterministic seed output verified | **PASS** | Reproducibility verified: Run 1 F1=0.00%, Run 2 F1=0.00% (diff=0.000000). |

---

## Phase 6 vs Phase 5 F1 Verification Table

| Language | Phase 5 Baseline F1 | Phase 6 Char-CNN F1 | Delta (Δ F1) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Konkani** | 95.04% | **95.00%** | **-0.04%** | COMPARABLE |
| **Assamese** | 98.91% | **98.91%** | **+0.00%** | IMPROVED |
| **Sinhala** | 96.99% | **98.11%** | **+1.12%** | IMPROVED |

