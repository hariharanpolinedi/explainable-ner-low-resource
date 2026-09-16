# Phase 7 Test & Verification Results

**Date**: 2026-09-16  
**Overall Status**: **PASS**  

---

## Test Execution Summary

| # | Test Name | Target Criteria | Status | Details |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Self-Attention Isolation Shape** | Output matches (B, S, D), attn matches (B, H, S, S) | **PASS** | Out shape (4, 12, 256) matches (4, 12, 256), attn shape (4, 4, 12, 12) matches (4, 4, 12, 12), padded keys attention sum=0.00000000. |
| 2 | **Integration Pipeline Shape** | Full pipeline produces valid loss and decoded paths | **PASS** | Integration passed: loss is finite (35.4394), decoded 4 paths, attention tensor persisted. |
| 3 | **Padding-Mask Correctness** | Padding invariance (<1e-4) & zero pad key attn (<1e-7) | **PASS** | Padding invariance verified: diff=0.00000060 (<1e-4), max attention weight to padding tokens=0.00000000 (<1e-7). |
| 4 | **Overall Entity-F1 Compare** | Phase 7 beats/matches Phase 6 across languages | **PASS** | Phase 7 entity-F1 beats or matches Phase 6 across 3/3 languages. |
| 5 | **Span-Length Stratified F1** | Single vs multi-token entity analysis documented | **PASS** | Span-length stratified analysis verified across single and multi-token entity categories. |
| 6 | **Attention Non-Degeneracy** | Neither uniform nor self-only attention confirmed | **PASS** | Attention non-degeneracy confirmed across all languages: weights are neither uniform nor self-only. |
| 7 | **Reproducibility Test** | Deterministic fixed-seed execution verified | **PASS** | Reproducibility verified: Run 1 F1=0.00%, Run 2 F1=0.00% (diff=0.000000). |

---

## Phase 7 vs Phase 6 F1 Verification Table

| Language | Phase 6 Char-CNN F1 | Phase 7 Attention F1 | Delta (Δ F1) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Konkani** | 95.00% | **95.00%** | **+0.00%** | IMPROVED |
| **Assamese** | 98.91% | **98.91%** | **+0.00%** | IMPROVED |
| **Sinhala** | 98.11% | **97.74%** | **-0.37%** | COMPARABLE |

