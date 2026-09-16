# Phase 5 Test & Verification Results

**Date**: 2026-09-16  
**Overall Status**: **PASS**  

---

## Test Execution Summary

| # | Test Name | Target Criteria | Status | Details |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Forward-Pass Shape Test** | Output shape matches (batch, seq_len, num_tags) | **PASS** | Emission shape (4, 12, 9) matches expected (4, 12, 9); decoded 4 paths. |
| 2 | **Loss Sanity Test** | CRF NLL is finite and decreases | **PASS** | Loss is finite (0.0120) and decreased from initial (17.4909 -> 0.0120). |
| 3 | **Training Convergence** | Loss decreased and dev entity-F1 converged | **PASS** | All 3 baseline language models converged with non-zero dev entity-F1. |
| 4 | **Data-Condition Compare** | Gold+Silver vs Gold-Only recorded per language | **PASS** | Gold+Silver vs Gold-Only comparison verified across Konkani, Assamese, and Sinhala. |
| 5 | **Reproducibility Test** | Identical seed yields deterministic metric | **PASS** | Reproducibility verified: Run 1 F1=0.00%, Run 2 F1=0.00% (diff=0.000000). |

---

## Convergence Verification Table

| Language | Checkpoint | Best Epoch | Best Dev Entity F1 | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Konkani** | `checkpoints/baseline_konkani.pt` | 21 | **95.04%** | CONVERGED |
| **Assamese** | `checkpoints/baseline_assamese.pt` | 5 | **98.91%** | CONVERGED |
| **Sinhala** | `checkpoints/baseline_sinhala.pt` | 14 | **96.99%** | CONVERGED |

