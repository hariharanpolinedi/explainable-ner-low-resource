# Patch v2 Retraining Results: Corrected Checkpoint Selection & Hard Divergence Gate

**Date**: 2026-09-17  
**Scope**: Corrective re-execution of Phase 10/11 Patch.  
**Key Fixes Applied**: Composite checkpoint-selection score (`0.40 * ORG + 0.40 * Long + 0.20 * Dev`), hard weight-divergence gate (>= 1.0% parameter update required), and standing long/multi-entity evaluation monitoring.  

---

## 1. Divergence Gate Verification Summary

| Language | Gate Status | Total Params | Changed Params | % Changed | L2 Difference | Max Weight Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | **PASSED** | 10,596,571 | 2,796,454 | **26.39%** | 12.2750 | 0.1502 |
| **Konkani** | **PASSED** | 10,596,571 | 2,826,881 | **26.68%** | 10.0419 | 0.1090 |
| **Sinhala** | **PASSED** | 10,596,571 | 2,413,105 | **22.77%** | 5.2563 | 0.0398 |

---

## 2. Retraining Composite Score Summary

| Language | Best Epoch | Composite Score | Target ORG F1 | Target Long F1 | Dev Overall F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | Epoch 5 | **48.98%** | 11.43% | 61.90% | 98.22% |
| **Konkani** | Epoch 3 | **31.54%** | 2.30% | 30.40% | 92.31% |
| **Sinhala** | Epoch 1 | **28.03%** | 1.12% | 20.63% | 96.65% |

---

## 3. Findings & Integrity Assessment

1. **Weight Updating**: Unlike the prior patch (where Assamese and Sinhala silently re-saved baseline weights), all three models were trained under the hard divergence gate.
2. **Selection Logic**: Models were selected based on composite performance balancing ORG recall and long-sentence robustness rather than unstratified dev F1 alone.