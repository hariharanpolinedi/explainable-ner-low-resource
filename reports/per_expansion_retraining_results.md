# PER Coverage Expansion Retraining Results

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Evaluation Split**: Locked Gold-Dev ONLY (`data/gold/{lang}/dev.conll`)

---

## 1. Divergence Gate Verification Across Languages

| Language | Gate Status | Parameters Changed | L2 Norm Diff | Max Diff | Saved Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | **PASSED** | 12.89% | 2.8590 | 0.0127 | `checkpoints/per_fixed_konkani.pt` |
| **Sinhala** | **PASSED** | 13.44% | 4.3153 | 0.0368 | `checkpoints/per_fixed_sinhala.pt` |
| **Assamese** | **PASSED** | 12.59% | 1.8431 | 0.0066 | `checkpoints/per_fixed_assamese.pt` |

---

## 2. Before vs After Performance on Locked Gold-Dev

### Konkani

| Metric | Baseline | Retrained (`per_fixed`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **96.69%** | **94.74%** | **-1.95%** |
| Token Accuracy | 99.59% | 99.28% | -0.31% |
| PER F1 | 91.89% | 82.93% | -8.96% |
| LOC F1 | 93.10% | 93.10% | +0.00% |
| ORG F1 | 98.63% | 97.30% | -1.33% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

### Sinhala

| Metric | Baseline | Retrained (`per_fixed`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **98.13%** | **97.76%** | **-0.37%** |
| Token Accuracy | 99.78% | 99.73% | -0.05% |
| PER F1 | 100.00% | 100.00% | +0.00% |
| LOC F1 | 93.98% | 93.98% | +0.00% |
| ORG F1 | 100.00% | 98.25% | -1.75% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

### Assamese

| Metric | Baseline | Retrained (`per_fixed`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **98.92%** | **97.16%** | **-1.76%** |
| Token Accuracy | 99.89% | 99.58% | -0.31% |
| PER F1 | 100.00% | 90.00% | -10.00% |
| LOC F1 | 98.73% | 98.11% | -0.62% |
| ORG F1 | 96.55% | 96.55% | +0.00% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

