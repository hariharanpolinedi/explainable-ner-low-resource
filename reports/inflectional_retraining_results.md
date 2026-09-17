# Cross-Language Inflectional Augmentation Retraining Results

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Evaluation Split**: Locked Gold-Dev ONLY (`data/gold/{lang}/dev.conll`)

---

## 1. Divergence Gate Verification Across Languages

| Language | Gate Status | Parameters Changed | L2 Norm Diff | Max Diff | Saved Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | **PASSED** | 12.94% | 3.8140 | 0.0241 | `checkpoints/inflected_konkani.pt` |
| **Sinhala** | **PASSED** | 13.42% | 4.2072 | 0.0292 | `checkpoints/inflected_sinhala.pt` |
| **Assamese** | **PASSED** | 13.15% | 4.1524 | 0.0224 | `checkpoints/inflected_assamese.pt` |

---

## 2. Before vs After Performance on Locked Gold-Dev

### Konkani

| Metric | Baseline | Retrained (`inflected`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **96.69%** | **96.69%** | **+0.00%** |
| Token Accuracy | 99.59% | 99.56% | -0.03% |
| PER F1 | 91.89% | 91.89% | +0.00% |
| LOC F1 | 93.10% | 93.10% | +0.00% |
| ORG F1 | 98.63% | 98.63% | +0.00% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

### Sinhala

| Metric | Baseline | Retrained (`inflected`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **98.13%** | **95.62%** | **-2.51%** |
| Token Accuracy | 99.78% | 99.48% | -0.30% |
| PER F1 | 100.00% | 100.00% | +0.00% |
| LOC F1 | 93.98% | 87.64% | -6.34% |
| ORG F1 | 100.00% | 98.25% | -1.75% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

### Assamese

| Metric | Baseline | Retrained (`inflected`) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Entity F1** | **98.92%** | **98.57%** | **-0.35%** |
| Token Accuracy | 99.89% | 99.86% | -0.03% |
| PER F1 | 100.00% | 100.00% | +0.00% |
| LOC F1 | 98.73% | 98.14% | -0.59% |
| ORG F1 | 96.55% | 96.55% | +0.00% |
| MISC F1 | 100.00% | 100.00% | +0.00% |

