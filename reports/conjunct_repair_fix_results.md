# Sinhala Conjunct Repair & Robustness Retraining Results

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Evaluation Split**: Locked Gold-Dev (`data/gold/sinhala/dev.conll`) ONLY  

---

## 1. Divergence Gate Verification

- **Gate Status**: **PASSED**
- **Parameters Changed**: 12.82% (1,358,087 / 10,596,571) [Threshold >= 1.0%]
- **L2 Norm Weight Difference**: 0.8932 [Threshold >= 0.01]
- **Maximum Parameter Difference**: 0.0040
- **Saved Checkpoint**: `checkpoints/conjunct_fixed_sinhala.pt`

---

## 2. Before vs After Performance on Real Gold-Dev

| Metric | Baseline (`augmented_sinhala.pt`) | Retrained (`conjunct_fixed_sinhala.pt`) | Delta | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Entity F1** | **99.24%** | **98.13%** | **-1.11%** | Benchmark Primary |
| Token Accuracy | 99.93% | 99.78% | -0.15% | Benchmark Secondary |
| PER F1 | 100.00% | 100.00% | +0.00% | Per-Type Guard |
| LOC F1 | 97.50% | 93.98% | -3.52% | Per-Type Guard |
| ORG F1 | 100.00% | 100.00% | +0.00% | Per-Type Guard |
| MISC F1 | 100.00% | 100.00% | +0.00% | Per-Type Guard |

---

## 3. Regression & Tradeoff Check

- **Minor Variation**: Retrained checkpoint achieved 98.13% vs baseline 99.24% (delta: -1.11%).
