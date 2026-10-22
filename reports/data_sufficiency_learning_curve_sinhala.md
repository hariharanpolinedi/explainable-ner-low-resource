# Data Sufficiency Learning-Curve Report: Sinhala

> **Phase 3 Deliverable — Controlled Empirical Volume Diagnosis**  
> Evaluates models trained from scratch on 25%, 50%, 75%, and 100% slices of the training pool (Gold + Silver = 8,050 sentences).
> Every retrained checkpoint is independently verified via the Checkpoint Divergence Gate.

---

## 1. Learning Curve Trajectory Matrix

| Data Fraction | Training Sentences | Best Epoch | Gold Dev Entity F1 | Entity Precision | Entity Recall | Token Accuracy | Divergence Gate Status | % Params Changed | Checkpoint Path |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **25%** | 2,012 | 5 | **14.04%** | 30.00% | 9.16% | 94.35% | **PASSED** | 17.92% | `checkpoints\learning_curve_sinhala_25pct.pt` |
| **50%** | 4,025 | 5 | **52.03%** | 55.65% | 48.85% | 96.25% | **PASSED** | 21.73% | `checkpoints\learning_curve_sinhala_50pct.pt` |
| **75%** | 6,037 | 5 | **65.05%** | 59.49% | 71.76% | 96.88% | **PASSED** | 24.35% | `checkpoints\learning_curve_sinhala_75pct.pt` |
| **100%** | 8,050 | 5 | **79.74%** | 69.71% | 93.13% | 97.74% | **PASSED** | 26.36% | `checkpoints\learning_curve_sinhala_100pct.pt` |

---

## 2. Per-Type Metric Progression

| Entity Type | 25% F1 (P / R) | 50% F1 (P / R) | 75% F1 (P / R) | 100% F1 (P / R) | Delta (75% -> 100%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PER** | 0.00% (0.0/0.0) | 70.97% (88.0/59.5) | 79.41% (87.1/73.0) | 96.00% (94.7/97.3) | +16.59 pp |
| **LOC** | 25.97% (26.3/25.6) | 40.00% (31.8/53.9) | 53.44% (38.0/89.7) | 60.80% (44.2/97.4) | +7.36 pp |
| **ORG** | 13.33% (100.0/7.1) | 80.77% (87.5/75.0) | 82.35% (91.3/75.0) | 92.86% (92.9/92.9) | +10.51 pp |
| **MISC** | 0.00% (0.0/0.0) | 0.00% (0.0/0.0) | 56.41% (91.7/40.7) | 88.00% (95.7/81.5) | +31.59 pp |

---

## 3. Curve Shape Characterization & Diagnosis

- **Slope between 75% and 100%**: $\Delta F_1 = +14.69$ pp
- **Slope per 1,000 sentences**: `+7.2976 pp/1k sentences`
- **Characterization**: **Steeply Rising (strong data-hungriness, highly sensitive to data volume)**

### Scientific Interpretation
1. **Saturation Diagnostic**: The learning curve trajectory confirms whether expanding raw training volume alone can resolve baseline NER errors.
2. **Implication for Lexical Interventions**: Because $\Delta F_1$ near 100% is `+14.69` pp, scaling silver data volume yields rapidly diminishing returns without structural architectural or representation improvements (e.g. attention attribution, subword boundary encoding).
