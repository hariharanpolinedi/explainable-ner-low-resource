# Data Sufficiency Learning-Curve Report: Konkani

> **Phase 3 Deliverable — Controlled Empirical Volume Diagnosis**  
> Evaluates models trained from scratch on 25%, 50%, 75%, and 100% slices of the training pool (Gold + Silver = 8,050 sentences).
> Every retrained checkpoint is independently verified via the Checkpoint Divergence Gate.

---

## 1. Learning Curve Trajectory Matrix

| Data Fraction | Training Sentences | Best Epoch | Gold Dev Entity F1 | Entity Precision | Entity Recall | Token Accuracy | Divergence Gate Status | % Params Changed | Checkpoint Path |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **25%** | 2,012 | 5 | **27.84%** | 39.13% | 21.60% | 94.89% | **PASSED** | 16.88% | `checkpoints\learning_curve_konkani_25pct.pt` |
| **50%** | 4,025 | 5 | **51.72%** | 56.07% | 48.00% | 95.93% | **PASSED** | 20.53% | `checkpoints\learning_curve_konkani_50pct.pt` |
| **75%** | 6,037 | 5 | **63.07%** | 65.52% | 60.80% | 96.82% | **PASSED** | 23.27% | `checkpoints\learning_curve_konkani_75pct.pt` |
| **100%** | 8,050 | 5 | **70.16%** | 70.73% | 69.60% | 97.26% | **PASSED** | 25.46% | `checkpoints\learning_curve_konkani_100pct.pt` |

---

## 2. Per-Type Metric Progression

| Entity Type | 25% F1 (P / R) | 50% F1 (P / R) | 75% F1 (P / R) | 100% F1 (P / R) | Delta (75% -> 100%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PER** | 0.00% (0.0/0.0) | 0.00% (0.0/0.0) | 33.33% (100.0/20.0) | 51.85% (100.0/35.0) | +18.52 pp |
| **LOC** | 41.76% (31.7/61.3) | 48.00% (34.8/77.4) | 50.53% (37.5/77.4) | 56.52% (42.6/83.9) | +5.99 pp |
| **ORG** | 34.78% (88.9/21.6) | 65.52% (90.5/51.4) | 76.67% (100.0/62.2) | 88.24% (96.8/81.1) | +11.57 pp |
| **MISC** | 0.00% (0.0/0.0) | 62.96% (100.0/46.0) | 80.65% (100.0/67.6) | 78.69% (100.0/64.9) | -1.96 pp |

---

## 3. Curve Shape Characterization & Diagnosis

- **Slope between 75% and 100%**: $\Delta F_1 = +7.09$ pp
- **Slope per 1,000 sentences**: `+3.5221 pp/1k sentences`
- **Characterization**: **Steeply Rising (strong data-hungriness, highly sensitive to data volume)**

### Scientific Interpretation
1. **Saturation Diagnostic**: The learning curve trajectory confirms whether expanding raw training volume alone can resolve baseline NER errors.
2. **Implication for Lexical Interventions**: Because $\Delta F_1$ near 100% is `+7.09` pp, scaling silver data volume yields rapidly diminishing returns without structural architectural or representation improvements (e.g. attention attribution, subword boundary encoding).
