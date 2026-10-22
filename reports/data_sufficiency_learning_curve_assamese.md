# Data Sufficiency Learning-Curve Report: Assamese

> **Phase 3 Deliverable — Controlled Empirical Volume Diagnosis**  
> Evaluates models trained from scratch on 25%, 50%, 75%, and 100% slices of the training pool (Gold + Silver = 8,050 sentences).
> Every retrained checkpoint is independently verified via the Checkpoint Divergence Gate.

---

## 1. Learning Curve Trajectory Matrix

| Data Fraction | Training Sentences | Best Epoch | Gold Dev Entity F1 | Entity Precision | Entity Recall | Token Accuracy | Divergence Gate Status | % Params Changed | Checkpoint Path |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **25%** | 2,012 | 5 | **56.43%** | 66.67% | 48.92% | 95.99% | **PASSED** | 17.45% | `checkpoints\learning_curve_assamese_25pct.pt` |
| **50%** | 4,025 | 5 | **64.91%** | 68.25% | 61.87% | 96.47% | **PASSED** | 21.24% | `checkpoints\learning_curve_assamese_50pct.pt` |
| **75%** | 6,037 | 5 | **72.97%** | 68.79% | 77.70% | 97.51% | **PASSED** | 23.81% | `checkpoints\learning_curve_assamese_75pct.pt` |
| **100%** | 8,050 | 5 | **74.67%** | 69.57% | 80.58% | 97.70% | **PASSED** | 25.90% | `checkpoints\learning_curve_assamese_100pct.pt` |

---

## 2. Per-Type Metric Progression

| Entity Type | 25% F1 (P / R) | 50% F1 (P / R) | 75% F1 (P / R) | 100% F1 (P / R) | Delta (75% -> 100%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PER** | 28.57% (100.0/16.7) | 64.29% (90.0/50.0) | 85.00% (77.3/94.4) | 97.14% (100.0/94.4) | +12.14 pp |
| **LOC** | 73.03% (65.7/82.3) | 71.74% (62.9/83.5) | 77.16% (64.4/96.2) | 75.86% (62.1/97.5) | -1.30 pp |
| **ORG** | 0.00% (0.0/0.0) | 0.00% (0.0/0.0) | 0.00% (0.0/0.0) | 23.53% (100.0/13.3) | +23.53 pp |
| **MISC** | 0.00% (0.0/0.0) | 57.89% (100.0/40.7) | 68.18% (88.2/55.6) | 71.11% (88.9/59.3) | +2.93 pp |

---

## 3. Curve Shape Characterization & Diagnosis

- **Slope between 75% and 100%**: $\Delta F_1 = +1.70$ pp
- **Slope per 1,000 sentences**: `+0.8445 pp/1k sentences`
- **Characterization**: **Modestly Climbing (slow progressive gain, mild volume sensitivity)**

### Scientific Interpretation
1. **Saturation Diagnostic**: The learning curve trajectory confirms whether expanding raw training volume alone can resolve baseline NER errors.
2. **Implication for Lexical Interventions**: Because $\Delta F_1$ near 100% is `+1.70` pp, scaling silver data volume yields rapidly diminishing returns without structural architectural or representation improvements (e.g. attention attribution, subword boundary encoding).
