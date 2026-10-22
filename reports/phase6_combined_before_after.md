# Phase 6: Combined Retraining & Controlled Before/After Comparison

> **Phase 6 Deliverable — Master Empirical Scorecard**  
> Direct, controlled comparison of Phase 0 Baseline vs. Best Single Isolated Intervention vs. Phase 6 Combined Model.
> Evaluated on standard Gold-Dev (200 sentences) and Long Multi-Entity Stress Benchmark (31–35 sentences).

---

## 1. Master Cross-Language Comparison Table

| Language | Model Specification | Gold Dev F1 (P / R) | $\Delta$ Dev F1 | Stress F1 | $\Delta$ Stress F1 | Middle Recall (Lists $\ge 3$) | Stress Collapse Rate | $\Delta$ Collapse | Current Best Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **Phase 0 Baseline (Anchor)** | **92.70%** (100.0/86.4) | *Ref* | **17.39%** | *Ref* | **0.00%** | **75.76%** | *Ref* | **CURRENT BEST** |
| | Best Isolated (`konkani_inflectional_loc`) | 94.12% (99.1/89.6) | +1.42 pp | 17.20% | -0.19 pp | 0.00% | 72.73% | -3.03 pp | Isolated |
| | Phase 6 Combined Model | 91.67% (95.7/88.0) | -1.03 pp | 17.39% | +0.00 pp | 0.00% | 75.76% | +0.00 pp | Retrained |
| **Assamese** | **Phase 0 Baseline (Anchor)** | **98.54%** (100.0/97.1) | *Ref* | **22.99%** | *Ref* | **7.41%** | **70.97%** | *Ref* | **CURRENT BEST** |
| | Best Isolated (`assamese_inflectional_augmentation`) | 98.18% (99.3/97.1) | -0.36 pp | 22.99% | +0.00 pp | 7.41% | 70.97% | +0.00 pp | Isolated |
| | Phase 6 Combined Model | 98.54% (100.0/97.1) | +0.00 pp | 22.99% | +0.00 pp | 7.41% | 70.97% | +0.00 pp | Retrained |
| **Sinhala** | **Phase 0 Baseline (Anchor)** | **93.98%** (99.2/89.3) | *Ref* | **36.36%** | *Ref* | **0.00%** | **60.00%** | *Ref* | Baseline |
| | Best Isolated (`sinhala_loc_precision_filtering`) | 98.11% (97.0/99.2) | +4.13 pp | 37.97% | +1.61 pp | 16.67% | 60.00% | +0.00 pp | Isolated |
| | Phase 6 Combined Model | 98.11% (97.0/99.2) | +4.13 pp | 38.46% | +2.10 pp | 16.67% | 60.00% | +0.00 pp | **CURRENT BEST** |

---

## 2. Per-Entity-Type Performance Breakdown

### Konkani

| Entity Type | Baseline Dev F1 (P / R) | Combined Dev F1 (P / R) | Delta F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 88.89% (100.0/80.0) | 88.89% (100.0/80.0) | +0.00 pp | HEALTHY |
| **LOC** | 89.29% (100.0/80.7) | 84.75% (89.3/80.7) | -4.54 pp | TRADE-OFF (-10.7 pp Prec) |
| **ORG** | 91.18% (100.0/83.8) | 92.96% (97.1/89.2) | +1.78 pp | HEALTHY |
| **MISC** | 98.63% (100.0/97.3) | 97.30% (97.3/97.3) | -1.33 pp | HEALTHY |

### Assamese

| Entity Type | Baseline Dev F1 (P / R) | Combined Dev F1 (P / R) | Delta F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 100.00% (100.0/100.0) | 100.00% (100.0/100.0) | +0.00 pp | HEALTHY |
| **LOC** | 98.72% (100.0/97.5) | 98.72% (100.0/97.5) | +0.00 pp | HEALTHY |
| **ORG** | 96.55% (100.0/93.3) | 92.86% (100.0/86.7) | -3.69 pp | HEALTHY |
| **MISC** | 98.11% (100.0/96.3) | 100.00% (100.0/100.0) | +1.89 pp | HEALTHY |

### Sinhala

| Entity Type | Baseline Dev F1 (P / R) | Combined Dev F1 (P / R) | Delta F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 91.18% (100.0/83.8) | 100.00% (100.0/100.0) | +8.82 pp | HEALTHY |
| **LOC** | 91.89% (97.1/87.2) | 97.50% (95.1/100.0) | +5.61 pp | HEALTHY |
| **ORG** | 96.30% (100.0/92.9) | 98.25% (96.5/100.0) | +1.95 pp | TRADE-OFF (-3.5 pp Prec) |
| **MISC** | 98.11% (100.0/96.3) | 96.30% (96.3/96.3) | -1.81 pp | TRADE-OFF (-3.7 pp Prec) |

---

## 3. Definitive Current Best Designation

- **Konkani**: Designated `checkpoints/improved_konkani_v2.pt` is based on **baseline_v1** (baseline_v1\checkpoints\final_konkani.pt).
  - *Empirical Rationale*: Neither combined (17.39% Stress F1, 75.76% Collapse) nor isolated interventions convincingly outperformed the Phase 0 baseline (17.39% Stress F1, 75.76% Collapse, 92.70% Dev F1). In strict adherence to failure-handling rules, Phase 0 baseline remains designated as Current Best.
- **Assamese**: Designated `checkpoints/improved_assamese_v2.pt` is based on **baseline_v1** (baseline_v1\checkpoints\final_assamese.pt).
  - *Empirical Rationale*: Neither combined (22.99% Stress F1, 70.97% Collapse) nor isolated interventions convincingly outperformed the Phase 0 baseline (22.99% Stress F1, 70.97% Collapse, 98.54% Dev F1). In strict adherence to failure-handling rules, Phase 0 baseline remains designated as Current Best.
- **Sinhala**: Designated `checkpoints/improved_sinhala_v2.pt` is based on **combined** (checkpoints\combined_sinhala_v2.pt).
  - *Empirical Rationale*: Combined model demonstrated superior stress-set F1 (38.46% vs 36.36%) without unacceptable dev regression (98.11%).
