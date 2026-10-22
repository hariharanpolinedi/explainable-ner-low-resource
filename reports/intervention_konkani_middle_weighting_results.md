# Isolated Intervention Report: Konkani Middle-Entity Class-Weighted Loss

> **Phase 5 Deliverable — Isolated Intervention Evaluation**  
> Candidate ID: `konkani_middle_weighting`  
> Target Language: **Konkani**  
> Category: **Objective / Loss Formulation**  
> Final Verdict: **REJECT**  

---

## 1. Evidenced Root Cause & Mechanism
- **Phases 1–4 Root Cause Traced**: Phase 2 Position Test: Middle-entity recall collapsed to 33.93% (+30.07 pp gap vs edges) due to CRF 'O'-emission bias.
- **Predicted Metric to Move**: Middle-Entity Recall (+10 to +20 pp) and Stress F1 improvement.
- **Intervention Details**: Tested strictly in isolation on top of the Phase 0 baseline configuration.
- **Checkpoint Divergence Gate**: `27.01%` parameters changed vs baseline ($L_2 = 9.3570$) — **PASSED**.
- **Saved Checkpoint**: `checkpoints\intervention_konkani_middle_weighting_konkani.pt`

---

## 2. Before vs. After Comparative Matrix

| Evaluation Dimension | Frozen Baseline (Reference) | Candidate Intervention (`konkani_middle_weighting`) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Gold Dev Entity F1** | **92.70%** | **90.83%** | **-1.87 pp** |
| **Gold Dev Precision** | 100.00% | 94.78% | -5.22 pp |
| **Gold Dev Recall** | 86.40% | 87.20% | +0.80 pp |
| **Gold Dev Token Accuracy** | 99.15% | 99.18% | +0.03 pp |
| **Stress-Set Entity F1** | **17.39%** | **17.39%** | **+0.00 pp** |
| **Stress-Set Recall** | 9.64% | 9.64% | +0.00 pp |
| **Middle-Entity Recall (Lists >= 3)** | **0.00%** | **0.00%** | **+0.00 pp** |
| **Stress Collapse Rate** | **75.76%** | **75.76%** | **+0.00 pp** |

---

## 3. Per-Entity-Type Breakdown & Tradeoff Audit

| Entity Type | Baseline Dev F1 (P / R) | Candidate Dev F1 (P / R) | Delta Type F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 88.89% (100.0/80.0) | 91.89% (100.0/85.0) | +3.00 pp | HEALTHY |
| **LOC** | 89.29% (100.0/80.7) | 84.75% (89.3/80.7) | -4.54 pp | TRADE-OFF (-10.7 pp Prec) |
| **ORG** | 91.18% (100.0/83.8) | 88.57% (93.9/83.8) | -2.61 pp | TRADE-OFF (-6.1 pp Prec) |
| **MISC** | 98.63% (100.0/97.3) | 97.30% (97.3/97.3) | -1.33 pp | HEALTHY |

---

## 4. Final Verdict & Scientific Rationale

- **Formal Verdict**: **REJECT**
- **Decision Rationale**: Middle-entity recall gain (+0.00 pp) insufficient or caused unacceptable dev regression (-1.87 pp).
