# Isolated Intervention Report: Konkani Locative Case-Suffix Augmentation

> **Phase 5 Deliverable — Isolated Intervention Evaluation**  
> Candidate ID: `konkani_inflectional_loc`  
> Target Language: **Konkani**  
> Category: **Data Augmentation / Morphological Coverage**  
> Final Verdict: **REJECT**  

---

## 1. Evidenced Root Cause & Mechanism
- **Phases 1–4 Root Cause Traced**: Phase 1 & 2 LOC Diagnostic: Inflected location forms like पणजींत and गोयाच्या suffered 50% F1 penalty and 33.3% recall.
- **Predicted Metric to Move**: LOC Recall on Dev (+10 to +15 pp) and LOC collapse reduction.
- **Intervention Details**: Tested strictly in isolation on top of the Phase 0 baseline configuration.
- **Checkpoint Divergence Gate**: `23.01%` parameters changed vs baseline ($L_2 = 4.3752$) — **PASSED**.
- **Saved Checkpoint**: `checkpoints\intervention_konkani_inflectional_loc_konkani.pt`

---

## 2. Before vs. After Comparative Matrix

| Evaluation Dimension | Frozen Baseline (Reference) | Candidate Intervention (`konkani_inflectional_loc`) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Gold Dev Entity F1** | **92.70%** | **94.12%** | **+1.42 pp** |
| **Gold Dev Precision** | 100.00% | 99.12% | -0.88 pp |
| **Gold Dev Recall** | 86.40% | 89.60% | +3.20 pp |
| **Gold Dev Token Accuracy** | 99.15% | 99.31% | +0.16 pp |
| **Stress-Set Entity F1** | **17.39%** | **17.20%** | **-0.19 pp** |
| **Stress-Set Recall** | 9.64% | 9.64% | +0.00 pp |
| **Middle-Entity Recall (Lists >= 3)** | **0.00%** | **0.00%** | **+0.00 pp** |
| **Stress Collapse Rate** | **75.76%** | **72.73%** | **-3.03 pp** |

---

## 3. Per-Entity-Type Breakdown & Tradeoff Audit

| Entity Type | Baseline Dev F1 (P / R) | Candidate Dev F1 (P / R) | Delta Type F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 88.89% (100.0/80.0) | 88.89% (100.0/80.0) | +0.00 pp | HEALTHY |
| **LOC** | 89.29% (100.0/80.7) | 91.23% (100.0/83.9) | +1.94 pp | HEALTHY |
| **ORG** | 91.18% (100.0/83.8) | 94.44% (97.1/91.9) | +3.26 pp | HEALTHY |
| **MISC** | 98.63% (100.0/97.3) | 98.63% (100.0/97.3) | +0.00 pp | HEALTHY |

---

## 4. Final Verdict & Scientific Rationale

- **Formal Verdict**: **REJECT**
- **Decision Rationale**: LOC F1 delta (+1.94 pp) insufficient to justify intervention.
