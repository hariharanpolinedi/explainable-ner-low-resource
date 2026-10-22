# Isolated Intervention Report: Assamese Targeted Inflectional Augmentation

> **Phase 5 Deliverable — Isolated Intervention Evaluation**  
> Candidate ID: `assamese_inflectional_augmentation`  
> Target Language: **Assamese**  
> Category: **Data Augmentation / Morphological Coverage**  
> Final Verdict: **REJECT**  

---

## 1. Evidenced Root Cause & Mechanism
- **Phases 1–4 Root Cause Traced**: Phase 1 & 2 OOV Triage (79.2% collapses) & Phase 3 frequency check: failure of high-frequency stems under novel case inflections.
- **Predicted Metric to Move**: LOC Recall on Dev (+5 to +10 pp) and Stress F1 (+5 to +15 pp).
- **Intervention Details**: Tested strictly in isolation on top of the Phase 0 baseline configuration.
- **Checkpoint Divergence Gate**: `23.69%` parameters changed vs baseline ($L_2 = 4.5130$) — **PASSED**.
- **Saved Checkpoint**: `checkpoints\intervention_assamese_inflectional_augmentation_assamese.pt`

---

## 2. Before vs. After Comparative Matrix

| Evaluation Dimension | Frozen Baseline (Reference) | Candidate Intervention (`assamese_inflectional_augmentation`) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Gold Dev Entity F1** | **98.54%** | **98.18%** | **-0.36 pp** |
| **Gold Dev Precision** | 100.00% | 99.26% | -0.74 pp |
| **Gold Dev Recall** | 97.12% | 97.12% | +0.00 pp |
| **Gold Dev Token Accuracy** | 99.86% | 99.78% | -0.08 pp |
| **Stress-Set Entity F1** | **22.99%** | **22.99%** | **+0.00 pp** |
| **Stress-Set Recall** | 13.16% | 13.16% | +0.00 pp |
| **Middle-Entity Recall (Lists >= 3)** | **7.41%** | **7.41%** | **+0.00 pp** |
| **Stress Collapse Rate** | **70.97%** | **70.97%** | **+0.00 pp** |

---

## 3. Per-Entity-Type Breakdown & Tradeoff Audit

| Entity Type | Baseline Dev F1 (P / R) | Candidate Dev F1 (P / R) | Delta Type F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 100.00% (100.0/100.0) | 100.00% (100.0/100.0) | +0.00 pp | HEALTHY |
| **LOC** | 98.72% (100.0/97.5) | 98.09% (98.7/97.5) | -0.63 pp | HEALTHY |
| **ORG** | 96.55% (100.0/93.3) | 92.86% (100.0/86.7) | -3.69 pp | HEALTHY |
| **MISC** | 98.11% (100.0/96.3) | 100.00% (100.0/100.0) | +1.89 pp | HEALTHY |

---

## 4. Final Verdict & Scientific Rationale

- **Formal Verdict**: **REJECT**
- **Decision Rationale**: Stress F1 delta (+0.00 pp) did not meet acceptance bar.
