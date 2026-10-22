# Isolated Intervention Report: Sinhala Silver Toponymic Precision Filtering

> **Phase 5 Deliverable — Isolated Intervention Evaluation**  
> Candidate ID: `sinhala_loc_precision_filtering`  
> Target Language: **Sinhala**  
> Category: **Weak Supervision Data Hygiene**  
> Final Verdict: **REJECT**  

---

## 1. Evidenced Root Cause & Mechanism
- **Phases 1–4 Root Cause Traced**: Phase 3 Silver Re-Audit: LOC precision on independent gold train is only 57.81% due to toponymic heuristic over-generation on common spatial terms.
- **Predicted Metric to Move**: LOC Precision on Dev (+5 to +15 pp) without recall regression.
- **Intervention Details**: Tested strictly in isolation on top of the Phase 0 baseline configuration.
- **Checkpoint Divergence Gate**: `24.18%` parameters changed vs baseline ($L_2 = 4.8735$) — **PASSED**.
- **Saved Checkpoint**: `checkpoints\intervention_sinhala_loc_precision_filtering_sinhala.pt`

---

## 2. Before vs. After Comparative Matrix

| Evaluation Dimension | Frozen Baseline (Reference) | Candidate Intervention (`sinhala_loc_precision_filtering`) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Gold Dev Entity F1** | **93.98%** | **98.11%** | **+4.13 pp** |
| **Gold Dev Precision** | 99.15% | 97.01% | -2.14 pp |
| **Gold Dev Recall** | 89.31% | 99.24% | +9.93 pp |
| **Gold Dev Token Accuracy** | 99.38% | 99.75% | +0.37 pp |
| **Stress-Set Entity F1** | **36.36%** | **37.97%** | **+1.61 pp** |
| **Stress-Set Recall** | 26.42% | 28.30% | +1.88 pp |
| **Middle-Entity Recall (Lists >= 3)** | **0.00%** | **16.67%** | **+16.67 pp** |
| **Stress Collapse Rate** | **60.00%** | **60.00%** | **+0.00 pp** |

---

## 3. Per-Entity-Type Breakdown & Tradeoff Audit

| Entity Type | Baseline Dev F1 (P / R) | Candidate Dev F1 (P / R) | Delta Type F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 91.18% (100.0/83.8) | 100.00% (100.0/100.0) | +8.82 pp | HEALTHY |
| **LOC** | 91.89% (97.1/87.2) | 96.30% (92.9/100.0) | +4.41 pp | TRADE-OFF (-4.3 pp Prec) |
| **ORG** | 96.30% (100.0/92.9) | 98.25% (96.5/100.0) | +1.95 pp | TRADE-OFF (-3.5 pp Prec) |
| **MISC** | 98.11% (100.0/96.3) | 98.11% (100.0/96.3) | +0.00 pp | HEALTHY |

---

## 4. Final Verdict & Scientific Rationale

- **Formal Verdict**: **REJECT**
- **Decision Rationale**: LOC precision delta (-4.28 pp) insufficient or caused unacceptable tradeoff.
