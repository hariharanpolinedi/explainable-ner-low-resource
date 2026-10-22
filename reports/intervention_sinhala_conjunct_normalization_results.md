# Isolated Intervention Report: Sinhala Conjunct Normalization & Robustness

> **Phase 5 Deliverable — Isolated Intervention Evaluation**  
> Candidate ID: `sinhala_conjunct_normalization`  
> Target Language: **Sinhala**  
> Category: **Orthography & Unicode Preprocessing**  
> Final Verdict: **REJECT**  

---

## 1. Evidenced Root Cause & Mechanism
- **Phases 1–4 Root Cause Traced**: Phase 2 Triage: 27.8% of Sinhala collapses caused by decomposed or missing ZWJ in conjuncts.
- **Predicted Metric to Move**: Stress-Set Collapse Rate reduction (-15 to -30 pp) and Stress Recall improvement.
- **Intervention Details**: Tested strictly in isolation on top of the Phase 0 baseline configuration.
- **Checkpoint Divergence Gate**: `22.43%` parameters changed vs baseline ($L_2 = 8.3260$) — **PASSED**.
- **Saved Checkpoint**: `checkpoints\intervention_sinhala_conjunct_normalization_sinhala.pt`

---

## 2. Before vs. After Comparative Matrix

| Evaluation Dimension | Frozen Baseline (Reference) | Candidate Intervention (`sinhala_conjunct_normalization`) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Gold Dev Entity F1** | **93.98%** | **82.20%** | **-11.78 pp** |
| **Gold Dev Precision** | 99.15% | 92.38% | -6.77 pp |
| **Gold Dev Recall** | 89.31% | 74.05% | -15.26 pp |
| **Gold Dev Token Accuracy** | 99.38% | 98.19% | -1.19 pp |
| **Stress-Set Entity F1** | **36.36%** | **35.90%** | **-0.46 pp** |
| **Stress-Set Recall** | 26.42% | 26.42% | +0.00 pp |
| **Middle-Entity Recall (Lists >= 3)** | **0.00%** | **16.67%** | **+16.67 pp** |
| **Stress Collapse Rate** | **60.00%** | **53.33%** | **-6.67 pp** |

---

## 3. Per-Entity-Type Breakdown & Tradeoff Audit

| Entity Type | Baseline Dev F1 (P / R) | Candidate Dev F1 (P / R) | Delta Type F1 | Tradeoff Status |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 91.18% (100.0/83.8) | 80.60% (90.0/73.0) | -10.58 pp | TRADE-OFF (-10.0 pp Prec) |
| **LOC** | 91.89% (97.1/87.2) | 86.11% (93.9/79.5) | -5.78 pp | TRADE-OFF (-3.2 pp Prec) |
| **ORG** | 96.30% (100.0/92.9) | 78.43% (87.0/71.4) | -17.87 pp | TRADE-OFF (-13.0 pp Prec) |
| **MISC** | 98.11% (100.0/96.3) | 82.61% (100.0/70.4) | -15.50 pp | HEALTHY |

---

## 4. Final Verdict & Scientific Rationale

- **Formal Verdict**: **REJECT**
- **Decision Rationale**: Insufficient collapse reduction (-6.67 pp).
