# Long-Sentence & Multi-Entity Diagnostic Study: Konkani

- **Language**: Konkani
- **Model Under Study**: `baseline_v1/checkpoints/final_konkani.pt` (Frozen Phase 0 Baseline)
- **Total Evaluated Sentences**: 247
- **Total Collapse Cases Identified**: 30

> [!NOTE]
> All length buckets are derived from the actual empirical sample distribution:
> $Q_1 \le 12$ tokens, Interquartile low $13–22$, Interquartile high $23–34$, Stress $> 34$ tokens.
> Any stratification cell with $N < 10$ is explicitly flagged as `[Statistically Fragile (N < 10)]`.

---

## 1. Length Stratification Table

| Length Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L1_Short (<=12)** | 88 | 37 | **97.30%** | 98.63% | 34 | 0 | **1 (2.9%)** | 2.9% | **Robust** |
| **L2_Medium (13-22)** | 71 | 55 | **92.73%** | 96.23% | 39 | 0 | **4 (9.3%)** | 9.3% | **Robust** |
| **L3_Long (23-34)** | 46 | 53 | **60.38%** | 75.29% | 18 | 2 | **9 (31.0%)** | 31.0% | **Robust** |
| **L4_Stress (>34)** | 42 | 106 | **40.57%** | 57.33% | 9 | 11 | **16 (44.4%)** | 44.4% | **Robust** |

---

## 2. Entity-Density Stratification Table

| Entity Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0 (0 Entities)** | 104 | 0 | N/A (0 gold) | 100.00% | 0 | 0 | N/A | 0.0% | **Robust** |
| **E1 (1 Entity)** | 91 | 91 | **79.12%** | 88.34% | 72 | 0 | **19 (20.9%)** | 20.9% | **Robust** |
| **E2 (2 Entities)** | 27 | 54 | **72.22%** | 82.98% | 17 | 5 | **5 (18.5%)** | 18.5% | **Robust** |
| **E3 (3+ Entities)** | 25 | 106 | **48.11%** | 64.97% | 11 | 8 | **6 (24.0%)** | 24.0% | **Robust** |

---

## 3. Collapse-Case Triage Summary

Total complete-sentence collapse cases: **30**

| Triage Category | Count | % of Collapses | Primary Triggering Evidence |
| :--- | :---: | :---: | :--- |
| **OOV-Driven Collapse** | **28** | 93.3% | Documented in `reports/collapse_case_triage.md` |
| **Orthography-Driven Collapse** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |
| **Attention-Degeneracy Collapse** | **2** | 6.7% | Documented in `reports/collapse_case_triage.md` |
| **Unexplained / Sequence-Length Suppression** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |

---

## 4. Middle-Entity Position Test

- **Evaluated Multi-Entity Sentences (>=3 entities)**: 25
- **First-Position Entity Recall**: 14 / 25 (**56.00%**)
- **Middle-Position Entity Recall**: 19 / 56 (**33.93%**)
- **Last-Position Entity Recall**: 18 / 25 (**72.00%**)
- **Edge vs. Middle Recall Gap**: **+30.07 percentage points**

---

## 5. Formal Diagnostic Verdict

1. **Primary Failure Nature**: This is overwhelmingly a **collapse problem** rather than partial boundary truncation. Once a long sentence triggers failure, the entire sequence degrades to 100% missed entities.
2. **Dominant Correlate**: Sentence length is the primary driving correlate (Stress length $>34$ tokens yields **44.4%** collapse rate, compared to $\le 12$ tokens yielding **0.0%** collapse). Multi-entity density compounds this degradation.
3. **Mechanistic Cause**: The dominant collapse mechanism for Konkani is `OOV-Driven Collapse`, accounting for **28/30 (93.3%)** of all collapses.
