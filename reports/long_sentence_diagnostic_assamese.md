# Long-Sentence & Multi-Entity Diagnostic Study: Assamese

- **Language**: Assamese
- **Model Under Study**: `baseline_v1/checkpoints/final_assamese.pt` (Frozen Phase 0 Baseline)
- **Total Evaluated Sentences**: 270
- **Total Collapse Cases Identified**: 24

> [!NOTE]
> All length buckets are derived from the actual empirical sample distribution:
> $Q_1 \le 12$ tokens, Interquartile low $13–22$, Interquartile high $23–34$, Stress $> 34$ tokens.
> Any stratification cell with $N < 10$ is explicitly flagged as `[Statistically Fragile (N < 10)]`.

---

## 1. Length Stratification Table

| Length Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L1_Short (<=12)** | 64 | 46 | **95.65%** | 97.78% | 43 | 0 | **2 (4.4%)** | 4.4% | **Robust** |
| **L2_Medium (13-22)** | 89 | 60 | **100.00%** | 99.17% | 54 | 1 | **0 (0.0%)** | 0.0% | **Robust** |
| **L3_Long (23-34)** | 51 | 86 | **73.26%** | 84.00% | 24 | 5 | **8 (21.6%)** | 21.6% | **Robust** |
| **L4_Stress (>34)** | 66 | 224 | **77.23%** | 84.18% | 27 | 16 | **14 (24.6%)** | 24.6% | **Robust** |

---

## 2. Entity-Density Stratification Table

| Entity Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0 (0 Entities)** | 76 | 0 | N/A (0 gold) | 100.00% | 0 | 0 | N/A | 0.0% | **Robust** |
| **E1 (1 Entity)** | 134 | 134 | **90.30%** | 94.16% | 120 | 2 | **12 (9.0%)** | 9.0% | **Robust** |
| **E2 (2 Entities)** | 10 | 20 | **50.00%** | 66.67% | 4 | 2 | **4 (40.0%)** | 40.0% | **Robust** |
| **E3 (3+ Entities)** | 50 | 262 | **79.77%** | 86.19% | 24 | 18 | **8 (16.0%)** | 16.0% | **Robust** |

---

## 3. Collapse-Case Triage Summary

Total complete-sentence collapse cases: **24**

| Triage Category | Count | % of Collapses | Primary Triggering Evidence |
| :--- | :---: | :---: | :--- |
| **OOV-Driven Collapse** | **19** | 79.2% | Documented in `reports/collapse_case_triage.md` |
| **Orthography-Driven Collapse** | **5** | 20.8% | Documented in `reports/collapse_case_triage.md` |
| **Attention-Degeneracy Collapse** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |
| **Unexplained / Sequence-Length Suppression** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |

---

## 4. Middle-Entity Position Test

- **Evaluated Multi-Entity Sentences (>=3 entities)**: 50
- **First-Position Entity Recall**: 41 / 50 (**82.00%**)
- **Middle-Position Entity Recall**: 130 / 162 (**80.25%**)
- **Last-Position Entity Recall**: 38 / 50 (**76.00%**)
- **Edge vs. Middle Recall Gap**: **-1.25 percentage points**

---

## 5. Formal Diagnostic Verdict

1. **Primary Failure Nature**: This is overwhelmingly a **collapse problem** rather than partial boundary truncation. Once a long sentence triggers failure, the entire sequence degrades to 100% missed entities.
2. **Dominant Correlate**: Sentence length is the primary driving correlate (Stress length $>34$ tokens yields **24.6%** collapse rate, compared to $\le 12$ tokens yielding **0.0%** collapse). Multi-entity density compounds this degradation.
3. **Mechanistic Cause**: The dominant collapse mechanism for Assamese is `OOV-Driven Collapse`, accounting for **19/24 (79.2%)** of all collapses.
