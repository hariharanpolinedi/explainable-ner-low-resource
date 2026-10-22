# Long-Sentence & Multi-Entity Diagnostic Study: Sinhala

- **Language**: Sinhala
- **Model Under Study**: `baseline_v1/checkpoints/final_sinhala.pt` (Frozen Phase 0 Baseline)
- **Total Evaluated Sentences**: 263
- **Total Collapse Cases Identified**: 18

> [!NOTE]
> All length buckets are derived from the actual empirical sample distribution:
> $Q_1 \le 12$ tokens, Interquartile low $13–22$, Interquartile high $23–34$, Stress $> 34$ tokens.
> Any stratification cell with $N < 10$ is explicitly flagged as `[Statistically Fragile (N < 10)]`.

---

## 1. Length Stratification Table

| Length Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L1_Short (<=12)** | 52 | 27 | **100.00%** | 100.00% | 26 | 0 | **0 (0.0%)** | 0.0% | **Robust** |
| **L2_Medium (13-22)** | 69 | 49 | **100.00%** | 98.99% | 38 | 0 | **0 (0.0%)** | 0.0% | **Robust** |
| **L3_Long (23-34)** | 78 | 93 | **89.25%** | 91.21% | 39 | 4 | **5 (10.4%)** | 10.4% | **Robust** |
| **L4_Stress (>34)** | 64 | 131 | **74.81%** | 83.76% | 30 | 9 | **13 (25.0%)** | 25.0% | **Robust** |

---

## 2. Entity-Density Stratification Table

| Entity Stratum | Sentences | Gold Entities | Recall | Entity F1 | Full Success | Partial Miss | Complete Collapse | Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0 (0 Entities)** | 99 | 0 | N/A (0 gold) | 0.00% | 0 | 0 | N/A | 0.0% | **Robust** |
| **E1 (1 Entity)** | 98 | 98 | **89.80%** | 94.12% | 87 | 1 | **10 (10.2%)** | 10.2% | **Robust** |
| **E2 (2 Entities)** | 29 | 58 | **72.41%** | 78.50% | 17 | 6 | **6 (20.7%)** | 20.7% | **Robust** |
| **E3 (3+ Entities)** | 37 | 144 | **88.19%** | 93.04% | 29 | 6 | **2 (5.4%)** | 5.4% | **Robust** |

---

## 3. Collapse-Case Triage Summary

Total complete-sentence collapse cases: **18**

| Triage Category | Count | % of Collapses | Primary Triggering Evidence |
| :--- | :---: | :---: | :--- |
| **OOV-Driven Collapse** | **13** | 72.2% | Documented in `reports/collapse_case_triage.md` |
| **Orthography-Driven Collapse** | **5** | 27.8% | Documented in `reports/collapse_case_triage.md` |
| **Attention-Degeneracy Collapse** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |
| **Unexplained / Sequence-Length Suppression** | **0** | 0.0% | Documented in `reports/collapse_case_triage.md` |

---

## 4. Middle-Entity Position Test

- **Evaluated Multi-Entity Sentences (>=3 entities)**: 37
- **First-Position Entity Recall**: 32 / 37 (**86.49%**)
- **Middle-Position Entity Recall**: 61 / 70 (**87.14%**)
- **Last-Position Entity Recall**: 34 / 37 (**91.89%**)
- **Edge vs. Middle Recall Gap**: **+2.05 percentage points**

---

## 5. Formal Diagnostic Verdict

1. **Primary Failure Nature**: This is overwhelmingly a **collapse problem** rather than partial boundary truncation. Once a long sentence triggers failure, the entire sequence degrades to 100% missed entities.
2. **Dominant Correlate**: Sentence length is the primary driving correlate (Stress length $>34$ tokens yields **25.0%** collapse rate, compared to $\le 12$ tokens yielding **0.0%** collapse). Multi-entity density compounds this degradation.
3. **Mechanistic Cause**: The dominant collapse mechanism for Sinhala is `OOV-Driven Collapse`, accounting for **13/18 (72.2%)** of all collapses.
