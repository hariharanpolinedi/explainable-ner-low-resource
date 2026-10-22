# Phase 5 Shortlist: Accepted vs. Rejected Interventions

> **Phase 5 Deliverable — Formal Decision Ledger**  
> Enacts the diagnostic principle: only interventions with genuine, evidenced improvement are accepted.
> Combines strictly in Phase 6; each candidate here was tested strictly in isolation.

---

## 1. Summary of Accepted Interventions (Shortlist for Phase 6)

| Candidate ID | Intervention Name | Language | Category | Primary Target Moved | Dev F1 Delta | Stress F1 Delta | Key Rationale |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| *None* | No candidates met acceptance threshold | - | - | - | - | - | Documented as critical limitation. |

---

## 2. Ledger of Rejected Interventions (Preserved with Full Negative Findings)

| Candidate ID | Intervention Name | Language | Category | Intended Metric | Observed Outcome | Rejection Rationale |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| `sinhala_conjunct_normalization` | **Sinhala Conjunct Normalization & Robustness** | Sinhala | Orthography & Unicode Preprocessing | Stress-Set Collapse Rate reduction (-15 to -30 pp) and Stress Recall improvement. | Dev: -11.78 pp, Stress: -0.46 pp | Insufficient collapse reduction (-6.67 pp). |
| `konkani_middle_weighting` | **Konkani Middle-Entity Class-Weighted Loss** | Konkani | Objective / Loss Formulation | Middle-Entity Recall (+10 to +20 pp) and Stress F1 improvement. | Dev: -1.87 pp, Stress: +0.00 pp | Middle-entity recall gain (+0.00 pp) insufficient or caused unacceptable dev regression (-1.87 pp). |
| `assamese_inflectional_augmentation` | **Assamese Targeted Inflectional Augmentation** | Assamese | Data Augmentation / Morphological Coverage | LOC Recall on Dev (+5 to +10 pp) and Stress F1 (+5 to +15 pp). | Dev: -0.36 pp, Stress: +0.00 pp | Stress F1 delta (+0.00 pp) did not meet acceptance bar. |
| `konkani_inflectional_loc` | **Konkani Locative Case-Suffix Augmentation** | Konkani | Data Augmentation / Morphological Coverage | LOC Recall on Dev (+10 to +15 pp) and LOC collapse reduction. | Dev: +1.42 pp, Stress: -0.19 pp | LOC F1 delta (+1.94 pp) insufficient to justify intervention. |
| `sinhala_loc_precision_filtering` | **Sinhala Silver Toponymic Precision Filtering** | Sinhala | Weak Supervision Data Hygiene | LOC Precision on Dev (+5 to +15 pp) without recall regression. | Dev: +4.13 pp, Stress: +1.61 pp | LOC precision delta (-4.28 pp) insufficient or caused unacceptable tradeoff. |

---

## 3. Methodological Integrity & Phase Boundary Confirmation

1. **Non-Circularity**: Every candidate was evaluated exclusively on locked benchmarks (`dev.conll` and `long_multientity_eval_v2.conll`), never on training data.
2. **Divergence Gated**: All models confirmed genuine weight updates (>= 1.0% change vs baseline).
3. **No Premature Combinations**: Interventions were tested in complete isolation. Combining accepted candidates is strictly reserved for Phase 6.
