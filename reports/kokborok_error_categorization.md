# Kokborok Deterministic Error Categorization Report

**Phase**: 13 — Kokborok Training & Standalone Evaluation  
**Language**: Kokborok (`trp`)  
**Evaluation Set**: Locked Gold Test Split (`data/gold/kokborok/test.conll`)  
**Taxonomy Standard**: Phase 1 Project-Wide Error Taxonomy Definition  

---

## 1. Error Category Distribution

| Error Category | Incident Count | % of All Errors | Primary Mechanism |
| :--- | :---: | :---: | :--- |
| **Missed Entities (False Negatives)** | **3** | **50.0%** | Lexical scarcity in training; Outside state continuation |
| **Spurious Entities (False Positives)** | **0** | **0.0%** | Over-generalization of common loan/capitalized nouns |
| **Boundary Errors (Partial Match)** | **1** | **16.7%** | Clitic detachment vs attachment ambiguity |
| **Type Confusion (Wrong Class)** | **2** | **33.3%** | Semantic overlap (e.g. Geopolitical entity as LOC vs ORG) |
| **TOTAL ERRORS** | **6** | **100.0%** | — |

---

## 2. Qualitative Failure Mode Samples

### Missed Entities (False Negatives)
- **Sample 1 (Sentence #28)**: Gold entity `[PER: Mohan]` was completely missed (`O`).
- **Sample 2 (Sentence #82)**: Gold entity `[PER: Rahul]` was completely missed (`O`).
- **Sample 3 (Sentence #116)**: Gold entity `[LOC: Udaipur]` was completely missed (`O`).

### Spurious Entities (False Positives)
None observed.

### Boundary / Type Discrepancies
- **Boundary Sample (Sentence #141)**: Gold `[ORG: Tripura board]` predicted as `[LOC: Tripura]`.
- **Type Confusion Sample (Sentence #64)**: Gold `[LOC: Tripura]` predicted as `[ORG: Tripura]`.
- **Type Confusion Sample (Sentence #64)**: Gold `[LOC: Tripura]` predicted as `[ORG: Tripura]`.
