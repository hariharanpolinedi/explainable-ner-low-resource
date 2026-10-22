# Annotation Consistency Audit: Konkani

> **Phase 3 Deliverable — Seed Batch (Sentences 1–600) vs Expansion Batch (Sentences 601–850)**  
> Audits structural annotation drift, guideline adherence, entity density shifts, and label contradictions.

---

## 1. Batch Comparison Summary

| Metric | Seed Batch (1–600) | Expansion Batch (601–850) | Drift / Ratio |
| :--- | :---: | :---: | :---: |
| **Sentence Count** | 600 | 250 | - |
| **Token Count** | 9,511 | 4,199 | - |
| **Total Entities** | 372 | 220 | - |
| **Entities / Sentence** | **0.62** | **0.88** | **1.42x** |
| **Entities / 100 Tokens** | 3.91 | 5.24 | - |
| **Single-Token Spans** | 64.8% | 53.6% | - |

---

## 2. Entity Type Proportions

| Entity Class | Seed Batch % (Count) | Expansion Batch % (Count) | Proportion Shift |
| :--- | :---: | :---: | :---: |
| **PER** | 16.4% (61) | 37.7% (83) | +21.3 pp |
| **LOC** | 24.5% (91) | 35.0% (77) | +10.5 pp |
| **ORG** | 29.6% (110) | 17.7% (39) | -11.9 pp |
| **MISC** | 29.6% (110) | 9.5% (21) | -20.1 pp |

---

## 3. Guideline Compliance & Concrete Discrepancies

- **Cross-Batch Contradictions Identified**: `0` surface forms
- **Guideline Violations Across Gold Train**: `0`

### Concrete Discrepancy Evidence

No identical surface forms received conflicting entity types across batches.
