# Annotation Consistency Audit: Assamese

> **Phase 3 Deliverable — Seed Batch (Sentences 1–600) vs Expansion Batch (Sentences 601–850)**  
> Audits structural annotation drift, guideline adherence, entity density shifts, and label contradictions.

---

## 1. Batch Comparison Summary

| Metric | Seed Batch (1–600) | Expansion Batch (601–850) | Drift / Ratio |
| :--- | :---: | :---: | :---: |
| **Sentence Count** | 600 | 250 | - |
| **Token Count** | 10,134 | 4,515 | - |
| **Total Entities** | 416 | 263 | - |
| **Entities / Sentence** | **0.693** | **1.052** | **1.52x** |
| **Entities / 100 Tokens** | 4.1 | 5.83 | - |
| **Single-Token Spans** | 74.0% | 75.3% | - |

---

## 2. Entity Type Proportions

| Entity Class | Seed Batch % (Count) | Expansion Batch % (Count) | Proportion Shift |
| :--- | :---: | :---: | :---: |
| **PER** | 13.2% (55) | 17.9% (47) | +4.7 pp |
| **LOC** | 57.0% (237) | 60.1% (158) | +3.1 pp |
| **ORG** | 10.6% (44) | 8.7% (23) | -1.9 pp |
| **MISC** | 19.2% (80) | 13.3% (35) | -5.9 pp |

---

## 3. Guideline Compliance & Concrete Discrepancies

- **Cross-Batch Contradictions Identified**: `0` surface forms
- **Guideline Violations Across Gold Train**: `0`

### Concrete Discrepancy Evidence

No identical surface forms received conflicting entity types across batches.
