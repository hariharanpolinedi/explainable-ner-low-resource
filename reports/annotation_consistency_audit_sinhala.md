# Annotation Consistency Audit: Sinhala

> **Phase 3 Deliverable — Seed Batch (Sentences 1–600) vs Expansion Batch (Sentences 601–850)**  
> Audits structural annotation drift, guideline adherence, entity density shifts, and label contradictions.

---

## 1. Batch Comparison Summary

| Metric | Seed Batch (1–600) | Expansion Batch (601–850) | Drift / Ratio |
| :--- | :---: | :---: | :---: |
| **Sentence Count** | 600 | 250 | - |
| **Token Count** | 11,626 | 4,915 | - |
| **Total Entities** | 393 | 248 | - |
| **Entities / Sentence** | **0.655** | **0.992** | **1.51x** |
| **Entities / 100 Tokens** | 3.38 | 5.05 | - |
| **Single-Token Spans** | 51.9% | 44.4% | - |

---

## 2. Entity Type Proportions

| Entity Class | Seed Batch % (Count) | Expansion Batch % (Count) | Proportion Shift |
| :--- | :---: | :---: | :---: |
| **PER** | 28.8% (113) | 26.2% (65) | -2.6 pp |
| **LOC** | 29.5% (116) | 43.5% (108) | +14.0 pp |
| **ORG** | 21.1% (83) | 14.5% (36) | -6.6 pp |
| **MISC** | 20.6% (81) | 15.7% (39) | -4.9 pp |

---

## 3. Guideline Compliance & Concrete Discrepancies

- **Cross-Batch Contradictions Identified**: `0` surface forms
- **Guideline Violations Across Gold Train**: `0`

### Concrete Discrepancy Evidence

No identical surface forms received conflicting entity types across batches.
