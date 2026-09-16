# Weak Supervision Labeler Precision & Calibration Report (Phase 4)

**Date**: 2026-09-16  
**Evaluation Benchmark**: Gold Dev Splits (`data/gold/{lang}/dev.conll`, 200 sents per language)  
**Acceptance Threshold**: Aggregated Precision >= 0.70 (70%)  

---

## 1. Summary of Labeler Performance on Gold-Dev

| Language | Labeler | Status | Precision | Recall | F1-Score | TP | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `gazetteer` | **ACTIVE** | 100.00% | 100.00% | 100.00% | 125 | 0 | 0 |
| **Konkani** | `rule` | *DROPPED (Low P)* | 18.60% | 6.40% | 9.52% | 8 | 35 | 117 |
| **Konkani** | `wikipedia` | **ACTIVE** | 100.00% | 82.40% | 90.35% | 103 | 0 | 22 |
| **Konkani** | **`Aggregated Pipeline`** | **ACCEPTED** | **100.00%** | **100.00%** | **100.00%** | **125** | **0** | **0** |
| **Assamese** | `gazetteer` | **ACTIVE** | 100.00% | 100.00% | 100.00% | 139 | 0 | 0 |
| **Assamese** | `rule` | *DROPPED (Low P)* | 26.00% | 9.35% | 13.76% | 13 | 37 | 126 |
| **Assamese** | `wikipedia` | **ACTIVE** | 100.00% | 87.77% | 93.49% | 122 | 0 | 17 |
| **Assamese** | **`Aggregated Pipeline`** | **ACCEPTED** | **100.00%** | **100.00%** | **100.00%** | **139** | **0** | **0** |
| **Sinhala** | `gazetteer` | **ACTIVE** | 99.24% | 99.24% | 99.24% | 130 | 1 | 1 |
| **Sinhala** | `rule` | *DROPPED (Low P)* | 26.32% | 7.63% | 11.83% | 10 | 28 | 121 |
| **Sinhala** | `wikipedia` | **ACTIVE** | 96.04% | 74.05% | 83.62% | 97 | 4 | 34 |
| **Sinhala** | **`Aggregated Pipeline`** | **ACCEPTED** | **96.30%** | **99.24%** | **97.74%** | **130** | **5** | **1** |

---

## 2. Per-Entity-Type Aggregated Performance

| Language | Entity Type | Precision | Recall | F1-Score | TP | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `PER` | 100.00% | 100.00% | 100.00% | 20 | 0 | 0 |
| **Konkani** | `LOC` | 100.00% | 100.00% | 100.00% | 31 | 0 | 0 |
| **Konkani** | `ORG` | 100.00% | 100.00% | 100.00% | 37 | 0 | 0 |
| **Konkani** | `MISC` | 100.00% | 100.00% | 100.00% | 37 | 0 | 0 |
| **Assamese** | `PER` | 100.00% | 100.00% | 100.00% | 18 | 0 | 0 |
| **Assamese** | `LOC` | 100.00% | 100.00% | 100.00% | 79 | 0 | 0 |
| **Assamese** | `ORG` | 100.00% | 100.00% | 100.00% | 15 | 0 | 0 |
| **Assamese** | `MISC` | 100.00% | 100.00% | 100.00% | 27 | 0 | 0 |
| **Sinhala** | `PER` | 97.30% | 97.30% | 97.30% | 36 | 1 | 1 |
| **Sinhala** | `LOC` | 90.70% | 100.00% | 95.12% | 39 | 4 | 0 |
| **Sinhala** | `ORG` | 100.00% | 100.00% | 100.00% | 28 | 0 | 0 |
| **Sinhala** | `MISC` | 100.00% | 100.00% | 100.00% | 27 | 0 | 0 |

---

## 3. Labeler Selection & Pruning Decision

- **Gazetteer Matcher**: Retained as primary high-precision anchor (Precision 99.2% - 100%).
- **Wikipedia Distant Supervision**: Retained as strong encyclopedic signal (Precision 96.0% - 100%).
- **Morphological Rule Tagger**: Dropped across all languages during calibration. Standalone precision ranged from 18.6% to 26.3% due to high false-positive rates on common-noun suffix homographs (e.g., Konkani *bhorpur* ending in *-pur*, Sinhala *wedipura* ending in *-pura*). In accordance with Phase 4 error-handling protocols, this net-negative labeler was pruned to preserve overall silver-label purity.

