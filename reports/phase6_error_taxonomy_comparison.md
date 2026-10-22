# Phase 6: Paired Error Taxonomy Comparison

> **Phase 6 Deliverable — Paired Error Category Census**  
> Evaluated on the identical Phase 1 diagnostic sample manifest (`reports/error_taxonomy_diagnostic_sample_manifest.md`).
> Direct category-by-category comparison between Phase 0 Frozen Baseline and Phase 6 Combined Model.

---

## 1. Paired Error Category Census across Diagnostic Population

### Konkani (Population: 247 sentences, 251 gold entities)

| Error Category | Baseline Errors | Combined Errors | Net Error Delta (Delta) | Baseline % Gold | Combined % Gold | Diagnostic Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complete Sentence Collapse** | 63 | 58 | -5 | 25.10% | 23.11% | **IMPROVED (-5)** |
| **Missed Entity** | 40 | 39 | -1 | 15.94% | 15.54% | **IMPROVED (-1)** |
| **False Positive** | 1 | 3 | +2 | 0.40% | 1.20% | **WORSENED (+2)** |
| **Incomplete Span** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **Overextended Span** | 0 | 3 | +3 | 0.00% | 1.20% | **WORSENED (+3)** |
| **Wrong Entity Type** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **B-/I- Transition Error** | 1 | 0 | -1 | 0.40% | 0.00% | **IMPROVED (-1)** |
| **O Inside Entity** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **TOTAL ERRORS** | **105** | **103** | **-2** | — | — | **NET IMPROVEMENT** |

### Assamese (Population: 270 sentences, 416 gold entities)

| Error Category | Baseline Errors | Combined Errors | Net Error Delta (Delta) | Baseline % Gold | Combined % Gold | Diagnostic Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complete Sentence Collapse** | 57 | 58 | +1 | 13.70% | 13.94% | **WORSENED (+1)** |
| **Missed Entity** | 16 | 17 | +1 | 3.85% | 4.09% | **WORSENED (+1)** |
| **False Positive** | 3 | 4 | +1 | 0.72% | 0.96% | **WORSENED (+1)** |
| **Incomplete Span** | 5 | 5 | +0 | 1.20% | 1.20% | UNCHANGED (0) |
| **Overextended Span** | 3 | 1 | -2 | 0.72% | 0.24% | **IMPROVED (-2)** |
| **Wrong Entity Type** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **B-/I- Transition Error** | 1 | 0 | -1 | 0.24% | 0.00% | **IMPROVED (-1)** |
| **O Inside Entity** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **TOTAL ERRORS** | **85** | **85** | **+0** | — | — | NEUTRAL |

### Sinhala (Population: 263 sentences, 300 gold entities)

| Error Category | Baseline Errors | Combined Errors | Net Error Delta (Delta) | Baseline % Gold | Combined % Gold | Diagnostic Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complete Sentence Collapse** | 37 | 28 | -9 | 12.33% | 9.33% | **IMPROVED (-9)** |
| **Missed Entity** | 21 | 14 | -7 | 7.00% | 4.67% | **IMPROVED (-7)** |
| **False Positive** | 11 | 14 | +3 | 3.67% | 4.67% | **WORSENED (+3)** |
| **Incomplete Span** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **Overextended Span** | 0 | 1 | +1 | 0.00% | 0.33% | **WORSENED (+1)** |
| **Wrong Entity Type** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **B-/I- Transition Error** | 1 | 0 | -1 | 0.33% | 0.00% | **IMPROVED (-1)** |
| **O Inside Entity** | 0 | 0 | +0 | 0.00% | 0.00% | UNCHANGED (0) |
| **TOTAL ERRORS** | **70** | **57** | **-13** | — | — | **NET IMPROVEMENT** |

