# Corpus Statistics Report (Phase 1)

**Date**: 2026-09-15  
**Scope**: Konkani, Assamese, Sinhala raw and filtered corpora  
**Constraint**: Zero pretrained models, tokenizers, or external NLP tools  

---

## 1. Executive Summary & Data Floor Verification

| Language | Pre-Filter Sentences | Filtered Sentences | Filtered Tokens | Unique Surface Vocab | 50k Floor Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | 112,235 | 30,980 | 2,752,735 | 18,218 | **SHORTFALL (~31k / 50k floor)** |
| **Assamese** | 119,265 | 55,002 | 4,376,034 | 25,288 | **MET (>55k / 50k floor)** |
| **Sinhala** | 113,296 | 66,954 | 5,799,421 | 30,230 | **MET (>66k / 50k floor)** |

### Data Floor Analysis & Mitigation Strategy
- **Assamese (`as`)**: MET floor with **55,002** high-quality, script-pure sentences (4,376,034 tokens).
- **Sinhala (`si`)**: MET floor with **66,954** high-quality, script-pure sentences (5,799,421 tokens).
- **Konkani (`gom`)**: Reached **30,980** sentences (2,752,735 tokens). The shortfall relative to the 50k floor is due to the inherent digital scarcity of Goan Devanagari Konkani (the entire gom.wikipedia has only 3,644 articles).
  - **Mitigation Plan**: For Konkani, Phase 4 (Weak Supervision) will rely more heavily on gazetteer expansion, morpho-syntactic rule-based labeling functions, and iterative pseudo-labeling / self-training (Phase 10) to bootstrap sufficient training density.

---

## 2. Granular Breakdown by Source File

| Language | Source File | Sentences | Tokens | Unique Surface Vocab |
| :--- | :--- | :---: | :---: | :---: |
| Konkani | `wikipedia.txt` | 30,798 | 2,740,318 | 18,174 |
| Konkani | `news.txt` | 162 | 11,985 | 664 |
| Konkani | `other.txt` | 20 | 432 | 101 |
| Assamese | `wikipedia.txt` | 54,893 | 4,365,795 | 25,227 |
| Assamese | `news.txt` | 92 | 9,907 | 695 |
| Assamese | `other.txt` | 17 | 332 | 106 |
| Sinhala | `wikipedia.txt` | 66,887 | 5,796,823 | 30,221 |
| Sinhala | `news.txt` | 52 | 2,272 | 378 |
| Sinhala | `other.txt` | 15 | 326 | 96 |

---

## 3. Filtering & Purity Criteria Enforced
- **Script Purity**: Threshold >= 0.90 (at least 90% of alphabetic characters must belong to the language's native Unicode script block).
- **Length Threshold**: Sentences with fewer than 3 tokens were dropped.
- **Deduplication**: Exact SHA-256 sentence hash deduplication across all sources per language.
- **Boilerplate Cleanup**: Dropped navigation headers, URL snippets, and Wikipedia category markers.
