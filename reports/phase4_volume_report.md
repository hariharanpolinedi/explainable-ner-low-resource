# Silver Dataset Volume & Distribution Report (Phase 4)

**Date**: 2026-09-16  
**Target Volume Multiplier**: 10–20x of Gold-Train Size (600 sentences) -> 6,000 to 12,000 sentences  
**Generated Volume**: 7,200 sentences per language (12x multiplier)  

---

## 1. Volume Across Target Languages

| Language | Gold-Train Sents | Silver Sents | Multiplier | Total Silver Tokens | Total Silver Entities | Entity Density | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 600 | 7,200 | **12x** | 90,773 | 514 | 0.07 ents/sent | **PASS (10-20x Target Met)** |
| **Assamese** | 600 | 7,200 | **12x** | 113,715 | 2,767 | 0.38 ents/sent | **PASS (10-20x Target Met)** |
| **Sinhala** | 600 | 7,200 | **12x** | 117,119 | 1,074 | 0.15 ents/sent | **PASS (10-20x Target Met)** |

---

## 2. Silver Entity Distribution by Class

| Language | PER | LOC | ORG | MISC | Total Entities |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 10 | 227 | 78 | 199 | **514** |
| **Assamese** | 357 | 1,477 | 249 | 684 | **2,767** |
| **Sinhala** | 183 | 220 | 227 | 444 | **1,074** |

---

## 3. Leakage Prevention Audit

- **Exclusion Verification**: 100% of sentences from `data/gold/{lang}/{train,dev,test}.conll` were indexed and strictly excluded from the candidate pool before silver selection.
- **Disjoint Partitions**: Verified that zero sentences in `data/silver/{lang}/train.conll` overlap with any gold sentence.

