# LOC-Specific Diagnostic Findings Report

**Date**: 2026-09-17  
**Target**: In-depth empirical diagnosis of Location (`LOC`) detection performance across Konkani, Assamese, and Sinhala.  

---

## 1. Gazetteer Coverage Disparity

| Language | PER Gazetteer Lines | LOC Gazetteer Lines | ORG Gazetteer Lines | LOC Ratio vs (PER+ORG) |
| :--- | :---: | :---: | :---: | :---: |
| **Konkani** | 40 | **69** | 88 | 53.9% |
| **Assamese** | 33 | **50** | 84 | 42.7% |
| **Sinhala** | 30 | **44** | 81 | 39.6% |

---

## 2. Weak Supervision Rule Availability for LOC

| Language | Suffix Patterns Count | Defined Toponymic Suffixes | Suffix Coverage Assessment |
| :--- | :---: | :--- | :--- |
| **Konkani** | 10 | `पूर`, `पुर`, `नगर`, `गांव`, `गाव`, `वाडो`, ... | Pure static suffixes; **zero case inflection handling** |
| **Assamese** | 10 | `পুৰ`, `নগৰ`, `গাঁও`, `পাৰা`, `বাৰী`, `ঘাট`, ... | Pure static suffixes; **zero case inflection handling** |
| **Sinhala** | 11 | `ගම`, `පුර`, `නුවර`, `වත්ත`, `දෙණිය`, `තොට`, ... | Pure static suffixes; **zero case inflection handling** |

---

## 3. LOC In-Vocab vs OOV Generalization Breakdown (Gold-Dev)

| Language | Total Gold LOC | In-Vocab LOC Count | In-Vocab LOC F1 (P/R) | OOV LOC Count | OOV LOC F1 (P/R) | OOV F1 Penalty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 31 | 25 | **100.00%** (100.0/100.0) | 6 | **50.00%** (100.0/33.3) | **+50.00%** |
| **Assamese** | 79 | 75 | **100.00%** (100.0/100.0) | 4 | **75.00%** (75.0/75.0) | **+25.00%** |
| **Sinhala** | 39 | 37 | **98.67%** (97.4/100.0) | 2 | **80.00%** (66.7/100.0) | **+18.67%** |

---

## 4. Root Cause Analysis for LOC Failures (Konkani Manual Test Case)

In the manual test case (`पणजींत जाल्ल्या एका विशेष परिषदेंत... गोयाच्या आर्थिक विकासा खातीर...`):
1. **Case Inflection & Agglutination**: Konkani places take bound locative (`-ंत`, e.g. `पणजी` -> `पणजींत`) and genitive (`-चो/-च्या`, e.g. `गोवा` -> `गोयाच्या`) postpositions.
2. **Gazetteer Blindspot**: Gazetteers contain base nominative lemmas (`पणजी`, `गोवा`), but zero inflected surface variants.
3. **Toponymic Rule Blindspot**: Suffix rules check for `-पूर`, `-गांव`, but never handle case morphemes appended after city names.
4. **Embedding Fallback**: When an inflected place name is OOV in word embeddings, the Char-CNN must shoulder 100% of the representation. Without morphological inflection augmentation, the BiLSTM-CRF misidentifies or drops the boundary.