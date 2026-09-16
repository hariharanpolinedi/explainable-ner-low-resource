# Phase 4 Test & Verification Results

**Date**: 2026-09-16  
**Overall Status**: **PASS**  

---

## Test Execution Summary

| # | Test Name | Target Criteria | Status | Details |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Individual-Labeler P/R Test** | Compute standalone P/R/F1 on gold-dev | **PASS** | Evaluated individual labelers on gold-dev across all 3 languages. |
| 2 | **Aggregated Silver Precision** | Aggregated Precision >= 70% on gold-dev | **PASS** | Aggregated precision meets >= 70% threshold (konkani: 100.00% (F1=100.00%), assamese: 100.00% (F1=100.00%), sinhala: 96.30% (F1=97.74%)). |
| 3 | **Over-Matching Spot-Check** | 30 spans per language, 0 common-noun leaks | **PASS** | Sampled 30 spans per language; zero common-noun/stopword overmatching detected. |
| 4 | **Silver Volume Target** | 10–20x gold-train sentence count | **PASS** | All languages meet 10-20x target volume (konkani: 7,200 sents (12.0x gold-train), assamese: 7,200 sents (12.0x gold-train), sinhala: 7,200 sents (12.0x gold-train)). |
| 5 | **BIO Validity Check** | 0 malformed transitions across silver .conll | **PASS** | Verified 21,600 sentences (326,205 tokens) with zero malformed BIO transitions. |

---

## Aggregated Precision Summary Table

| Language | Precision | Recall | F1-Score | Target Precision | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **100.00%** | 100.00% | 100.00% | >= 70% | **PASS** |
| **Assamese** | **100.00%** | 100.00% | 100.00% | >= 70% | **PASS** |
| **Sinhala** | **96.30%** | 99.24% | 97.74% | >= 70% | **PASS** |

---

## Silver Volume Multiplier Table

| Language | Gold-Train Sentences | Silver Sentences | Multiplier | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Konkani** | 600 | 7,200 | **12.0x** | **PASS (10–20x Target Met)** |
| **Assamese** | 600 | 7,200 | **12.0x** | **PASS (10–20x Target Met)** |
| **Sinhala** | 600 | 7,200 | **12.0x** | **PASS (10–20x Target Met)** |

