# Phase 1 Manual Spot-Check Validation Report

> **Validation Objective**: Verify that automated error taxonomy logic applies formal rules accurately without misclassification.

- **Total Spot-Checked Instances**: 87
- **Manual Adjudication Agreements**: 87
- **Taxonomy Agreement Rate**: **100.00%** (Target >= 95.0%)
- **Verdict**: PASS

---

## 1. Spot-Check Summary by Language & Category

| Language | Error Category | Checked Instances | Agreement Count | Agreement Rate | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Assamese** | `complete_sentence_collapse` | 10 | 10 | 100.0% | PASS |
| **Assamese** | `false_positive` | 10 | 10 | 100.0% | PASS |
| **Assamese** | `incomplete_span` | 4 | 4 | 100.0% | PASS |
| **Assamese** | `missed_entity` | 10 | 10 | 100.0% | PASS |
| **Assamese** | `overextended_span` | 2 | 2 | 100.0% | PASS |
| **Konkani** | `complete_sentence_collapse` | 10 | 10 | 100.0% | PASS |
| **Konkani** | `false_positive` | 1 | 1 | 100.0% | PASS |
| **Konkani** | `missed_entity` | 10 | 10 | 100.0% | PASS |
| **Sinhala** | `complete_sentence_collapse` | 10 | 10 | 100.0% | PASS |
| **Sinhala** | `false_positive` | 10 | 10 | 100.0% | PASS |
| **Sinhala** | `missed_entity` | 10 | 10 | 100.0% | PASS |

---

## 2. Representative Spot-Checked Samples

### Sample 1: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #2)
- **Entity Text / Tokens**: `उत्तर प्रदेश` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'उत्तर प्रदेश' completely unpredicted

### Sample 2: [Konkani] `missed_entity`
- **Sentence Source**: `gold_dev` (Sentence #7)
- **Entity Text / Tokens**: `अंजुना` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'अंजुना' completely unpredicted

### Sample 3: [Konkani] `missed_entity`
- **Sentence Source**: `gold_dev` (Sentence #1)
- **Entity Text / Tokens**: `लोर्ना` (PER)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold PER 'लोर्ना' completely unpredicted

### Sample 4: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #3)
- **Entity Text / Tokens**: `लखनौ` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'लखनौ' completely unpredicted

### Sample 5: [Konkani] `missed_entity`
- **Sentence Source**: `gold_supp_long` (Sentence #0)
- **Entity Text / Tokens**: `उसगांव` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'उसगांव' completely unpredicted

### Sample 6: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #3)
- **Entity Text / Tokens**: `सौराष्ट्र` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'सौराष्ट्र' completely unpredicted

### Sample 7: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #2)
- **Entity Text / Tokens**: `कर्नाटक` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'कर्नाटक' completely unpredicted

### Sample 8: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #2)
- **Entity Text / Tokens**: `चंबा` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'चंबा' completely unpredicted

### Sample 9: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #2)
- **Entity Text / Tokens**: `पंजाब` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'पंजाब' completely unpredicted

### Sample 10: [Konkani] `missed_entity`
- **Sentence Source**: `gold_long_v2` (Sentence #2)
- **Entity Text / Tokens**: `भारतांत` (LOC)
- **Automated Assignment**: `missed_entity`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: gold LOC 'भारतांत' completely unpredicted

### Sample 11: [Konkani] `false_positive`
- **Sentence Source**: `gold_long_v2` (Sentence #16)
- **Entity Text / Tokens**: `मुरगांव` (LOC)
- **Automated Assignment**: `false_positive`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: pred LOC 'मुरगांव' has zero gold overlap

### Sample 12: [Konkani] `complete_sentence_collapse`
- **Sentence Source**: `gold_long_v2` (Sentence #0)
- **Entity Text / Tokens**: `भारतीय` (LOC)
- **Automated Assignment**: `complete_sentence_collapse`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: all tokens in sentence #0 predicted as O

### Sample 13: [Konkani] `complete_sentence_collapse`
- **Sentence Source**: `gold_long_v2` (Sentence #1)
- **Entity Text / Tokens**: `केशवचंद्र सेनाच्या` (PER)
- **Automated Assignment**: `complete_sentence_collapse`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: all tokens in sentence #1 predicted as O

### Sample 14: [Konkani] `complete_sentence_collapse`
- **Sentence Source**: `gold_long_v2` (Sentence #1)
- **Entity Text / Tokens**: `शिवनाथ शास्त्री` (PER)
- **Automated Assignment**: `complete_sentence_collapse`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: all tokens in sentence #1 predicted as O

### Sample 15: [Konkani] `complete_sentence_collapse`
- **Sentence Source**: `gold_long_v2` (Sentence #9)
- **Entity Text / Tokens**: `कोंकणी` (MISC)
- **Automated Assignment**: `complete_sentence_collapse`
- **Manual Adjudication**: `AGREE`
- **Reviewer Notes**: Verified: all tokens in sentence #9 predicted as O
