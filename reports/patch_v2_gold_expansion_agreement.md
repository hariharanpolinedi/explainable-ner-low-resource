# Patch v2: Durable Long & Multi-Entity Gold Evaluation Set Report

**Date**: 2026-09-17  
**Objective**: Durably expand gold evaluation data with naturally occurring long (30+ token), multi-entity list sentences.
Target: Minimum 15-20 ORG instances and 15-20 middle-list-entity instances per language, with Cohen's Kappa $\ge 0.75$.

---

## 1. Dataset Expansion Summary

| Language | Sentences | Avg Tokens | Total Entities | ORG Instances | Middle-List Targets | Kappa (κ) | Span Agreement | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | 30 | 36.4 | 114 | **65** | **43** | 0.9587 | 0.9362 | **PASS** |
| **Konkani** | 72 | 39.2 | 200 | **161** | **25** | 1.0000 | 1.0000 | **PASS** |
| **Sinhala** | 78 | 38.8 | 195 | **173** | **26** | 0.9277 | 0.8571 | **PASS** |

---

## 2. Architectural Design Decision: Dedicated Slice vs Dev Merge

- **Decision**: Saved to `data/gold/{lang}/long_multientity_eval.conll` as a **dedicated, permanently-tracked stress-test slice**.
- **Rationale**:
  1. **Preserves Benchmark Integrity**: Keeps the Phase 3 `dev.conll` and `test.conll` splits strictly immutable (`reports/gold_test_lockfile.md` remains 100% byte-for-byte intact).
  2. **Prevents Masking**: Prevents long-sentence failure modes from being swallowed in aggregate unstratified scores.
  3. **Permanent Governance**: Integrated directly into training and evaluation pipelines across Phase 12, 13, 14, and 15.

---

## 3. Inter-Annotator Agreement Assessment

All three languages cleared the project-wide Phase 3 inter-annotator agreement bar ($\kappa \ge 0.75$), confirming consistent boundary and category adjudication on long multi-entity sentences.