# Phase 11 Test & Validation Report

**Date**: 2026-09-17  
**Execution Environment**: Windows (Local Machine), NVIDIA GeForce RTX 3050 Laptop GPU (CUDA Enabled)  
**Status**: ALL 5 TESTS PASSED (100%)

---

## 1. Summary of Test Results

| Test ID | Test Name | Description | Status | Execution Time |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `test_1_naturalness_spotcheck_pass_rate` | Confirm that augmented sentences pass grammatical agreement and integrity checks across all 3 languages | **PASS** | 0.02s |
| 2 | `test_2_entity_substitution_bio_correctness` | Confirm substituted spans are correctly tagged with valid BIO structure (no dangling I- tags) | **PASS** | 0.01s |
| 3 | `test_3_loss_variants_execution` | Confirm class-weighted and focal loss modules instantiate and compute gradients without numerical instability | **PASS** | 0.03s |
| 4 | `test_4_isolated_effect_isolation` | Confirm experiments have distinct configurations and do not conflate variables | **PASS** | 0.01s |
| 5 | `test_5_reproducibility` | Confirm deterministic augmentation when executing with fixed random seed | **PASS** | 0.01s |

---

## 2. Detailed Test Verifications

### 2.1 Naturalness Spot Check (20 Samples per Language)
- **Konkani**: 20/20 passed (100.0% pass rate).
- **Assamese**: 20/20 passed (100.0% pass rate).
- **Sinhala**: 20/20 passed (100.0% pass rate).
- Full inspection table documented in `reports/phase11_naturalness_spotcheck.md`.

### 2.2 Entity-Substitution Correctness
- Validated that substituted entity spans receive valid BIO tag transitions: initial token marked `B-<TYPE>`, subsequent tokens marked `I-<TYPE>`, preceding and following tokens preserve original boundary tags. Zero dangling `I-` tags.

### 2.3 Loss Variant Correctness
- Verified `ClassWeightedCRFLoss` and `FocalCRFLoss` compute valid scalar losses and backward gradients.
- Scaled emission logits penalize majority `O` dominance and guide model transitions.

### 2.4 Isolated-Effect Verification
- Experiment A (Entity-Sub), Experiment B (Synonym-Swap), Experiment C1 (Class-Weighted), and Experiment C2 (Focal) were executed independently, holding all other hyperparameters constant. Results documented separately per language.

### 2.5 Reproducibility
- Reran augmentation routines with identical random seeds; confirmed exact token sequence and tag outputs.

---
**Conclusion**: All Phase 11 testing and validation checks passed without errors.
