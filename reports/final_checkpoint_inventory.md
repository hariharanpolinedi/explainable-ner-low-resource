# Comprehensive Checkpoint Inventory & Loadability Audit

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Constraint**: Zero pretrained models/embeddings. Architectural verification under current code.  

---

## 1. Summary: 50 Total Checkpoints Audited

- **Total Checkpoints Found**: 50
- **Loadable Checkpoints**: 50 / 50 (100.0%)
- **Excluded Checkpoints**: 0

---

## 2. Complete Checkpoint Inventory Table

| Checkpoint File | Target Lang | Architecture / Class | Size (MB) | Provenance Phase | Commit | Load Status |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: |
| `attention_assamese.pt` | Assamese | NERAttentionModel (widths=(3, 4, 5)) | 16.19 | Phase 7 (Self-Attention) | `8cf71bd3` | **LOADABLE** |
| `attention_konkani.pt` | Konkani | NERAttentionModel (widths=(2, 3, 4)) | 16.43 | Phase 7 (Self-Attention) | `8cf71bd3` | **LOADABLE** |
| `attention_sinhala.pt` | Sinhala | NERAttentionModel (widths=(3, 4, 5)) | 15.86 | Phase 7 (Self-Attention) | `8cf71bd3` | **LOADABLE** |
| `augmented_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Phase 11 (Data Augmentation) | `82daf0e6` | **LOADABLE** |
| `augmented_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Phase 11 (Data Augmentation) | `82daf0e6` | **LOADABLE** |
| `baseline_assamese.pt` | Assamese | NERBaselineModel (BiLSTM+CRF) | 41.6 | Phase 5 (Baseline BiLSTM+CRF) | `cef928cf` | **LOADABLE** |
| `baseline_konkani.pt` | Konkani | NERBaselineModel (BiLSTM+CRF) | 41.6 | Phase 5 (Baseline BiLSTM+CRF) | `cef928cf` | **LOADABLE** |
| `baseline_sinhala.pt` | Sinhala | NERBaselineModel (BiLSTM+CRF) | 41.6 | Phase 5 (Baseline BiLSTM+CRF) | `cef928cf` | **LOADABLE** |
| `charcnn_assamese.pt` | Assamese | NERCharCNNModel (widths=(3, 4, 5)) | 15.18 | Phase 6 (CharCNN+BiLSTM) | `cf36405b` | **LOADABLE** |
| `charcnn_konkani.pt` | Konkani | NERCharCNNModel (widths=(2, 3, 4)) | 15.43 | Phase 6 (CharCNN+BiLSTM) | `cf36405b` | **LOADABLE** |
| `charcnn_sinhala.pt` | Sinhala | NERCharCNNModel (widths=(3, 4, 5)) | 14.85 | Phase 6 (CharCNN+BiLSTM) | `cf36405b` | **LOADABLE** |
| `conjunct_fixed_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Remediation (Sinhala Conjunct) | `ba2ca34b` | **LOADABLE** |
| `final_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Phase 12 (Locked Benchmark) | `63ae655e` | **LOADABLE** |
| `final_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Phase 12 (Locked Benchmark) | `63ae655e` | **LOADABLE** |
| `final_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Phase 12 (Locked Benchmark) | `63ae655e` | **LOADABLE** |
| `finetuned_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 42.48 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `finetuned_assamese_immediate.pt` | Assamese | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `finetuned_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `finetuned_konkani_immediate.pt` | Konkani | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `finetuned_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `finetuned_sinhala_immediate.pt` | Sinhala | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Language Fine-Tuning) | `f877c2fa` | **LOADABLE** |
| `fromscratch_control_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Control Baseline) | `f877c2fa` | **LOADABLE** |
| `fromscratch_control_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Control Baseline) | `f877c2fa` | **LOADABLE** |
| `fromscratch_control_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 121.3 | Phase 9 (Control Baseline) | `f877c2fa` | **LOADABLE** |
| `inflected_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Remediation (Inflection Patch) | `9050d2ee` | **LOADABLE** |
| `inflected_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Remediation (Inflection Patch) | `9050d2ee` | **LOADABLE** |
| `inflected_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Remediation (Inflection Patch) | `9050d2ee` | **LOADABLE** |
| `joint_best.pt` | Multilingual | NERMultilingualModel (shared CRF) | 121.3 | Phase 8 (Joint Multilingual) | `236d5eab` | **LOADABLE** |
| `joint_per_language_proportional.pt` | Multilingual | NERMultilingualModel (per_language CRF) | 121.37 | Phase 8 (Joint Multilingual) | `236d5eab` | **LOADABLE** |
| `joint_per_language_uniform.pt` | Multilingual | NERMultilingualModel (per_language CRF) | 121.37 | Phase 8 (Joint Multilingual) | `236d5eab` | **LOADABLE** |
| `joint_shared_proportional.pt` | Multilingual | NERMultilingualModel (shared CRF) | 121.3 | Phase 8 (Joint Multilingual) | `236d5eab` | **LOADABLE** |
| `joint_shared_uniform.pt` | Multilingual | NERMultilingualModel (shared CRF) | 121.3 | Phase 8 (Joint Multilingual) | `236d5eab` | **LOADABLE** |
| `loc_patched_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Remediation (LOC Recovery) | `18f40772` | **LOADABLE** |
| `loc_patched_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Remediation (LOC Recovery) | `18f40772` | **LOADABLE** |
| `loc_patched_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Remediation (LOC Recovery) | `18f40772` | **LOADABLE** |
| `patched_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v1) | `6e330a84` | **LOADABLE** |
| `patched_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v1) | `6e330a84` | **LOADABLE** |
| `patched_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v1) | `6e330a84` | **LOADABLE** |
| `patched_v2_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v2) | `8a0695ee` | **LOADABLE** |
| `patched_v2_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v2) | `8a0695ee` | **LOADABLE** |
| `patched_v2_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Post-Phase-11 (Patch v2) | `8a0695ee` | **LOADABLE** |
| `per_fixed_assamese.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Remediation (PER Boundary) | `315d0eb9` | **LOADABLE** |
| `per_fixed_konkani.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Remediation (PER Boundary) | `315d0eb9` | **LOADABLE** |
| `per_fixed_sinhala.pt` | Sinhala | NERMultilingualModel (shared CRF) | 40.44 | Remediation (PER Boundary) | `315d0eb9` | **LOADABLE** |
| `selftrained_assamese_best.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |
| `selftrained_assamese_round1.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |
| `selftrained_assamese_round2.pt` | Assamese | NERMultilingualModel (shared CRF) | 40.44 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |
| `selftrained_konkani_best.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |
| `selftrained_konkani_round1.pt` | Konkani | NERMultilingualModel (shared CRF) | 40.44 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |
| `selftrained_sinhala_best.pt` | Sinhala | NERMultilingualModel (shared CRF) | 121.3 | Phase 10 (Self-Training) | `ddb8e552` | **LOADABLE** |