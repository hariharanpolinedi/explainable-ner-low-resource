# Checkpoint Provenance Trace & Deployment Audit

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Objective**: Determine exact file path, SHA-256 hash, git history, and phase of origin for the model currently deployed in the web UI demo (`deployment/bundles/{lang}/v1/model.pt`).

---

## 1. Summary of Deployed Models

| Language | Deployed Bundle Path | SHA-256 Checksum | Identical Source in `checkpoints/` | Origin Phase | Commit Hash | Is Rejected Patch? |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Konkani** | `deployment/bundles/konkani/v1/model.pt` | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | `final_konkani.pt`<br>`selftrained_konkani_best.pt` | **Phase 10 (Self-Training Baseline)** | `c76f8c2` | **NO** |
| **Assamese** | `deployment/bundles/assamese/v1/model.pt` | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | `final_assamese.pt`<br>`augmented_assamese.pt` | **Phase 11 (Data Augmentation Baseline)** | `c76f8c2` | **NO** |
| **Sinhala** | `deployment/bundles/sinhala/v1/model.pt` | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | `final_sinhala.pt`<br>`augmented_sinhala.pt` | **Phase 11 (Data Augmentation Baseline)** | `c76f8c2` | **NO** |

---

## 2. Key Findings & Provenance Trace

1. **Not Running Rejected Patches**:
   - The deployed live demo is **NOT** running `patched_{lang}.pt` (from Phase 10/11 Patch v1) and is **NOT** running `patched_v2_{lang}.pt` (from Patch v2).
   - The bundles were created during Phase 15 (commit `c76f8c2`), which copied `checkpoints/final_{lang}.pt`.
2. **Byte-Level Identity**:
   - For Konkani, `final_konkani.pt` is byte-for-byte identical to `selftrained_konkani_best.pt` (Phase 10).
   - For Assamese, `final_assamese.pt` is byte-for-byte identical to `augmented_assamese.pt` (Phase 11).
   - For Sinhala, `final_sinhala.pt` is byte-for-byte identical to `augmented_sinhala.pt` (Phase 11).
3. **Implication**:
   - The visible failures in the manual UI testing (such as missing `पणजींत` and `गोयाच्या` in Konkani) reflect the genuine, unpatched baseline model (`selftrained_konkani_best.pt`).
