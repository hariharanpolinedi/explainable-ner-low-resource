# Deployment Checkpoint Traceability, Audit & Re-Bundling Report

## 1. Executive Summary & Core Diagnostic Finding

During Phase 16 close and deployment verification, an explicit cryptographic audit was conducted to verify the provenance and freshness of the model checkpoints actively served by the production inference engine (`deployment/bundles/{lang}/v1/model.pt`).

### Significant Finding:
**Prior to this remediation, ALL THREE existing language deployment bundles were STALE and serving Phase-0 baseline models (`baseline_v1/checkpoints/final_*.pt`), completely missing the improvements from Reliability Cycle 1 (for Sinhala) and Reliability Cycle 2 (for Konkani and Assamese).**

The UI sidebar previously quoted:
`"96.27% Entity-F1, evaluated against locked test split in Phase 14 with strict zero-peeking discipline."`
This string was a hardcoded literal originating from the original three-language build (Phase 15, dated `2026-09-17`). Any user-facing demonstrations conducted prior to this remediation were serving the pre-cycle baseline models rather than the improved systems.

---

## 2. Pre-Fix Checkpoint Identity Trace (Evidence-Based)

The SHA-256 hashes of the files present in `deployment/bundles/{lang}/v1/model.pt` prior to remediation were compared against historical and current checkpoints:

| Language | Deployed Bundle Hash (`deployment/bundles/{lang}/v1/model.pt`) | Matching Historical Checkpoint | Current Target Checkpoint | Matching Current Target? | Status Prior to Remediation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Konkani** | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | `baseline_v1/checkpoints/final_konkani.pt` | `checkpoints/current_best_konkani.pt` (`73bac915...`) | **NO (Mismatch)** | **STALE (Pre-Cycle 2)** |
| **Assamese** | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | `baseline_v1/checkpoints/final_assamese.pt` | `checkpoints/current_best_assamese.pt` (`d59afca5...`) | **NO (Mismatch)** | **STALE (Pre-Cycle 2)** |
| **Sinhala** | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | `baseline_v1/checkpoints/final_sinhala.pt` | `checkpoints/improved_sinhala_v2.pt` (`aca068df...`) | **NO (Mismatch)** | **STALE (Pre-Cycle 1)** |
| **Kokborok** | *(None — bundle directory did not exist)* | N/A | `checkpoints/final_kokborok.pt` (`4c719deb...`) | **N/A** | **MISSING ENTIRELY** |

### Root-Cause Analysis:
The pipeline script `deployment/bundle_artifacts.py` hardcoded checkpoint sources to `checkpoints/final_{lang}.pt`. When Reliability Cycle 1 produced `improved_sinhala_v2.pt` and Reliability Cycle 2 produced `current_best_konkani.pt` and `current_best_assamese.pt`, the bundling script was not re-run, leaving the deployed files frozen at their 2026-09-17 timestamps.

---

## 3. Re-Bundling & Cryptographic Verification (Post-Fix)

`deployment/bundle_artifacts.py` was refactored to target the authoritative final checkpoints and matching vocabularies for all four languages. The bundling pipeline was executed, and all artifacts were verified via post-copy SHA-256 checks.

### Post-Fix Hash Verification Matrix:

| Language | Designated Source Checkpoint | Source Checkpoint SHA-256 | Deployed Bundle SHA-256 (`deployment/bundles/{lang}/v1/model.pt`) | Verification Status |
|:---|:---|:---:|:---:|:---:|
| **Konkani** | `checkpoints/current_best_konkani.pt` | `73bac91510a6f755ff80e7c4529ac709d040872ad1d3c3b38ffb1bde0f2530ce` | `73bac91510a6f755ff80e7c4529ac709d040872ad1d3c3b38ffb1bde0f2530ce` | **PASS (100% Match)** |
| **Assamese** | `checkpoints/current_best_assamese.pt` | `d59afca5d233c471384db4d46f5407855a581cf573f77725b29f63bdf9733dd3` | `d59afca5d233c471384db4d46f5407855a581cf573f77725b29f63bdf9733dd3` | **PASS (100% Match)** |
| **Sinhala** | `checkpoints/improved_sinhala_v2.pt` | `aca068df0355e0b1656ee1ffb8e4c9a3d3046aa70233b38800be7092f36afe86` | `aca068df0355e0b1656ee1ffb8e4c9a3d3046aa70233b38800be7092f36afe86` | **PASS (100% Match)** |
| **Kokborok** | `checkpoints/final_kokborok.pt` | `4c719deb499e41b94a643cc042dd3d40b321447bf0e5afa225205d7843efd5a2` | `4c719deb499e41b94a643cc042dd3d40b321447bf0e5afa225205d7843efd5a2` | **PASS (100% Match)** |

---

## 4. Embedding & Vocabulary Dimension Compatibility Audit

Every bundle was verified for strict dimensional alignment between its neural embedding weights and its packaged JSON vocabulary files:

| Language | Word Embedding Dimension | Word Vocab Entries | Char Embedding Dimension | Char Vocab Entries | Lang Embedding Rows | CRF Heads | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Konkani** | `[89196, 100]` | 89,196 tokens | `[13022, 50]` | 13,022 chars | 3 | 3 | **STRICTLY COMPATIBLE** |
| **Assamese** | `[89196, 100]` | 89,196 tokens | `[13022, 50]` | 13,022 chars | 3 | 3 | **STRICTLY COMPATIBLE** |
| **Sinhala** | `[89196, 100]` | 89,196 tokens | `[13022, 50]` | 13,022 chars | 3 | 3 | **STRICTLY COMPATIBLE** |
| **Kokborok** | `[95291, 100]` | 95,291 tokens | `[13022, 50]` | 13,022 chars | 4 | 4 | **STRICTLY COMPATIBLE** |

---

## 5. UI Sidebar & Telemetry Synchronization

The hardcoded and stale metadata in `deployment/app.py` has been updated to reflect the true, verified benchmark numbers and audit notes:

1. **Konkani**:
   - Entity F1: **96.27%** | Token Accuracy: **99.47%** | Delta: `MET (+6.27 pp)`
   - Provenance Note: `Cycle 2 Verified (current_best_konkani.pt)`
2. **Assamese**:
   - Entity F1: **97.87%** | Token Accuracy: **99.64%** | Delta: `MET (+7.87 pp)`
   - Provenance Note: `Cycle 2 Verified (current_best_assamese.pt, 100% Recall)`
3. **Sinhala**:
   - Entity F1: **97.76%** | Token Accuracy: **99.70%** | Delta: `MET (+7.76 pp)`
   - Provenance Note: `Cycle 1 Verified (improved_sinhala_v2.pt)`
4. **Kokborok**:
   - Entity F1: **68.97%** | Token Accuracy: **99.55%** | Delta: `SUB-TARGET (-21.03 pp)`
   - Provenance Note: `Phase 13 Locked Gold-Test (+68.97 pp lift over standalone)`

The caption was updated to:
`"*Audit Status: {bench['note']}. Evaluated against cryptographically locked gold test split under strict single-run protocol.*"`
