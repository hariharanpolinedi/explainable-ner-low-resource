# Phase 3 Test & Verification Results

**Date**: 2026-09-16  
**Overall Status**: **PASS**  

---

## Test Execution Summary

| # | Test Check | Target Criteria | Status | Details |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **Tagset & BIO Transitions** | 9-class BIO, 0 illegal transitions | **PASS** | All 9 CoNLL files strictly conform to the 9-class BIO tagset with 0 illegal transitions. |
| 2 | **Inter-Annotator Agreement** | Cohen's κ >= 0.75 | **PASS** | All 3 languages achieved Cohen's kappa >= 0.75 (Konkani: 0.9923, Assamese: 0.9738, Sinhala: 0.9317). |
| 3 | **Split-Balance Check** | Non-zero PER/LOC/ORG/MISC in all splits | **PASS** | All splits across all 3 languages contain non-zero counts of PER, LOC, ORG, and MISC. |
| 4 | **Test-Lock Checksum Check** | SHA-256 matches lockfile exactly | **PASS** | All gold test set checksums match reports/gold_test_lockfile.md exactly. |

---

## Detailed Split Balance Table

| Language | Split | PER | LOC | ORG | MISC | Total Entities |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `train` | 61 | 91 | 110 | 110 | **372** |
| **Konkani** | `dev` | 20 | 31 | 37 | 37 | **125** |
| **Konkani** | `test` | 20 | 30 | 37 | 37 | **124** |
| **Assamese** | `train` | 55 | 237 | 44 | 80 | **416** |
| **Assamese** | `dev` | 18 | 79 | 15 | 27 | **139** |
| **Assamese** | `test` | 19 | 79 | 14 | 26 | **138** |
| **Sinhala** | `train` | 113 | 116 | 83 | 81 | **393** |
| **Sinhala** | `dev` | 37 | 39 | 28 | 27 | **131** |
| **Sinhala** | `test` | 38 | 39 | 28 | 27 | **132** |

---

## Cryptographic Checksum Lock Verification

| Language | File | SHA-256 Checksum | Verified in Lockfile |
| :--- | :--- | :--- | :---: |
| **Konkani** | `data/gold/konkani/test.conll` | `bd08ebbf60e048d4fce5d9f5fd233f7dbdeb5bf1f066ee0718eb4b04404ab2d1` | **VERIFIED** |
| **Assamese** | `data/gold/assamese/test.conll` | `2a49321609716b38a893d7b7ca5aa3c27ba0c90863fbca8ac0d4aa87d143c070` | **VERIFIED** |
| **Sinhala** | `data/gold/sinhala/test.conll` | `47cf890b0e9c76e235e17355806b521f8925525e80256bf6667b2835a16854d0` | **VERIFIED** |

