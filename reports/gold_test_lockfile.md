# Gold Test Split Lockfile (Phase 3)

> [!IMPORTANT]
> **IMMUTABLE GOLD TEST BENCHMARK LOCK**  
> The test splits documented below represent the final, adjudicated gold standard test sets for Konkani, Assamese, and Sinhala.
> In accordance with project governance, **these files must never be modified, filtered, augmented, or retagged** between Phase 3 and Phase 14.
> Phase 14 evaluation will independently compute SHA-256 checksums and verify exact byte-level equality against this lockfile.

**Date Created**: 2026-09-16  
**Lock Status**: **LOCKED & VERIFIED**  

---

## Cryptographic Checksums (SHA-256)

| Language | File Path | Sentences | Tokens | Entities | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Konkani** | `data/gold/konkani/test.conll` | 200 | 3,210 | 124 | `bd08ebbf60e048d4fce5d9f5fd233f7dbdeb5bf1f066ee0718eb4b04404ab2d1` |
| **Assamese** | `data/gold/assamese/test.conll` | 200 | 3,349 | 138 | `2a49321609716b38a893d7b7ca5aa3c27ba0c90863fbca8ac0d4aa87d143c070` |
| **Sinhala** | `data/gold/sinhala/test.conll` | 200 | 3,996 | 132 | `47cf890b0e9c76e235e17355806b521f8925525e80256bf6667b2835a16854d0` |

---

## Verification Instructions for Phase 14

To verify integrity at Phase 14 prior to final benchmark scoring:
```powershell
Get-FileHash data/gold/*/test.conll -Algorithm SHA256
```
All hashes must match the entries above byte-for-byte.

