# Gold Dataset Split & Entity Distribution Report (Phase 3)

**Date**: 2026-09-16  
**Format**: CoNLL BIO (`<token>\t<tag>`)  
**Splits**: Train (60%), Dev (20%), Test (20%)  

---

## 1. Summary Across Languages and Splits

| Language | Split | Sentences | Tokens | Total Entities | PER | LOC | ORG | MISC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `train` | 600 | 9,511 | **372** | 61 | 91 | 110 | 110 |
| **Konkani** | `dev` | 200 | 3,173 | **125** | 20 | 31 | 37 | 37 |
| **Konkani** | `test` | 200 | 3,210 | **124** | 20 | 30 | 37 | 37 |
| **Assamese** | `train` | 600 | 10,134 | **416** | 55 | 237 | 44 | 80 |
| **Assamese** | `dev` | 200 | 3,569 | **139** | 18 | 79 | 15 | 27 |
| **Assamese** | `test` | 200 | 3,349 | **138** | 19 | 79 | 14 | 26 |
| **Sinhala** | `train` | 600 | 11,626 | **393** | 113 | 116 | 83 | 81 |
| **Sinhala** | `dev` | 200 | 4,032 | **131** | 37 | 39 | 28 | 27 |
| **Sinhala** | `test` | 200 | 3,996 | **132** | 38 | 39 | 28 | 27 |

---

## 2. Verification of Split Balance

- **Zero Missing Entity Types**: Every split (train, dev, test) across all three languages contains non-zero instances of all four entity types (`PER`, `LOC`, `ORG`, `MISC`).
- **Ratio Fidelity**: The 60/20/20 sentence split ratio was strictly maintained across all 1,000 annotated sentences per language.
- **BIO Sequence Integrity**: All 3,000 sentences were verified against valid BIO transitions with zero orphan `I-` tags.

