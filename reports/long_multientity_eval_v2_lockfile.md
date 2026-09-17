# Long Multi-Entity Evaluation Set (v2) Lockfile

> [!IMPORTANT]
> **BIAS-FREE EVALUATION BENCHMARK LOCK (v2)**  
> This lockfile documents the newly constructed, unbiased `long_multientity_eval_v2.conll` datasets.
> Sourced strictly via length/structure filtering, with zero gazetteer/rule-tagger filters, and verified by dual-pass human annotation.

**Date Created**: 2026-09-17  
**Lock Status**: **LOCKED & VERIFIED**  

---

## Cryptographic Checksums (SHA-256)

| Language | File Path | Sentences | Total Entities | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| **Konkani** | `data\gold\konkani\long_multientity_eval_v2.conll` | 35 | 83 | `4884414b52492e5d80270e5594f76f0101593efeb6db7af44beb3c8723bc4aa5` |
| **Assamese** | `data\gold\assamese\long_multientity_eval_v2.conll` | 35 | 76 | `bce6d69071a26e3180eae58f54e9e22cf66a9bd6389ca5aa51f26fbb5106b17c` |
| **Sinhala** | `data\gold\sinhala\long_multientity_eval_v2.conll` | 35 | 53 | `645d4b040262fd5184738b75266df99ffdba3108248292857f2f0f905d69f8fd` |

---

## Verification Command

```powershell
Get-FileHash data/gold/*/long_multientity_eval_v2.conll -Algorithm SHA256
```