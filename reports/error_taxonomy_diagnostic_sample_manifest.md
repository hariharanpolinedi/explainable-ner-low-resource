# Error Taxonomy Diagnostic Sample Manifest

> **Phase 1 Deliverable — Audit Trail & Reproducibility Ledger**  
> This manifest records the exact composition of the diagnostic sample evaluated during Phase 1.
> In Phase 7 (Post-Intervention Re-Analysis), this exact sample composition will be re-evaluated to enable a clean, paired comparison.

- **Creation Date**: `2026-09-23T04:06:20.231941+00:00`
- **Random Seed**: `42`
- **Test Set Isolation**: Verified 100% disjoint from `data/gold/{lang}/test.conll`

---

## 1. Gold Diagnostic Sample Structure per Language

| Language | Component Dataset | File Path | Sentences | Tokens | Gold Entities | SHA-256 Checksum |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Konkani** | Gold Dev | `data\gold\konkani\dev.conll` | 200 | 3173 | 125 | `d90e1e4718c69db97da9f00d90c1e3711895378a13c72c9a66466d26bd535898` |
| **Konkani** | Long Multi-Entity v2 | `data\gold\konkani\long_multientity_eval_v2.conll` | 35 | 1382 | 83 | `4884414b52492e5d80270e5594f76f0101593efeb6db7af44beb3c8723bc4aa5` |
| **Konkani** | Supplementary Long | `data\gold\konkani\supplementary_long_multientity.conll` | 12 | 506 | 43 | `50f62c1c9e8a7d7f825cc60cc52bc4dea948d002093ab0d40ef9d5641da96367` |
| **Assamese** | Gold Dev | `data\gold\assamese\dev.conll` | 200 | 3569 | 139 | `d4b5a59b4daa76621777992174cef842ae94e9a1da2081a59100803a7df92a81` |
| **Assamese** | Long Multi-Entity v2 | `data\gold\assamese\long_multientity_eval_v2.conll` | 35 | 1385 | 76 | `bce6d69071a26e3180eae58f54e9e22cf66a9bd6389ca5aa51f26fbb5106b17c` |
| **Assamese** | Supplementary Long | `data\gold\assamese\supplementary_long_multientity.conll` | 35 | 1778 | 201 | `c608755568bf6b010a4331c3a347d02113d87d47a86ddb883226014d324f28c8` |
| **Sinhala** | Gold Dev | `data\gold\sinhala\dev.conll` | 200 | 4032 | 131 | `3ce4f3011d2878f99a6fa408232e92103971281f6628a6171fabf127390fe3da` |
| **Sinhala** | Long Multi-Entity v2 | `data\gold\sinhala\long_multientity_eval_v2.conll` | 35 | 1500 | 53 | `645d4b040262fd5184738b75266df99ffdba3108248292857f2f0f905d69f8fd` |
| **Sinhala** | Supplementary Long | `data\gold\sinhala\supplementary_long_multientity.conll` | 28 | 1230 | 116 | `1ce03c5dd126670ee43b3abe31de2be21bbff55b1e040b37c70f3627c4871852` |

---

## 2. Held-Out Raw Corpus Diagnostic Pool

| Language | Source Corpus | Sample Size | Total Tokens | Sampling Method | Disjoint Guarantee |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Konkani** | `data/processed/konkani/sentences.txt` | 1500 | 17984 | Seed `42` pseudo-random sample | Verified zero token overlap with `test.conll` |
| **Assamese** | `data/processed/assamese/sentences.txt` | 1500 | 22740 | Seed `42` pseudo-random sample | Verified zero token overlap with `test.conll` |
| **Sinhala** | `data/processed/sinhala/sentences.txt` | 1500 | 22686 | Seed `42` pseudo-random sample | Verified zero token overlap with `test.conll` |
