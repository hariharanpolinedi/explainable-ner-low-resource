# Master Results Table: Final Gold-Test Evaluation

**Paper Section Target**: Results / Evaluation Benchmarks  
**Primary Research Target**: >90.0% Strict Span-Level Entity F1 (CoNLL Exact Boundary + Class Match)  
**Evaluation Protocol**: Single Unbiased Evaluation on Cryptographically Locked Gold-Test Splits (`data/gold/{lang}/test.conll`)  
**Hardware Device**: NVIDIA GeForce RTX 3050 Laptop GPU (`cuda`)  
**Pretrained Model Usage**: STRICTLY NONE (All embeddings, representations, and layers trained from scratch)  

---

## 1. Traceability & Run Provenance

- **Primary Source Report**: [`reports/phase14_final_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md)
- **Evaluation Run Date/Timestamp**: `2026-09-17T05:03:59.541563+00:00` (UTC)
- **Git Commit of Record**: `8efcd16` (*Phase 14: final locked gold-test evaluation and error analysis*)
- **Subsequent Audit Re-verification**: Commit `0ea0e46` audited all 50 checkpoints in project history on locked gold-dev and verified that the locked deployment models `final_{lang}.pt` are byte-identical to `selftrained_konkani_best.pt` (Phase 10), `augmented_assamese.pt` (Phase 11), and `augmented_sinhala.pt` (Phase 11). To maintain the strict "single-evaluation" discipline and prevent test-set peeking/contamination, the test set was not re-run after `0ea0e46`; this report documents the authoritative, single-run Phase 14 evaluation.
- **Gold-Test Cryptographic Checksums (Verified Unmodified)**:
  - Konkani (`data/gold/konkani/test.conll`): `bd08ebbf60e048d4fce5d9f5fd233f7dbdeb5bf1f066ee0718eb4b04404ab2d1`
  - Assamese (`data/gold/assamese/test.conll`): `2a49321609716b38a893d7b7ca5aa3c27ba0c90863fbca8ac0d4aa87d143c070`
  - Sinhala (`data/gold/sinhala/test.conll`): `47cf890b0e9c76e235e17355806b521f8925525e80256bf6667b2835a16854d0`

---

## 2. Headline Results Table

*Span-level Entity F1 is the primary evaluation metric. Token-level accuracy is provided as a secondary metric.*

| Language | Model Checkpoint | Entity Precision (P) | Entity Recall (R) | **Entity F1** (Primary) | Target (>90%) | Margin (pp) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `checkpoints/final_konkani.pt` | 99.15% | 93.55% | **96.27%** | >90.0% | +6.27 | **MET** |
| *Konkani (Joint Baseline)* | `checkpoints/joint_best.pt` | 98.31% | 93.55% | 95.87% | >90.0% | +5.87 | Baseline |
| **Assamese** | `checkpoints/final_assamese.pt` | 95.83% | 100.00% | **97.87%** | >90.0% | +7.87 | **MET** |
| *Assamese (Joint Baseline)* | `checkpoints/joint_best.pt` | 96.48% | 99.28% | 97.86% | >90.0% | +7.86 | Baseline |
| **Sinhala** | `checkpoints/final_sinhala.pt` | 96.32% | 99.24% | **97.76%** | >90.0% | +7.76 | **MET** |
| *Sinhala (Joint Baseline)* | `checkpoints/joint_best.pt` | 94.93% | 99.24% | 97.04% | >90.0% | +7.04 | Baseline |

### Secondary Row: Token-Level Sequence Accuracy

| Language | Final Model Token Accuracy | Joint Baseline Token Accuracy | Delta (pp) | Reference File |
| :--- | :---: | :---: | :---: | :--- |
| **Konkani** | **99.47%** | 99.41% | +0.06 | [`reports/phase14_final_results.md#L18-L19`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L18-L19) |
| **Assamese** | **99.64%** | 99.61% | +0.03 | [`reports/phase14_final_results.md#L20-L21`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L20-L21) |
| **Sinhala** | **99.70%** | 99.60% | +0.10 | [`reports/phase14_final_results.md#L22-L23`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L22-L23) |

---

## 3. Granular Breakdown Per Semantic Entity Type

*Strict exact boundary match and semantic category classification on locked gold-test splits.*

### 3.1 Konkani (`checkpoints/final_konkani.pt`)
*Source: [`reports/phase14_final_results.md#L41-L49`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L41-L49)*

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 100.00% | 85.00% | **91.89%** | 17 | 0 | 3 | 20 | 17 |
| **LOC** | 100.00% | 93.33% | **96.55%** | 28 | 0 | 2 | 30 | 28 |
| **ORG** | 97.14% | 91.89% | **94.44%** | 34 | 1 | 3 | 37 | 35 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 37 | 0 | 0 | 37 | 37 |
| **OVERALL** | **99.15%** | **93.55%** | **96.27%** | **116** | **1** | **8** | **124** | **117** |

### 3.2 Assamese (`checkpoints/final_assamese.pt`)
*Source: [`reports/phase14_final_results.md#L50-L58`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L50-L58)*

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 86.36% | 100.00% | **92.68%** | 19 | 3 | 0 | 19 | 22 |
| **LOC** | 97.53% | 100.00% | **98.75%** | 79 | 2 | 0 | 79 | 81 |
| **ORG** | 93.33% | 100.00% | **96.55%** | 14 | 1 | 0 | 14 | 15 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 26 | 0 | 0 | 26 | 26 |
| **OVERALL** | **95.83%** | **100.00%** | **97.87%** | **138** | **6** | **0** | **138** | **144** |

### 3.3 Sinhala (`checkpoints/final_sinhala.pt`)
*Source: [`reports/phase14_final_results.md#L59-L67`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L59-L67)*

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 94.87% | 97.37% | **96.10%** | 37 | 2 | 1 | 38 | 39 |
| **LOC** | 92.86% | 100.00% | **96.30%** | 39 | 3 | 0 | 39 | 42 |
| **ORG** | 100.00% | 100.00% | **100.00%** | 28 | 0 | 0 | 28 | 28 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 27 | 0 | 0 | 27 | 27 |
| **OVERALL** | **96.32%** | **99.24%** | **97.76%** | **131** | **5** | **1** | **132** | **136** |

---

## 4. In-Vocabulary (IV) vs Out-of-Vocabulary (OOV) Generalization

*Lexical generalization analysis evaluating entity tokens seen in training vs unseen entities.*  
*Source: [`reports/phase14_final_results.md#L70-L79`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase14_final_results.md#L70-L79)*

| Language | In-Vocab F1 | Out-of-Vocab F1 | In-Vocab Token Acc | Out-of-Vocab Token Acc | Gold-Train Vocab Types | Reference Split File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | 97.09% | 91.43% | 99.35% | 99.72% | 4,036 | [`data/gold/konkani/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/konkani/train.conll) |
| **Assamese** | 99.21% | 86.67% | 99.76% | 99.30% | 3,767 | [`data/gold/assamese/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/assamese/train.conll) |
| **Sinhala** | 98.94% | 95.00% | 99.65% | 99.83% | 4,599 | [`data/gold/sinhala/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/sinhala/train.conll) |
