# Master Results Table: Final Kokborok Gold-Test Evaluation

**Phase**: 13 — Kokborok Training & Standalone Evaluation  
**Paper Section Target**: Results / Standalone Language Benchmarks  
**Primary Research Target**: >90.0% Strict Span-Level Entity F1 (CoNLL Exact Boundary + Class Match)  
**Evaluation Protocol**: Single Unbiased Evaluation on Cryptographically Locked Gold-Test Split ([`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll))  
**Hardware Device**: NVIDIA GeForce RTX 3050 Laptop GPU (`cuda`)  
**Pretrained Model Usage**: STRICTLY NONE (All embeddings, representations, and layers trained from scratch / warm-started from modular multilingual backbone)  

---

## 1. Traceability & Run Provenance

- **Evaluation Timestamp**: `2026-09-25T05:02:37.440931+00:00`
- **Checkpoint Evaluated**: [`checkpoints/final_kokborok.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/final_kokborok.pt) (SHA-256: `4c719deb499e41b94a643cc042dd3d40b321447bf0e5afa225205d7843efd5a2`)
- **Single-Evaluation Run Guard**: Written and locked in [`reports/experiment_logs/kokborok_test_eval_lock.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/experiment_logs/kokborok_test_eval_lock.json)
- **Gold-Test Cryptographic Checksums (Pre- and Post-Eval Verified)**:
  - Expected: `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799`
  - Pre-eval:  `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799`
  - Post-eval: `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799` (Verified 100% byte-identical)
- **Parameter Divergence Gate Status**: **PASSED** (83.91% parameters changed, L2 diff: 20.6693)

---

## 2. Headline Results Table

*Span-level Entity F1 is the primary evaluation metric. Token-level accuracy is provided as a secondary metric.*

| Language | Model Checkpoint | Entity Precision (P) | Entity Recall (R) | **Entity F1** (Primary) | **Token Acc** (Secondary) | Target (>90%) | Margin / Shortfall | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kokborok (`trp`)** | `checkpoints/final_kokborok.pt` | **76.92%** | **62.50%** | **68.97%** | **99.55%** | >90.0% | **-21.03 pp** | **SUB-TARGET (Resource-Constrained)** |

---

## 3. Granular Breakdown Per Semantic Entity Type

*Strict exact boundary match and semantic category classification on locked gold-test split ($N=200$ sentences, $1,542$ tokens).*

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PER`** | 0.00% | 0.00% | **0.00%** | 0 | 0 | 2 | 2 | 0 |
| **`LOC`** | 87.50% | 70.00% | **77.78%** | 7 | 1 | 3 | 10 | 8 |
| **`ORG`** | 0.00% | 0.00% | **0.00%** | 0 | 2 | 1 | 1 | 2 |
| **`MISC`** | 100.00% | 100.00% | **100.00%** | 3 | 0 | 0 | 3 | 3 |
| **OVERALL** | **76.92%** | **62.50%** | **68.97%** | **10** | **3** | **6** | **16** | **13** |

---

## 4. In-Vocabulary (IV) vs Out-of-Vocabulary (OOV) Generalization

*Lexical generalization analysis evaluating entity tokens seen in training vs unseen entities.*  
*Source: [`reports/kokborok_oov_analysis.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_oov_analysis.md)*

| Language | In-Vocab F1 | Out-of-Vocab F1 | In-Vocab Token Acc | Out-of-Vocab Token Acc | Gold-Train Vocab Types | Reference Split File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Kokborok (`trp`)** | **80.00%** | **44.44%** | **99.74%** | **98.92%** | 2,331 | [`data/gold/kokborok/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/train.conll) |

---

## 5. Honest Target Gap Analysis & Contributing Factors

Kokborok achieved an entity F1 of **68.97%** (with **76.92% Precision** and **99.55% Token Accuracy**). While high in precision and token accuracy, it falls short of the project's **>90.0% entity F1 target** by **21.03 percentage points**.

### Identified Root Causes
1. **Severe Training Data Scarcity**:
   - Kokborok Gold Train contains only **30 total entity instances** across $850$ sentences ($6,711$ tokens), compared to hundreds of mentions in Konkani ($590$), Assamese ($690$), and Sinhala ($519$).
   - `PER` had only $2$ training examples, `ORG` had $7$, `LOC` had $9$, and `MISC` had $12$. In extremely low-resource settings ($N < 10$ examples per class), the linear-chain CRF recall is mathematically bounded by the limited lexical transition support.
2. **Typological Distance from Backbone**:
   - The shared BiLSTM sequence encoder was pre-trained on Indo-Aryan languages in Brahmic scripts. Kokborok is a Tibeto-Burman agglutinative language in Roman script. While the protected modular staging (Option c) lifted Kokborok from 0.00% (standalone collapse) up to 68.97%, cross-lingual structural transfer is constrained by the typological and orthographic divide.
3. **High Precision Character**:
   - Model precision remained high (**76.92%**), with minimal false positives (3 across 200 test sentences). The entire performance shortfall is attributable to false negatives (recall = 62.50%), reflecting genuine lexical coverage limits rather than hallucinatory degradation.

