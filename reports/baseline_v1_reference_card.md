# Baseline v1 Reference Card

> [!IMPORTANT]
> **IMMUTABLE BENCHMARK NUMBERS — NEVER TO BE OVERWRITTEN**  
> `baseline_v1/` is an immutable historical snapshot. Nothing under this directory is ever modified, retrained, or overwritten by any later phase. All v2 development happens outside this directory.  
> Every subsequent phase in both the **Reliability Overhaul track (Phases 1–7)** and the **Kokborok integration track (Phases 8–17)** must diff strictly against the frozen numbers below.

- **Anchor Commit**: `3235ff12925d3e5393ab1bfcb8bcd68402746f3e`
- **Git Tag**: `baseline-v1-frozen`
- **Primary Source Ledger**: [`paper_assets/results_master_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/results_master_table.md)
- **Ablation Source**: [`paper_assets/ablation_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/ablation_table.md)
- **Faithfulness Source**: [`paper_assets/faithfulness_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/faithfulness_results.md)
- **Model Checkpoints**: [`baseline_v1/checkpoints/final_{konkani,assamese,sinhala}.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/checkpoints)

---

## 1. Primary Gold-Test Headline Benchmarks

Evaluated on cryptographically locked `data/gold/{lang}/test.conll` (200 sentences per language):

| Language | Checkpoint | Precision (P) | Recall (R) | **Entity F1 (Strict)** | **Token Accuracy** | Target (>90%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `final_konkani.pt` | 99.15% | 93.55% | **96.27%** | **99.47%** | >90.0% | **MET (+6.27 pp)** |
| **Assamese** | `final_assamese.pt` | 95.83% | 100.00% | **97.87%** | **99.64%** | >90.0% | **MET (+7.87 pp)** |
| **Sinhala** | `final_sinhala.pt` | 96.32% | 99.24% | **97.76%** | **99.70%** | >90.0% | **MET (+7.76 pp)** |
| **Macro Average** | — | **97.10%** | **97.60%** | **97.30%** | **99.60%** | >90.0% | **MET (+7.30 pp)** |

---

## 2. Granular Per-Entity-Type Performance (Gold-Test)

Strict span-level boundary and class evaluation:

### 2.1 Konkani (`final_konkani.pt`)
| Entity Type | Precision | Recall | **F1 Score** | TP | FP | FN | Gold Count | Pred Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 100.00% | 85.00% | **91.89%** | 17 | 0 | 3 | 20 | 17 |
| **LOC** | 100.00% | 93.33% | **96.55%** | 28 | 0 | 2 | 30 | 28 |
| **ORG** | 97.14% | 91.89% | **94.44%** | 34 | 1 | 3 | 37 | 35 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 37 | 0 | 0 | 37 | 37 |
| **OVERALL** | **99.15%** | **93.55%** | **96.27%** | **116** | **1** | **8** | **124** | **117** |

### 2.2 Assamese (`final_assamese.pt`)
| Entity Type | Precision | Recall | **F1 Score** | TP | FP | FN | Gold Count | Pred Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 86.36% | 100.00% | **92.68%** | 19 | 3 | 0 | 19 | 22 |
| **LOC** | 97.53% | 100.00% | **98.75%** | 79 | 2 | 0 | 79 | 81 |
| **ORG** | 93.33% | 100.00% | **96.55%** | 14 | 1 | 0 | 14 | 15 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 26 | 0 | 0 | 26 | 26 |
| **OVERALL** | **95.83%** | **100.00%** | **97.87%** | **138** | **6** | **0** | **138** | **144** |

### 2.3 Sinhala (`final_sinhala.pt`)
| Entity Type | Precision | Recall | **F1 Score** | TP | FP | FN | Gold Count | Pred Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 94.87% | 97.37% | **96.10%** | 37 | 2 | 1 | 38 | 39 |
| **LOC** | 92.86% | 100.00% | **96.30%** | 39 | 3 | 0 | 39 | 42 |
| **ORG** | 100.00% | 100.00% | **100.00%** | 28 | 0 | 0 | 28 | 28 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 27 | 0 | 0 | 27 | 27 |
| **OVERALL** | **96.32%** | **99.24%** | **97.76%** | **131** | **5** | **1** | **132** | **136** |

---

## 3. Generalization: In-Vocabulary (IV) vs Out-of-Vocabulary (OOV)

| Language | In-Vocab F1 | Out-of-Vocab F1 | In-Vocab Token Acc | Out-of-Vocab Token Acc | Train Vocabulary Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 97.09% | 91.43% | 99.35% | 99.72% | 4,036 |
| **Assamese** | 99.21% | 86.67% | 99.76% | 99.30% | 3,767 |
| **Sinhala** | 98.94% | 95.00% | 99.65% | 99.83% | 4,599 |

---

## 4. Component-Isolation Ablation Benchmarks (Gold-Dev)

Evaluated on locked `data/gold/{lang}/dev.conll` (200 sentences per language):

| Ablation Condition | Konkani Dev F1 | Assamese Dev F1 | Sinhala Dev F1 | **Macro Dev F1** | **Macro $\Delta$** | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Final System (Reference)** | **96.69%** | **98.92%** | **99.24%** | **98.28%** | *ref* | **Authoritative Baseline** |
| (a) No Char-CNN | 90.42% | 97.87% | 88.89% | 92.39% | **-5.89%** | Verified (Data Parity) |
| (b) No Self-Attention (Matched Full Pool) | 89.66% | 95.50% | 89.82% | 91.66% | **-6.62%** | Verified (Data Parity) |
| (c) No CRF (Softmax Cross-Entropy) | 89.47% | 96.53% | 87.37% | 91.12% | **-7.16%** | Verified (Data Parity) |
| (d) No Multilingual Joint Warm-Start | 94.17% | 98.91% | 96.18% | 96.42% | **-1.86%** | Verified (Phase 9) |
| (e) No Self-Training | 96.69% | 98.91% | 98.13% | 97.91% | **-0.37%** | Verified (Phase 9) |
| (f) No Augmentation | 96.69% | 98.91% | 98.13% | 97.91% | **-0.37%** | Verified (Phase 10) |
| (g) Tokenization: Subword BPE | 88.79% | 96.11% | 87.29% | 90.73% | **-7.55%** | Verified (Data Parity) |

---

## 5. Explainability & Faithfulness Benchmarks (Gold-Dev, Symmetric $N=300$)

- **Selected Explainer**: **Raw Attention (`raw_attention`)**
- Mean Faithfulness Gap ($k=1$): **-0.0523** (Raw Attention) vs. **-0.0529** (Gradient $\times$ Attention)

| Language | Method | Top-1 Drop | Rand-1 Drop | **Gap ($k=1$)** | Win Rate ($k=1$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Konkani** | Gradient $\times$ Attention | 0.0015 | 0.0354 | -0.0339 | 15.0% |
| **Konkani** | **Raw Attention** *(Winner)* | 0.0018 | 0.0354 | **-0.0336** | **19.0%** |
| **Assamese** | Gradient $\times$ Attention | 0.0000 | 0.0058 | -0.0058 | 24.0% |
| **Assamese** | **Raw Attention** *(Winner)* | 0.0000 | 0.0058 | **-0.0058** | **30.0%** |
| **Sinhala** | Gradient $\times$ Attention | 0.0000 | 0.0296 | -0.0296 | 12.0% |
| **Sinhala** | **Raw Attention** *(Winner)* | 0.0007 | 0.0296 | **-0.0290** | **23.0%** |

---

## 6. Target Comparison Instructions for Downstream Phases

1. **Phases 1–7 (Reliability Overhaul)**:
   - Target problem: Entity misses (Konkani: 8 FN, Sinhala: 1 FN, Assamese: 0 FN).
   - Any proposed intervention (threshold tuning, boundary calibration, loss weighting) must evaluate against `final_{lang}.pt` on locked gold splits and prove non-regression on Precision and Macro F1 ($\ge 97.30\%$).
2. **Phases 8–17 (Kokborok Integration Track)**:
   - Evaluates whether extending the multilingual architecture to a 4th language (Kokborok) preserves the three-language baseline without negative transfer (Konkani $\ge 96.27\%$, Assamese $\ge 97.87\%$, Sinhala $\ge 97.76\%$).
