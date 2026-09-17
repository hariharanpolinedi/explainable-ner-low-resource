# Comprehensive Empirical Results Tables

This document compiles the complete, camera-ready empirical results of the project across all 16 phases. All evaluations were conducted strictly in compliance with project governance: **zero external pretrained models, tokenizers, or embeddings**, and the gold-test splits were touched **exactly once** under cryptographic lock.

---

## Table 1: Final Official Gold-Test Performance (Phase 14)

Evaluated on the locked gold-test splits (`data/gold/{lang}/test.conll`, 200 sentences per language). Primary metric is strict entity-level span F1 (seqeval exact boundary and type matching); secondary metric is token-level classification accuracy.

| Language | Script Family | Checkpoint | Precision | Recall | **Entity F1** (Primary) | **Token Acc** (Secondary) | Research Target | Target Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | Devanagari | `checkpoints/final_konkani.pt` | 99.15% | 93.55% | **96.27%** | 99.47% | >90.0% | **MET (+6.27 pp)** |
| *Konkani (Joint Baseline)* | Devanagari | `checkpoints/joint_best.pt` | 98.31% | 93.55% | 95.87% | 99.41% | >90.0% | Baseline |
| **Assamese** | Bengali-Assamese | `checkpoints/final_assamese.pt` | 95.83% | 100.00% | **97.87%** | 99.64% | >90.0% | **MET (+7.87 pp)** |
| *Assamese (Joint Baseline)* | Bengali-Assamese | `checkpoints/joint_best.pt` | 96.48% | 99.28% | 97.86% | 99.61% | >90.0% | Baseline |
| **Sinhala** | Sinhala | `checkpoints/final_sinhala.pt` | 96.32% | 99.24% | **97.76%** | 99.70% | >90.0% | **MET (+7.76 pp)** |
| *Sinhala (Joint Baseline)* | Sinhala | `checkpoints/joint_best.pt` | 94.93% | 99.24% | 97.04% | 99.60% | >90.0% | Baseline |
| **Macro Average (Final)** | — | — | **97.10%** | **97.60%** | **97.30%** | **99.60%** | **>90.0%** | **ALL MET** |

---

## Table 2: Per-Entity-Type Performance Breakdown on Gold-Test (Phase 14)

Span-level precision (P), recall (R), and F1 score per semantic entity category, alongside raw confusion counts (True Positives, False Positives, False Negatives).

| Language | Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Gold Spans | Pred Spans |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **PER** | 100.00% | 85.00% | **91.89%** | 17 | 0 | 3 | 20 | 17 |
| | **LOC** | 100.00% | 93.33% | **96.55%** | 28 | 0 | 2 | 30 | 28 |
| | **ORG** | 97.14% | 91.89% | **94.44%** | 34 | 1 | 3 | 37 | 35 |
| | **MISC** | 100.00% | 100.00% | **100.00%** | 37 | 0 | 0 | 37 | 37 |
| **Assamese** | **PER** | 86.36% | 100.00% | **92.68%** | 19 | 3 | 0 | 19 | 22 |
| | **LOC** | 97.53% | 100.00% | **98.75%** | 79 | 2 | 0 | 79 | 81 |
| | **ORG** | 93.33% | 100.00% | **96.55%** | 14 | 1 | 0 | 14 | 15 |
| | **MISC** | 100.00% | 100.00% | **100.00%** | 26 | 0 | 0 | 26 | 26 |
| **Sinhala** | **PER** | 94.87% | 97.37% | **96.10%** | 37 | 2 | 1 | 38 | 39 |
| | **LOC** | 92.86% | 100.00% | **96.30%** | 39 | 3 | 0 | 39 | 42 |
| | **ORG** | 100.00% | 100.00% | **100.00%** | 28 | 0 | 0 | 28 | 28 |
| | **MISC** | 100.00% | 100.00% | **100.00%** | 27 | 0 | 0 | 27 | 27 |

---

## Table 3: Out-of-Vocabulary (OOV) Generalization Analysis on Gold-Test (Phase 14)

Measuring the subword generalization capacity of the Char-CNN and Word representation on previously unseen lexical items compared to in-vocabulary training words.

| Language | In-Vocab F1 | Out-of-Vocab F1 | In-Vocab Acc | Out-of-Vocab Acc | Training Vocab Size | Reference Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | 97.09% | 91.43% | 99.35% | 99.72% | 4,036 | `data/gold/konkani/train.conll` |
| **Assamese** | 99.21% | 86.67% | 99.76% | 99.30% | 3,767 | `data/gold/assamese/train.conll` |
| **Sinhala** | 98.94% | 95.00% | 99.65% | 99.83% | 4,599 | `data/gold/sinhala/train.conll` |
| **Macro Average** | **98.41%** | **91.03%** | **99.59%** | **99.62%** | — | — |

---

## Table 4: Master Component Ablation Suite (Phase 12)

Systematic evaluation on gold-dev isolating the marginal contribution ($\Delta$ F1) of each architectural module, training phase, and tokenization scheme by removing exactly one component from the finalized pipeline.

| Model Configuration / Ablation Condition | Konkani Dev F1 | Konkani $\Delta$ | Assamese Dev F1 | Assamese $\Delta$ | Sinhala Dev F1 | Sinhala $\Delta$ | Macro Dev F1 | Macro $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Final Pipeline (Locked)** | **96.69%** | ref | **98.92%** | ref | **99.24%** | ref | **98.28%** | ref |
| **(a) No Char-CNN Word Encoder** | 87.66% | -9.03% | 88.89% | -10.03% | 83.20% | -16.04% | 86.58% | **-11.70%** |
| **(b) No Self-Attention (BiLSTM-CRF only)** | 58.58% | -38.11% | 72.59% | -26.33% | 51.45% | -47.79% | 60.87% | **-37.41%** |
| **(c) No CRF (Softmax Cross-Entropy)** | 83.19% | -13.50% | 83.92% | -15.00% | 76.11% | -23.13% | 81.07% | **-17.21%** |
| **(d) No Multilingual Joint Warm-Start** | 94.17% | -2.52% | 98.91% | -0.01% | 96.18% | -3.06% | 96.42% | **-1.86%** |
| **(e) No Iterative Self-Training** | 96.69% | 0.00% | 98.91% | -0.01% | 98.13% | -1.11% | 97.91% | **-0.37%** |
| **(f) No Targeted Data Augmentation** | 96.69% | 0.00% | 98.91% | -0.01% | 98.13% | -1.11% | 97.91% | **-0.37%** |
| **(g) Tokenization: Subword BPE vs Word+Char** | 86.08% | -10.61% | 87.29% | -11.63% | 81.70% | -17.54% | 85.02% | **-13.26%** |

### Cumulative Progression Across Engineering Phases (Dev F1)

| Milestone / Phase | Core Architectural / Methodological Addition | Konkani | Assamese | Sinhala | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 5** | Baseline Word BiLSTM-CRF | 89.45% | 98.91% | 90.69% | 93.02% |
| **Phase 6** | + Character-Level CNN Encoder (Grapheme Clusters) | 95.00% | 98.91% | 98.11% | 97.34% |
| **Phase 7** | + Multi-Head Self-Attention Layer ($H=4$, $d_k=64$) | 95.00% | 98.91% | 97.74% | 97.22% |
| **Phase 8** | + Multilingual Joint Warm-Start (16d Lang Conditioning) | 96.30% | 98.91% | 98.50% | 97.90% |
| **Phase 9** | + Layer-Unfreezing Fine-Tuning Specialization | 96.69% | 98.91% | 98.13% | 97.91% |
| **Phase 10** | + CRF Sequence Confidence Self-Training | 96.69% | 98.91% | 98.13% | 97.91% |
| **Phase 11 (Final)** | + Targeted Entity Substitution & Class Loss | **96.69%** | **98.92%** | **99.24%** | **98.28%** |

---

## Table 5: Quantitative Token-Erasure Faithfulness Evaluation (Phase 13)

Evaluated across $\ge 100$ entity predictions per language on gold-dev. Confidence drop measures the decrease in predicted tag probability when top-$k$ attributed context tokens are erased vs. random non-attributed tokens. Faithfulness Gap ($\Delta = \text{Drop}_{\text{top}} - \text{Drop}_{\text{rand}}$) measures causal necessity.

| Language | Attribution Method | Top-1 Drop | Rand-1 Drop | **Gap ($k=1$)** | Top-2 Drop | Rand-2 Drop | **Gap ($k=2$)** | Top-3 Drop | Rand-3 Drop | **Gap ($k=3$)** | Win Rate ($k=1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **Gradient × Attention** | 0.0015 | 0.0910 | **-0.0894** | 0.0013 | 0.1754 | **-0.1741** | 0.0099 | 0.2871 | **-0.2772** | 14.0% |
| | Raw Attention | 0.0018 | 0.0910 | **-0.0892** | 0.0019 | 0.1754 | **-0.1734** | 0.0115 | 0.2871 | **-0.2757** | 17.0% |
| **Assamese** | **Gradient × Attention** | 0.0000 | 0.0961 | **-0.0960** | 0.0000 | 0.2563 | **-0.2563** | 0.0007 | 0.2909 | **-0.2901** | 21.0% |
| | Raw Attention | 0.0000 | 0.0961 | **-0.0960** | 0.0000 | 0.2563 | **-0.2562** | 0.0008 | 0.2909 | **-0.2901** | 24.0% |
| **Sinhala** | **Gradient × Attention** | 0.0000 | 0.0576 | **-0.0576** | 0.0003 | 0.2011 | **-0.2008** | 0.0103 | 0.3866 | **-0.3763** | 11.0% |
| | Raw Attention | 0.0007 | 0.0576 | **-0.0569** | 0.0008 | 0.2011 | **-0.2003** | 0.0120 | 0.3866 | **-0.3746** | 22.0% |
| **Mean Metrics** | **Gradient × Attention** | 0.0005 | 0.0816 | **-0.0810** | 0.0005 | 0.2109 | **-0.2104** | 0.0070 | 0.3215 | **-0.3145** | 15.3% |
| | Raw Attention | 0.0008 | 0.0816 | **-0.0807** | 0.0009 | 0.2109 | **-0.2100** | 0.0081 | 0.3215 | **-0.3135** | 21.0% |

- **Mean Overall Faithfulness Gap (Grad × Attn)**: **-0.2020**
- **Mean Overall Faithfulness Gap (Raw Attention)**: **-0.2014**
