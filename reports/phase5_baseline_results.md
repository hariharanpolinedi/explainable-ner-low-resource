# Phase 5 Baseline Model (Word-Level BiLSTM-CRF) Results Report

**Date**: 2026-09-16  
**Architecture**: Random Word Embeddings (100d, scratch) -> BiLSTM (128 hidden) -> Linear-Chain CRF  
**Hardware Device**: `NVIDIA GeForce RTX 3050 Laptop GPU`  
**Primary Metric**: Entity-Level F1 (seqeval strict span matching)  
**Secondary Metric**: Token-Level Accuracy  

---

## 1. Experiment A: Data Condition Comparison (Gold-Only vs Gold+Silver)

Comparison of training exclusively on small human gold-train seed (600 sentences) vs combined gold-train + silver weak-supervision data (7,800 sentences total per language).

| Language | Condition | Training Sents | Best Dev Entity F1 | Dev Entity Prec | Dev Entity Rec | Dev Token Acc | Epochs | Duration (s) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `Gold Only` | 600 | **92.05%** | 96.49% | 88.00% | 98.99% | 19 | 42.03s | Baseline |
| **Konkani** | `Gold + Silver` | 7,800 | **94.17%** | 98.26% | 90.40% | 99.34% | 10 | 193.13s | **WINNER** |
| **Assamese** | `Gold Only` | 600 | **90.04%** | 92.42% | 87.77% | 98.94% | 26 | 53.22s | Baseline |
| **Assamese** | `Gold + Silver` | 7,800 | **98.55%** | 99.27% | 97.84% | 99.83% | 10 | 216.16s | **WINNER** |
| **Sinhala** | `Gold Only` | 600 | **93.28%** | 91.24% | 95.42% | 99.36% | 30 | 67.73s | Baseline |
| **Sinhala** | `Gold + Silver` | 7,800 | **95.88%** | 94.12% | 97.71% | 99.58% | 16 | 335.9s | **WINNER** |

---

## 2. Experiment B: Architecture Depth Comparison (1-Layer vs 2-Layer BiLSTM)

Comparison of 1-layer vs 2-layer BiLSTM encoder depth on the winning data condition from Experiment A.

| Language | Winning Condition | BiLSTM Layers | Best Dev Entity F1 | Dev Entity Prec | Dev Entity Rec | Dev Token Acc | Epochs | Duration (s) | Best Model |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `gold_plus_silver` | 1 | **94.17%** | 98.26% | 90.40% | 99.34% | 10 | 193.13s |  |
| **Konkani** | `gold_plus_silver` | 2 | **95.04%** | 98.29% | 92.00% | 99.34% | 26 | 539.77s | **SELECTED** |
| **Assamese** | `gold_plus_silver` | 1 | **98.55%** | 99.27% | 97.84% | 99.83% | 10 | 216.16s |  |
| **Assamese** | `gold_plus_silver` | 2 | **98.91%** | 100.00% | 97.84% | 99.86% | 10 | 237.92s | **SELECTED** |
| **Sinhala** | `gold_plus_silver` | 1 | **95.88%** | 94.12% | 97.71% | 99.58% | 16 | 335.9s |  |
| **Sinhala** | `gold_plus_silver` | 2 | **96.99%** | 95.56% | 98.47% | 99.73% | 19 | 484.59s | **SELECTED** |

---

## 3. Per-Entity-Type Baseline Breakdown (Selected Checkpoints)

| Language | Model Checkpoint | Entity Type | Precision | Recall | F1-Score | Gold Entities |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `checkpoints/baseline_konkani.pt` | `PER` | 100.00% | 85.00% | **91.89%** | 20 |
| **Konkani** | `checkpoints/baseline_konkani.pt` | `LOC` | 96.43% | 87.10% | **91.53%** | 31 |
| **Konkani** | `checkpoints/baseline_konkani.pt` | `ORG` | 97.14% | 91.89% | **94.44%** | 37 |
| **Konkani** | `checkpoints/baseline_konkani.pt` | `MISC` | 100.00% | 100.00% | **100.00%** | 37 |
| **Assamese** | `checkpoints/baseline_assamese.pt` | `PER` | 100.00% | 100.00% | **100.00%** | 18 |
| **Assamese** | `checkpoints/baseline_assamese.pt` | `LOC` | 100.00% | 97.47% | **98.72%** | 79 |
| **Assamese** | `checkpoints/baseline_assamese.pt` | `ORG` | 100.00% | 93.33% | **96.55%** | 15 |
| **Assamese** | `checkpoints/baseline_assamese.pt` | `MISC` | 100.00% | 100.00% | **100.00%** | 27 |
| **Sinhala** | `checkpoints/baseline_sinhala.pt` | `PER` | 92.31% | 97.30% | **94.74%** | 37 |
| **Sinhala** | `checkpoints/baseline_sinhala.pt` | `LOC` | 92.68% | 97.44% | **95.00%** | 39 |
| **Sinhala** | `checkpoints/baseline_sinhala.pt` | `ORG` | 100.00% | 100.00% | **100.00%** | 28 |
| **Sinhala** | `checkpoints/baseline_sinhala.pt` | `MISC` | 100.00% | 100.00% | **100.00%** | 27 |

---

## 4. Key Findings and Analysis

- **Impact of Silver Weak Supervision**: Incorporating Phase 4 silver data (7,200 sentences per language) consistently boosted dev Entity F1 over the gold-only baseline across all three languages (+2.12% for Konkani, +8.51% for Assamese, +2.60% for Sinhala), confirming that the high-precision weak supervision signal provides vital contextual vocabulary expansion for from-scratch embeddings.
- **Impact of Encoder Depth**: The 2-layer BiLSTM consistently outperformed the 1-layer BiLSTM across all three languages on the winning Gold+Silver condition (Konkani: 95.04% vs 94.17%; Assamese: 98.91% vs 98.55%; Sinhala: 96.99% vs 95.88%), providing richer hierarchical context representation for subsequent CRF transition modeling.
- **Floor Reference Point**: These documented F1 figures establish the definitive empirical floor that Phase 6 (Character-Level CNN) and Phase 7 (Self-Attention) must surpass.
- **Class Imbalance Note**: Inverse-frequency class weighting experimentation is preserved as a documented extension for Phase 11 tuning, as the unweighted CRF transition matrix already learned strong entity boundary segmentation under the expanded silver corpus.

