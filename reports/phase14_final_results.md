# Phase 14: Final Gold-Test Evaluation & Target Performance Claims

## Executive Summary & Headline Results

This document records the **single, unbiased, official evaluation** of the locked Phase 12 final checkpoints
against the locked gold-test splits (`data/gold/{lang}/test.conll`). These numbers represent the definitive,
reproducible performance claims of this project against the **>90% entity-level F1** research target.

> [!IMPORTANT]
> **Single-Evaluation Discipline**: Gold-test splits were touched exactly once in this phase.
> No model tuning, checkpoint selection, threshold adjustment, or retraining was performed.
> All evaluation tools strictly adhere to the no-pretrained-models constraint.

### Headline Results Table

| Language | Model Checkpoint | Entity Prec (P) | Entity Rec (R) | **Entity F1** (Primary) | **Token Acc** (Secondary) | Target (>90%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `checkpoints/final_konkani.pt` | 99.15% | 93.55% | **96.27%** | 99.47% | >90.0% | **MET** |
| *Konkani (Joint Baseline)* | `checkpoints/joint_best.pt` | 98.31% | 93.55% | 95.87% | 99.41% | >90.0% | Baseline |
| **Assamese** | `checkpoints/final_assamese.pt` | 95.83% | 100.00% | **97.87%** | 99.64% | >90.0% | **MET** |
| *Assamese (Joint Baseline)* | `checkpoints/joint_best.pt` | 96.48% | 99.28% | 97.86% | 99.61% | >90.0% | Baseline |
| **Sinhala** | `checkpoints/final_sinhala.pt` | 96.32% | 99.24% | **97.76%** | 99.70% | >90.0% | **MET** |
| *Sinhala (Joint Baseline)* | `checkpoints/joint_best.pt` | 94.93% | 99.24% | 97.04% | 99.60% | >90.0% | Baseline |

---

## Trajectory: Improvement from Joint Baseline to Final Specialized Model

| Language | Phase 8 Joint F1 | Phase 12 Final F1 | **F1 Delta (pp)** | Prec Delta (pp) | Rec Delta (pp) | Token Acc Delta (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 95.87% | 96.27% | **+0.40** | +0.84 | +0.00 | +0.06 |
| **Assamese** | 97.86% | 97.87% | **+0.01** | -0.65 | +0.72 | +0.03 |
| **Sinhala** | 97.04% | 97.76% | **+0.72** | +1.39 | +0.00 | +0.10 |

---

## Detailed Per-Entity-Type Breakdown

Strict span-level precision, recall, F1, and raw span counts per semantic entity type:

### Konkani (Final Model)

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 100.00% | 85.00% | **91.89%** | 17 | 0 | 3 | 20 | 17 |
| **LOC** | 100.00% | 93.33% | **96.55%** | 28 | 0 | 2 | 30 | 28 |
| **ORG** | 97.14% | 91.89% | **94.44%** | 34 | 1 | 3 | 37 | 35 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 37 | 0 | 0 | 37 | 37 |

### Assamese (Final Model)

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 86.36% | 100.00% | **92.68%** | 19 | 3 | 0 | 19 | 22 |
| **LOC** | 97.53% | 100.00% | **98.75%** | 79 | 2 | 0 | 79 | 81 |
| **ORG** | 93.33% | 100.00% | **96.55%** | 14 | 1 | 0 | 14 | 15 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 26 | 0 | 0 | 26 | 26 |

### Sinhala (Final Model)

| Entity Class | Precision | Recall | **F1 Score** | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Total Gold | Total Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 94.87% | 97.37% | **96.10%** | 37 | 2 | 1 | 38 | 39 |
| **LOC** | 92.86% | 100.00% | **96.30%** | 39 | 3 | 0 | 39 | 42 |
| **ORG** | 100.00% | 100.00% | **100.00%** | 28 | 0 | 0 | 28 | 28 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 27 | 0 | 0 | 27 | 27 |

---

## Out-Of-Vocabulary (OOV) vs In-Vocabulary (IV) Split Analysis

Evaluating generalization beyond lexical memorization using the training corpus vocabulary as reference:

| Language | In-Vocab F1 | Out-of-Vocab F1 | In-Vocab Accuracy | Out-of-Vocab Accuracy | Train Vocab Size | Reference Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | 97.09% | 91.43% | 99.35% | 99.72% | 4,036 | `data/gold/konkani/train.conll` |
| **Assamese** | 99.21% | 86.67% | 99.76% | 99.30% | 3,767 | `data/gold/assamese/train.conll` |
| **Sinhala** | 98.94% | 95.00% | 99.65% | 99.83% | 4,599 | `data/gold/sinhala/train.conll` |

---

## Target Performance Analysis & Discussion

### Konkani: Target MET (Entity F1 = 96.27%)

- **Target**: >90.0% Entity F1
- **Actual Gold-Test F1**: **96.27%**
- **Margin / Shortfall**: +6.27 percentage points

#### Analysis & Correlation with Earlier Phases:
- **Corpus Scale**: Konkani possessed the smallest raw corpus (30,980 sentences in Phase 1), yet achieved strong test performance due to high silver precision (100% in Phase 4) and synergistic transfer from the joint multilingual backbone in Phase 8.
- **Dev vs Test Gap**: Test performance closely mirrors dev-set trajectory with no catastrophic degradation, demonstrating reliable generalization.

### Assamese: Target MET (Entity F1 = 97.87%)

- **Target**: >90.0% Entity F1
- **Actual Gold-Test F1**: **97.87%**
- **Margin / Shortfall**: +7.87 percentage points

#### Analysis & Correlation with Earlier Phases:
- **Corpus Scale & Morphological Richness**: Assamese (55,002 raw sentences) demonstrated exceptional stability across both high-frequency and low-frequency entities. Self-training (Phase 10) and fine-tuning (Phase 9) expanded vocabulary coverage effectively.
- **Dev vs Test Gap**: Generalization on gold-test confirms that character-CNN representations adequately captured eastern Indic inflectional suffixes.

### Sinhala: Target MET (Entity F1 = 97.76%)

- **Target**: >90.0% Entity F1
- **Actual Gold-Test F1**: **97.76%**
- **Margin / Shortfall**: +7.76 percentage points

#### Analysis & Correlation with Earlier Phases:
- **Script & Dialectal Distance**: Sinhala uses an entirely distinct southern Brahmic script and grammar compared to Indo-Aryan Konkani and Assamese. Multilingual pooling provided regularization while Phase 9 specialization and Phase 11 augmentation reinforced rare entity recovery.
- **Dev vs Test Gap**: Test performance exhibits consistent high precision across all four entity types, validating the self-attention and CRF decoding mechanism.

---

## Hardware & Evaluation Metadata

- **Timestamp (UTC)**: `2026-09-17T05:03:59.541563+00:00`
- **Device Used**: `NVIDIA GeForce RTX 3050 Laptop GPU` (`cuda`)
- **Single-Run Guard Verified**: `OFFICIAL_SINGLE_EVALUATION` (Force Re-run: `False`)
- **Gold-Test Cryptographic Checksums Verified**: PASS for Konkani, Assamese, and Sinhala.