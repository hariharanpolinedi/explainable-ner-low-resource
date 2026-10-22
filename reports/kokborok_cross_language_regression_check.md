# Cross-Language Regression Verification Report

**Phase**: Post-Phase 14 Verification / Integrity Audit  
**Auditor**: Senior ML/NLP Research Engineer  
**Date**: 2026-09-30  
**Device**: `NVIDIA GeForce RTX 3050 Laptop GPU` (CUDA 12.8 / PyTorch 2.6.0)  
**Evaluation Scope**: Official `gold-dev` splits for Konkani, Assamese, and Sinhala (Zero Test Touch)  
**Status**: **NEGATIVE TRANSFER CONFIRMED — "ZERO REGRESSION" CLAIM REFUTED**

---

## 1. Executive Summary

Previous reports claimed:
> *"The Option-C protected backbone ensures zero regression on existing languages... Konkani, Assamese, and Sinhala heads remained strictly frozen with zero regression."*

This audit empirically tested that assertion by evaluating all four languages on their official `gold-dev` splits using the actual checkpoints from each optimization round.

### Key Finding
The claim of "zero regression" is **empirically false**. While the language-specific classification and CRF heads (`fc_heads.{0,1,2}` and `crf_heads.{0,1,2}`) were frozen, the **shared BiLSTM encoder and word embedding layers were NOT frozen**. Backpropagating gradients from Kokborok's Latin script caused severe architectural interference, resulting in:
* **Sinhala Dev F1 plunged by -19.40 pp** (from 97.38% down to 77.98%)
* **Konkani Dev F1 dropped by -6.65 pp** (from 96.69% down to 90.04%)
* **Assamese Dev F1 dropped by -4.25 pp** (from 98.55% down to 94.30%)

---

## 2. Empirical Performance Comparison on Official Gold-Dev

*Evaluated on GPU using identical tokenization, vocabulary mappings, and strict seqeval scoring. Zero test set exposure.*

| Language | Script | Phase 14 Protected Baseline | Round 1 Checkpoint (`kokborok_improved_v2.pt`) | Round 4 Checkpoint (`kokborok_calib_best.pt`) | **Delta vs Baseline ($\Delta$)** | Regression Status |
|:--- |:--- |:---:|:---:|:---:|:---:|:---:|
| **Konkani (`kok`)** | Devanagari | **96.69%** | 90.16% | 90.04% | **-6.65 pp** | ⚠️ Moderate Regression |
| **Assamese (`asm`)** | Eastern Nagari | **98.55%** | 97.08% | 94.30% | **-4.25 pp** | ⚠️ Minor Regression |
| **Sinhala (`sin`)** | Sinhala | **97.38%** | 91.70% | 77.98% | **-19.40 pp** | 🚨 **Severe Degradation** |
| **MACRO-AVERAGE** | — | **97.54%** | **92.98%** | **87.44%** | **-10.10 pp** | 🚨 **Target Lost (<90%)** |

---

## 3. Root-Cause Architectural Failure Analysis

In Phase 12, the architecture decision for **Option (c) Staged / Modular Integration** was established to prevent the catastrophic negative transfer observed under Option (a) Joint Training (where unconstrained joint training plunged Sinhala from 96.30% to 86.87%).

However, inspecting the training script `training/kokborok_improvement_pipeline.py` (lines 727–734) reveals why Option (c) failed during the Kokborok improvement rounds:

```python
# Lines 727-734 in training/kokborok_improvement_pipeline.py:
for name, param in model.named_parameters():
    # Freeze other languages' CRF and FC heads (indices 0, 1, 2)
    if any(f"fc_heads.{i}" in name or f"crf_heads.{i}" in name for i in [0, 1, 2]):
        param.requires_grad = False
    else:
        param.requires_grad = True
```

### The Flaw
* Only `fc_heads.{0,1,2}` and `crf_heads.{0,1,2}` were set to `requires_grad = False`.
* All shared layers:
  * `bilstm.weight_ih_l*` and `bilstm.weight_hh_l*` (Shared BiLSTM)
  * `word_embedding.embedding.weight` (Shared Lexical Space)
  * `char_cnn.*` (Shared Character Convolutions)
  * `attention_blocks.*` (Shared Multi-Head Attention)
  remained fully trainable (`requires_grad = True`).
* During the 20 epochs of Round 1 (and the subsequent fine-tuning passes), **9,700+ synthetic Kokborok sentences** updated the shared weights.
* Because Kokborok uses Roman (Latin) script with completely distinct phonotactics, affixation patterns, and vocabulary, these gradient updates substantially distorted the shared latent representations for Brahmic-derived scripts (Devanagari, Eastern Nagari, Sinhala).
* Sinhala was damaged most severely (-19.40 pp) because its complex conjunct representations and zero-width joiner dependencies were erased in the shared character and subword spaces.

---

## 4. Master Cross-Language Benchmark Status

| Language | Official Locked Test F1 (Phase 14) | True Status Under Deployed Kokborok Model |
|---|:---:|:---|
| **Konkani** | 96.27% | **Degraded** (~90.0% dev proxy) |
| **Assamese** | 97.87% | **Degraded** (~94.3% dev proxy) |
| **Sinhala** | 97.76% | **Severely Compromised** (~78.0% dev proxy) |
| **Kokborok** | 68.97% (N=16 test) | **Overfitted / Non-Generalizing** (0% fresh F1) |

---

## 5. Corrective Recommendation

1. **Immediate Rollback**: The shared checkpoints `final_kokborok.pt` and `deployment/bundles/kokborok/v1/model.pt` must not be presented as a valid integrated checkpoint across all 4 languages.
2. **True Option (c) Architecture**: To train Kokborok without damaging the other three languages, the **entire shared encoder (BiLSTM, CharCNN, Attention, Word Embeddings) MUST be frozen** (`param.requires_grad = False`), and only `fc_heads.3` and `crf_heads.3` may receive gradient updates.
3. **Academic Reporting**: Any research report or paper submission must transparently document this negative transfer experiment as empirical proof of the strict necessity of end-to-end modular freezing when integrating typologically distant scripts.
