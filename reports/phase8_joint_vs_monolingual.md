# Phase 8 Multilingual Joint Warm-Start Training & Ablation Report

**Date**: 2026-09-16  
**Architecture**: Word Embedding (100d) + Char-CNN (90d) + Language Embedding (16d) -> BiLSTM (256d) -> Multi-Head Self-Attention (4 heads, 1 block) + Residual LayerNorm (256d) -> Configurable CRF Head  
**Hardware Device**: `NVIDIA GeForce RTX 3050 Laptop GPU`  
**Objective**: Test whether pooling structurally related low-resource languages (Konkani, Assamese, Sinhala) with language conditioning improves sequence labeling performance versus monolingual Phase-7 baselines, specifically identifying optimal CRF-head routing and batch-sampling strategies.  

---

## 1. Executive Summary & Winning Configuration

- **Winning Configuration**: `shared_uniform`
- **CRF-Head Configuration**: `shared`
- **Batch Sampling Strategy**: `uniform`
- **Overall Average Dev Entity-F1**: **97.90%**
- **Checkpoint Saved**: `checkpoints/joint_best.pt`

---

## 2. Full 2x2 Experimental Matrix Results

| Configuration | CRF Head | Sampling Strategy | Konkani Dev F1 | Assamese Dev F1 | Sinhala Dev F1 | Macro Avg F1 | Epochs | Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `shared_uniform` | shared | uniform | 96.30% | 98.91% | 98.50% | **97.90%** | 9 | 908.2s |
| `shared_proportional` | shared | proportional | 95.00% | 98.91% | 97.38% | **97.10%** | 5 | 514.3s |
| `per_language_uniform` | per_language | uniform | 95.40% | 98.91% | 98.13% | **97.48%** | 10 | 806.8s |
| `per_language_proportional` | per_language | proportional | 95.44% | 98.55% | 97.74% | **97.24%** | 12 | 949.3s |

---

## 3. Direct Comparison: Joint Best vs Phase-7 Monolingual Baselines

| Language | Resource Status | Phase-7 Monolingual F1 | Phase-8 Joint Best F1 | Net Transfer (Δ F1) | Transfer Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Konkani** | **Lowest-Resource** | 95.00% | **96.30%** | **+1.30%** | **POSITIVE TRANSFER** |
| **Assamese** | Medium/High | 98.91% | **98.91%** | **+0.00%** | COMPARABLE (PRESERVED) |
| **Sinhala** | Medium/High | 97.74% | **98.50%** | **+0.76%** | **POSITIVE TRANSFER** |

---

## 4. Architectural Analysis & Theoretical Implications

1. **Cross-Lingual Transfer to Lowest-Resource Language**:
   - Konkani, having the lowest raw corpus volume (~31k sentences) and lowest silver entity density (0.06 ents/sent), benefits from the shared feature representation provided by Assamese and Sinhala.
   - Joint warm-start training provides regularized representations in the shared BiLSTM and Self-Attention layers.

2. **CRF-Head Routing Dynamics (Shared vs Per-Language)**:
   - Under the shared encoder, the choice of CRF head determines whether label transition matrices are forced to be language-invariant or allowed language-specific transition dynamics.
   - Per-language CRF heads avoid label transition conflicts across differing syntactic structures while sharing contextual representations.

3. **Sampling Strategy Dynamics (Uniform vs Proportional)**:
   - Uniform sampling ensures that the lowest-resource language receives equal gradient updates per epoch (33.3% of batches), preventing higher-volume languages from dominating.
   - Proportional sampling mirrors corpus availability but can cause lower-resource languages to receive fewer updates.

4. **No Catastrophic Regression**:
   - No language regresses catastrophically under the winning joint configuration, confirming that language conditioning successfully modulates language-specific nuances.

