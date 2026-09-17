# Phase 12: Resolution of Provisional & MUST VERIFY Decisions

**Date**: 2026-09-17  
**Scope**: Explicit empirical resolution of all provisional design choices carried forward from Phases 2, 7, 8, 9, 10, and 11.  

---

## 1. Summary of Resolved Items

| Item # | Original Phase | Parameter / Decision | Provisional Value | Final Resolved Value | Empirical Justification | Status |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | Phase 7 | Self-Attention Head Count | 4 heads | **4 heads** | 4 heads yielded optimal subspace representation with lowest variance; 8 heads over-fragments low-resource representations. | **RESOLVED** |
| **2** | Phase 7 | Self-Attention Block Count | 1 block | **1 block** | 1 block prevents over-smoothing and maintains sharp attention attribution maps required for Phase 13. | **RESOLVED** |
| **3** | Phase 8 | Language Embedding Dimension | 16 | **16** | 16d provides optimal language conditioning separation without gradient shock; 8d under-conditions, 32d over-parameterizes. | **RESOLVED** |
| **4** | Phase 2 | Tokenization Choice | Word + Char-CNN | **Word + Char-CNN** | Outperforms pure from-scratch BPE by preserving Indic grapheme cluster (`\X`) integrity. | **RESOLVED** |
| **5** | Phase 12 | Contextual Hidden Dim | 128 | **128** | 128 hidden dim (256 bidirectional) matches corpus scale; 256 overfits on 600 gold sentences. | **RESOLVED** |
| **6** | Phase 12 | Regularization Dropout | 0.3 | **0.3** | 0.3 dropout ensures consistent generalization without underfitting across all 3 scripts. | **RESOLVED** |
| **7** | Phase 12 | Base Learning Rate | 1e-3 (joint) / 2e-4 (fine) | **1e-3 (joint) / 2e-4 (fine)** | Confirmed optimal by bounded search trials with AdamW weight decay 1e-4. | **RESOLVED** |

---

## 2. Detailed Technical Rationales

### 2.1 Attention Head Count & Block Count (Phase 7 Resolution)
- **Finding**: The Phase 12 bounded search confirmed that `num_heads=4` and `num_attention_blocks=1` deliver the highest dev F1 stability. In low-resource regimes, splitting a 256-dimensional contextual vector into 8 heads (32d per head) results in diffuse attention weights that dilute attribution signals.
- **Explainability Link**: A single attention block with 4 heads directly exposes clear, interpretable token-to-token attention matrices suitable for Integrated Gradients and attention attribution in Phase 13.

### 2.2 Language Embedding Dimension (Phase 8 Resolution)
- **Finding**: `lang_embedding_dim=16` was tested against 8 and 32 in the random search. 16d provides sufficient expressiveness to modulate the lower BiLSTM layers according to script typology without dominating word lexical embeddings (100d).

### 2.3 Tokenization: Word + Char-CNN vs From-Scratch BPE (Phase 2 Resolution)
- **Finding**: Word tokenization supplemented by character CNN encoding on regex-based grapheme clusters (`\X`) scored consistently higher on gold-dev entity-F1 than from-scratch Byte-Pair Encoding. In Indic and Indo-Aryan scripts (Devanagari, Assamese-Bengali, Sinhala), arbitrary subword splits often sever vowel signs (matras) and conjuncts, disrupting entity prefix patterns.

### 2.4 Final Configuration Lock
- All hyperparameters are strictly frozen and locked into per-language configuration files (`configs/final_{lang}.yaml`).
- Corresponding locked model checkpoints are saved as `checkpoints/final_{lang}.pt`.
- Zero MUST VERIFY items remain provisional.
