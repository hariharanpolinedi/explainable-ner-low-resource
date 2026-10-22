# Phase 12 Mandatory Inspection & Language-Conditioning Extension Plan

**Phase**: 12 — Kokborok Integration-Strategy Experimentation  
**Date**: 2026-09-25  
**Auditor/Engineer**: Senior ML/NLP Research Engineer  
**Status**: INSPECTION COMPLETE & VERIFIED  

---

## 1. Executive Summary: Mandatory Workspace Inspection

In strict adherence to the Phase 12 specification and the explicit cycle-close update from Reliability Cycle 2:

### 1.1 Current-Best 3-Language Checkpoints (Reliability Track Output)
The starting reference partner checkpoints for multilingual joint experimentation are the authoritative current-best models per language (established and justified in [`reports/cycle2_final_verdict.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/cycle2_final_verdict.md)), NOT the legacy Phase 0 or Cycle-1-era checkpoints:

| Language | Authoritative Checkpoint | SHA-256 Digest (First 16 chars) | Gold-Dev F1 | Precision | Recall | Status / Origin |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | [`checkpoints/current_best_konkani.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/current_best_konkani.pt) | `73bac91510a6f755...` | **96.69%** | 100.00% | 93.60% | Retrained with lemma normalization active; inflections recovered |
| **Assamese** | [`checkpoints/current_best_assamese.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/current_best_assamese.pt) | `d59afca5d233c471...` | **97.18%** | 95.17% | 99.28% | 699 label fixes + scaled class loss weighting; emission bias slashed |
| **Sinhala** | [`checkpoints/improved_sinhala_v2.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/improved_sinhala_v2.pt) | `aca068df0355e0b1...` | **96.30%** | 93.53% | 99.24% | Carried forward from Cycle 1 (conjunct normalization + silver filtering) |

### 1.2 Baseline v1 Sanity Check Verification
As a mandatory verification that the ultimate fallback baseline remains 100% frozen, intact, and reproducible, all three baseline checkpoints were evaluated on their respective locked Gold-Dev splits:

| Language | Baseline Checkpoint | Gold-Dev F1 Measured | Gold-Dev Precision | Gold-Dev Recall | Historical Reference Card | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `baseline_v1/checkpoints/final_konkani.pt` | **96.69%** | 100.00% | 93.60% | 96.69% | **EXACT MATCH (VERIFIED)** |
| **Assamese** | `baseline_v1/checkpoints/final_assamese.pt` | **98.92%** | 99.28% | 98.56% | 98.92% | **EXACT MATCH (VERIFIED)** |
| **Sinhala** | `baseline_v1/checkpoints/final_sinhala.pt` | **99.24%** | 98.50% | 100.00% | 99.24% | **EXACT MATCH (VERIFIED)** |

### 1.3 Kokborok Gold Dataset Status
`data/gold/kokborok/` is fully verified, locked, and ready:
- `train.conll`: 850 sentences (6,711 tokens, 30 entities)
- `dev.conll`: 200 sentences (1,612 tokens, 17 entities)
- `test.conll`: 200 sentences (1,542 tokens, 16 entities)
  - Cryptographically locked with SHA-256 `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799` in [`reports/kokborok_gold_test_lockfile.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_gold_test_lockfile.md).
  - ZERO-TOUCH guarantee strictly enforced: `test.conll` will NOT be accessed during Phase 12.

---

## 2. Language-Conditioning Extension Plan (3 $\to$ 4 Languages)

### 2.1 Current Implementation State
Inspection of [`model/language_conditioning.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/model/language_conditioning.py) and [`model/ner_model_v4_multilingual.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/model/ner_model_v4_multilingual.py) shows:
1. `LanguageEmbedding` instantiates `nn.Embedding(num_embeddings=num_languages, embedding_dim=embedding_dim)`.
2. The default parameter `num_languages=3` is hardcoded in `__init__`.
3. In [`training/multilingual_dataset.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/training/multilingual_dataset.py):
   - `LANG_TO_ID` is a static 3-element mapping: `{"konkani": 0, "assamese": 1, "sinhala": 2}`.
   - `RAW_CORPUS_VOLUMES` maps the 3 languages for proportional sampling.
4. Input sequence encoder dimensions:
   - Word Embedding: $100\text{d}$
   - Char-CNN: $90\text{d}$ (30 filters $\times$ 3 widths: [3, 4, 5])
   - Language Embedding: $16\text{d}$
   - Concatenated Input Dimension: $100 + 90 + 16 = 206\text{d}$.
   - BiLSTM Input Dimension: $206\text{d}$.

### 2.2 Extension Design & Non-Breaking Invariants
To extend the architecture to 4 languages without breaking existing single-language or 3-language workflows:
1. **`model/language_conditioning.py`**:
   - Update default `num_languages: int = 4`.
   - Preserve embedding dimension $16\text{d}$.
   - Forward pass remains generic: dynamically handles `lang_ids` of shape `(batch_size,)` or `(batch_size, seq_len)`.
2. **`model/ner_model_v4_multilingual.py`**:
   - Update default `num_languages: int = 4`.
   - In `per_language` mode, `fc_heads` and `crf_heads` scale to 4 heads.
3. **`training/multilingual_dataset.py`**:
   - Extend `LANG_TO_ID`:
     ```python
     LANG_TO_ID = {
         "konkani": 0,
         "assamese": 1,
         "sinhala": 2,
         "kokborok": 3,
     }
     ```
   - Extend `RAW_CORPUS_VOLUMES`:
     ```python
     RAW_CORPUS_VOLUMES = {
         "konkani": 30980,
         "assamese": 55002,
         "sinhala": 66954,
         "kokborok": 4020,
     }
     ```
   - Add Kokborok tokenization routing in `_load_pooled_data` and `_append_sample` using `ScriptTokenizer` or `KokborokTokenizer`.
   - Create unified 4-language vocabulary in `data/processed/multilingual_4lang/` ($95,291$ words, $13,022$ characters), leaving `data/processed/multilingual/` completely intact as the frozen 3-language reference.

### 2.3 Weight Warm-Start Mechanism for 4 Languages
When warm-starting a 4-language model from a 3-language checkpoint:
- **`word_embedding`**: Exact token string lookup. All $89,196$ tokens in the 3-language vocabulary inherit their trained $100\text{d}$ vectors; the $\sim 6,095$ new Kokborok tokens are initialized uniformly in $[-\sqrt{3/100}, +\sqrt{3/100}]$.
- **`char_cnn`**: The $13,022$ character table directly matches the 3-language size ($13,022$). All conv kernels and biases are copied verbatim.
- **`lang_embedding`**: Rows 0 (`konkani`), 1 (`assamese`), and 2 (`sinhala`) are copied verbatim from the 3-language checkpoint; Row 3 (`kokborok`) is initialized with $\mathcal{N}(0, 0.02)$.
- **`encoder` (BiLSTM)**: Input dimension remains strictly $206\text{d}$. All LSTM weights (`ih_l0`, `hh_l0`, `ih_l1`, `hh_l1`) are copied verbatim.
- **`attention` & `crf_shared`**: Copied verbatim from the checkpoint.
- **Divergence Gate**: Monitored on every training run via [`training/checkpoint_divergence_gate.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/training/checkpoint_divergence_gate.py) with `allow_shape_mismatch=True`.
