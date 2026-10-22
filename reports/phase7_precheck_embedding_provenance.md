# Phase 7 Pre-Check Report: Word Embedding Provenance Verification

**Date**: 2026-09-23  
**Auditor**: Senior ML/NLP Research Engineer  
**Objective**: Mandatory Pre-Check A — Direct inspection and verification that all word-embedding layers across evaluated checkpoints were randomly initialized and trained end-to-end from scratch, with zero pretrained vectors (FastText, Word2Vec, GloVe, or external embeddings) loaded at any point.

---

## 1. Executive Summary: Pre-Check A Status

- **Pre-Check A Status**: **PASS (VERIFIED 100% FROM SCRATCH)**
- **External Pretrained Vector Files Found**: **0** (Zero `.vec`, `.bin`, `.glove`, or `.fasttext` files in repository)
- **External Checkpoint Calls (`from_pretrained`) Found**: **0** (Zero invocations across all scripts)
- **Embedding Initialization Mechanism**: Uniform distribution within $[-\sqrt{3/d}, \sqrt{3/d}] = [-0.1732, 0.1732]$ with $\text{padding\_idx}=0$ zeroed via [`model/word_embedding.py`](file:///C:/Users/harih/Desktop/NLP-MARK-3/model/word_embedding.py#L58-L65).
- **Weight Tensor Verification**: All evaluated checkpoints possess identically-sized static tables `[89196, 100]` matching the unified vocabulary built in Phase 8 ([`data/processed/multilingual/word_vocab.json`](file:///C:/Users/harih/Desktop/NLP-MARK-3/data/processed/multilingual/word_vocab.json)).

---

## 2. Checkpoint Weight Tensor Inspection Ledger

Every checkpoint evaluated in Phase 7 (`baseline_v1` anchors, Phase 6 current best, and Phase 6 combined checkpoints) was directly inspected via PyTorch tensor inspection:

| Checkpoint Identifier | File Path | Parameter Key | Tensor Shape | Mean | Std Dev | Min | Max | Pad Idx 0 Norm |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Konkani** | `baseline_v1/checkpoints/final_konkani.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001669` | `0.103126` | `-0.431004` | `+0.427530` | `0.000000` |
| **Baseline Assamese** | `baseline_v1/checkpoints/final_assamese.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001710` | `0.103228` | `-0.420764` | `+0.441232` | `0.000000` |
| **Baseline Sinhala** | `baseline_v1/checkpoints/final_sinhala.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001680` | `0.103155` | `-0.431004` | `+0.426555` | `0.000000` |
| **Current Best Konkani** | `checkpoints/improved_konkani_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001669` | `0.103126` | `-0.431004` | `+0.427530` | `0.000000` |
| **Current Best Assamese** | `checkpoints/improved_assamese_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001710` | `0.103228` | `-0.420764` | `+0.441232` | `0.000000` |
| **Current Best Sinhala** | `checkpoints/improved_sinhala_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001684` | `0.103184` | `-0.431004` | `+0.426555` | `0.000000` |
| **Combined Konkani** | `checkpoints/combined_konkani_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001671` | `0.103175` | `-0.431004` | `+0.443269` | `0.000000` |
| **Combined Assamese** | `checkpoints/combined_assamese_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001700` | `0.103254` | `-0.426557` | `+0.439744` | `0.000000` |
| **Combined Sinhala** | `checkpoints/combined_sinhala_v2.pt` | `word_embedding.embedding.weight` | `[89196, 100]` | `+0.001684` | `0.103184` | `-0.431004` | `+0.426555` | `0.000000` |

### Key Tensor Observations
1. **Dimension Uniformity**: All checkpoints utilize an embedding dimension of exactly 100 ($d=100$), with exactly 89,196 vocabulary entries. This corresponds 1-to-1 with the token count in `data/processed/multilingual/word_vocab.json`.
2. **Zero Padding Vector**: In every checkpoint, index 0 (the `<PAD>` token) is strictly all-zeros (`pad0 = 0.0`), matching the initialization contract of `nn.Embedding(padding_idx=0)`.
3. **Weight Distribution**: The weights exhibit a zero-centered Gaussian-like distribution ($\mu \approx 0.0017$, $\sigma \approx 0.103$) that evolved naturally through gradient descent from the initial uniform distribution $[-0.1732, +0.1732]$. There are no clustered artifacts, normalized unit spheres, or typical signature distributions of pretrained Word2Vec or FastText vectors (which typically have 300 dimensions and norm 1.0).

---

## 3. Code Trace & Training Pipeline Provenance

The entire training lineage from Phase 0 baseline through Phase 6 compound retraining was traced through the actual source files:

1. **Model Layer Definition**:
   [`model/word_embedding.py`](file:///C:/Users/harih/Desktop/NLP-MARK-3/model/word_embedding.py#L30-L75):
   ```python
   class WordEmbedding(nn.Module):
       def __init__(self, vocab_size: int, embedding_dim: int = 100, padding_idx: int = 0, dropout: float = 0.3):
           ...
           self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
           self._init_weights()

       def _init_weights(self):
           bound = math.sqrt(3.0 / self.embedding_dim)
           nn.init.uniform_(self.embedding.weight, -bound, bound)
           if self.padding_idx is not None:
               with torch.no_grad():
                   self.embedding.weight[self.padding_idx].fill_(0.0)
   ```
2. **Multilingual Architecture**:
   [`model/ner_model_v4_multilingual.py`](file:///C:/Users/harih/Desktop/NLP-MARK-3/model/ner_model_v4_multilingual.py#L106-L111):
   Instantiates `WordEmbedding` strictly from the unified vocab size (89,196) and dimension 100.
3. **Training Scripts**:
   - `training/train_multilingual_joint.py`: Trains `joint_best.pt` starting from scratch (`_init_weights()`) with Adam optimizer and CRF loss.
   - `training/train_finetune.py`: Initializes from joint weights and fine-tunes on monolingual gold+silver data.
   - `training/train_selftraining.py`: Iteratively fine-tunes on confident pseudo-labels.
   - `training/train_augmentation.py`: Trains `augmented_{lang}.pt` (Phase 11) using rule-based lexical substitution.
   - `evaluation/phase6_combined_retraining.py`: Initializes `b_model` directly from `baseline_v1/checkpoints/final_{lang}.pt` and fine-tunes for 5 epochs with class weights / augmentation.

At no point in any script is any external vector file, `gensim`, `fasttext`, or pretrained checkpoint loaded.

---

## 4. Resolution of "FastText" Mentions in Project History

A comprehensive search of the entire codebase for the term "FastText" reveals:
1. `weak_supervision/pretrained_component_audit.md`: Table checking pretrained libraries lists `gensim` (Word2Vec/FastText pretrained) with status **"NO - PASS (None Present)"**.
2. `NO_PRETRAINED_MODEL_CHECKLIST.md` (Line 172): Explicitly affirms:
   > *"Zero pretrained embeddings: No Word2Vec, GloVe, FastText, or external vector tables were loaded or referenced under any circumstances. All word embeddings were instantiated randomly via `nn.Embedding` and learned purely from the NER training objective."*
3. `README.md` (Line 26): Re-asserts zero pretrained embeddings.
4. `model/word_embedding.py` (Line 5): Constraint docstring confirms no FastText/Word2Vec.

**Definitive Resolution**:
Any mention of FastText in the repository's history or literature comparisons is strictly a negative compliance assertion confirming that FastText was **never** used. No pretrained FastText vector file was ever introduced into this repository.

---

## 5. Pre-Check A Conclusion

Pre-Check A is formally **PASSED**. All inputs under evaluation in Phase 7 are verified to have random initialization provenance and zero pretrained vector leakage.
