# Final Checkpoint Provenance & Cryptographic Lineage

**Paper Section Target**: Methodology / Reproducibility & Model Provenance  
**Primary Source Reports**: 
- [`reports/current_best_checkpoint_recommendation.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/current_best_checkpoint_recommendation.md)
- [`reports/checkpoint_provenance_trace.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/checkpoint_provenance_trace.md)
- [`reports/final_checkpoint_inventory.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/final_checkpoint_inventory.md)  
**Verification Date**: `2026-09-17`  
**Git Commit of Record**: `0ea0e46` (*Audit all 50 checkpoints, evaluate master cross-checkpoint comparison on locked gold-dev, select genuine best per language, and cryptographically lock deployment bundles*)  

---

## 1. Authoritative Best Checkpoints Per Language

Following the exhaustive audit of all 50 checkpoints across the 16 phases of development, the definitive best checkpoints are:

| Language | Authoritative Checkpoint | Official Deployment Alias | Origin Phase | SHA-256 Cryptographic Checksum | Gold-Dev F1 | Gold-Test F1 | Total Parameters | Base Size (Bytes) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Konkani** | `selftrained_konkani_best.pt` | `final_konkani.pt` | **Phase 10** (Self-Training) | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | **96.69%** | **96.27%** | 10,596,571 | 42,400,357 |
| **Assamese** | `augmented_assamese.pt` | `final_assamese.pt` | **Phase 11** (Data Augmentation) | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | **98.92%** | **97.87%** | 10,596,571 | 42,400,757 |
| **Sinhala** | `augmented_sinhala.pt` | `final_sinhala.pt` | **Phase 11** (Data Augmentation) | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | **99.24%** | **97.76%** | 10,596,571 | 42,401,269 |

---

## 2. Parameter Count & File Size Analysis

All three language checkpoints share an identical parameter architecture:
- **Total Float32 Parameters**: **10,596,571** across 42 distinct parameter tensors.
- **Unified Multilingual Vocabulary**: $89,196$ word types, $13,022$ character types.
- **Parameter Calculation**:
  - Word Embeddings: $89,196 \times 100 = 8,919,600$
  - Char Embeddings: $13,022 \times 50 = 651,100$
  - Char CNN Conv1D Kernels: $30 \times (50 \times 3) + 30 \times (50 \times 4) + 30 \times (50 \times 5) + 90 = 18,090$
  - Language Conditioning: $3 \times 16 = 48$
  - BiLSTM Layer 1 (Input 206, Hidden 128 $\times$ 2 dir): $4 \times (206 \times 128 + 128 \times 128 + 128 + 128) \times 2 = 344,064$
  - BiLSTM Layer 2 (Input 256, Hidden 128 $\times$ 2 dir): $4 \times (256 \times 128 + 128 \times 128 + 128 + 128) \times 2 = 395,264$
  - Multi-Head Self-Attention (4 heads, hidden 256): $W_Q, W_K, W_V, W_O \in \mathbb{R}^{256 \times 256} + \text{biases} \approx 263,168$
  - LayerNorm & Emission Linear ($256 \to 9$): $2,835$
  - CRF Transition Matrix ($9 \times 9$): $81$
  - **Base Tensor Footprint**: $10,596,571 \times 4 \text{ bytes} \approx 42,386,284 \text{ bytes}$ (~$40.4$ MB).
- **Cross-Language Divergence**: Pairwise tensor diffs show $44.5\% - 47.1\%$ parameter divergence ($L_2$ norm diffs 10.60 to 14.34), confirming genuine, language-specialized weights rather than identical copies.

---

## 3. Full Training Lineage from Scratch

Every model was constructed from scratch following this documented lineage:

```
[Phase 5: Monolingual Baseline BiLSTM-CRF]
  (Word embeddings randomly initialized, trained on 600 gold train sents)
        │
        ▼
[Phase 6: Character-Level CNN Encoder Integration]
        │
        ▼
[Phase 7: Multi-Head Self-Attention Block Insertion]
        │
        ▼
[Phase 8: Multilingual Joint Warm-Start Training]
  (Trained across all 3 languages simultaneously on pooled gold+silver;
   produced checkpoints/joint_best.pt)
        │
        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
 [Konkani Branch]          [Assamese Branch]          [Sinhala Branch]
Phase 9: Fine-tuning       Phase 9: Fine-tuning       Phase 9: Fine-tuning
        │                          │                          │
Phase 10: Self-Training    Phase 10: Self-Training    Phase 10: Self-Training
(Round 1 accepted:         (Rounds 1 & 2 maintained:  (Plateau reached)
 96.69% Gold-Dev F1)       98.91% Gold-Dev F1)                │
        │                          │                          ▼
        │                  Phase 11: Augmentation    Phase 11: Augmentation
        │                  (Entity Substitution:     (Class-Weighted CRF:
        │                   98.92% Gold-Dev F1)       99.24% Gold-Dev F1)
        │                          │                          │
        ▼                          ▼                          ▼
selftrained_konkani_best.pt augmented_assamese.pt     augmented_sinhala.pt
 (LOCKED BEST: 96.69%)      (LOCKED BEST: 98.92%)     (LOCKED BEST: 99.24%)
        │                          │                          │
        ▼                          ▼                          ▼
final_konkani.pt           final_assamese.pt          final_sinhala.pt
```

---

## 4. Cryptographic Verification of Deployment Bundles

The production deployment bundles in `deployment/bundles/{lang}/v1/model.pt` were verified against the locked checkpoints:

| Language | Checkpoint Path | Checkpoint SHA-256 | Deployment Bundle SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Konkani** | `checkpoints/final_konkani.pt` | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | **EXACT MATCH** |
| **Assamese** | `checkpoints/final_assamese.pt` | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | **EXACT MATCH** |
| **Sinhala** | `checkpoints/final_sinhala.pt` | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | **EXACT MATCH** |
