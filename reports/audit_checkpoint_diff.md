# Verification Audit: Checkpoint Distinctness & Parameter Diffs

**Date**: 2026-09-17  
**Audit Target**: `checkpoints/patched_{konkani,assamese,sinhala}.pt` vs pre-patch baselines  
**Methodology**: Byte-level checksum/size analysis, parameter-wise L2 norm, and cross-language tensor divergence.  

---

## 1. Executive Findings: Summary Table

| Comparison | Pre-Patch Checkpoint | Post-Patch Checkpoint | L2 Norm Diff | Max Abs Diff | Differing Parameters | Weight Divergence (%) | Audit Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** (Pre vs Post) | `konkani_pre` (42,401,269 B) | `konkani_post` (42,400,357 B) | **4.9389** | 0.040852 | 2,404,319 / 10,596,571 | **22.69%** | **UPDATED (Genuinely Trained)** |
| **Assamese** (Pre vs Post) | `assamese_pre` (42,400,693 B) | `assamese_post` (42,400,597 B) | **0.0000** | 0.000000 | 0 / 10,596,571 | **0.00%** | **UNTOUCHED (Exact Pre-Patch Re-save)** |
| **Sinhala** (Pre vs Post) | `sinhala_pre` (42,400,645 B) | `sinhala_post` (42,400,357 B) | **0.0000** | 0.000000 | 0 / 10,596,571 | **0.00%** | **UNTOUCHED (Exact Pre-Patch Re-save)** |

---

## 2. Cross-Language Post-Patch Distinctness

| Comparison Pair | L2 Norm Diff | Max Abs Diff | Differing Parameters | Parameter Divergence | Audit Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani vs Assamese** | **14.2043** | 0.214795 | 4,992,923 / 10,596,571 | **47.12%** | **GENUINELY DISTINCT MODELS** |
| **Konkani vs Sinhala** | **10.6029** | 0.083853 | 4,855,965 / 10,596,571 | **45.83%** | **GENUINELY DISTINCT MODELS** |
| **Assamese vs Sinhala** | **14.3402** | 0.190138 | 4,725,475 / 10,596,571 | **44.59%** | **GENUINELY DISTINCT MODELS** |

---

## 3. Explanation of the '40.4 MB' File Size Finding

A critical question in the audit was why all three patched checkpoints have virtually identical file sizes (~42,400,357 to 42,400,597 bytes):

1. **Architectural Determinism**: All models in Phase 8–11 utilize `NERMultilingualModel` instantiated with the **unified multilingual vocabulary** (`load_or_create_multilingual_vocabs`), consisting of exactly **89,196 words** and **13,022 characters** pooled across all three languages.
2. **Exact Parameter Count**: Every multilingual checkpoint contains exactly **10,596,571 float32 parameters** across 42 weight tensors, representing $10,596,571 \times 4 = 42,386,284$ bytes of uncompressed raw parameter storage.
3. **File Size Breakdown**:
   - Raw parameters: 42,386,284 bytes
   - PyTorch ZIP archive overhead & metadata dictionary: ~14,073 bytes
   - Total size: **42,400,357 bytes** (Konkani and Sinhala) and **42,400,597 bytes** (Assamese, differing by 240 bytes due to string length variations in pickled metric dictionaries).
4. **Non-Cloning Verification**: Despite identical byte lengths, cross-language tensor diffs confirm that Konkani, Assamese, and Sinhala differ in **4.7 to 5.0 million parameters (44.5% to 47.1% divergence)**. They are strictly NOT duplicate copies.

---

## 4. Critical Finding: Why Assamese and Sinhala Had 0.00% Pre/Post Diffs

> [!IMPORTANT]
> **RETRAINING ROLLBACK MECHANISM LEFT ASSAMESE AND SINHALA 100% UNCHANGED**
> 
> Inspection of `training/run_patch_pipeline.py` reveals the exact operational reason why Assamese and Sinhala showed 0.00% deltas:
> 1. Before fine-tuning, `train_patched_model` established `best_dev_f1` as the baseline dev score (Assamese: 98.92%, Sinhala: 99.24%).
> 2. During fine-tuning on the patch data:
>    - **Konkani**: Epoch 1 reached **96.72%** (beating baseline 96.69%). `best_state` was updated, training ran 4 epochs, and early-stopped, saving the improved Epoch 1 weights.
>    - **Assamese**: Epochs 1, 2, and 3 all scored **98.92%** (matching but not strictly exceeding `best_dev_f1`). Because the check was `if dev_f1 > best_dev_f1:`, `best_state` was **never updated**, and early stopping triggered at Epoch 3. The initial baseline weights (`checkpoints/augmented_assamese.pt`) were re-saved untouched.
>    - **Sinhala**: Epochs 1, 2, and 3 scored **96.30%, 95.91%, and 95.94%** (below baseline 99.24%). Because dev F1 did not improve, `best_state` remained the initial weights, and early stopping triggered at Epoch 3. The initial baseline weights (`checkpoints/augmented_sinhala.pt`) were re-saved untouched.
> 
> **Conclusion**: Only **Konkani** underwent a genuine weight update during the patch (`L2 = 4.9389`, 2.4M params changed). For Assamese and Sinhala, the patch retraining failed to beat their already near-ceiling baselines on Gold-Dev, so the safety guard correctly preserved their pre-patch weights. **The reported 0.00% deltas were not evaluation artifacts — they were mathematically exact reflections of untouched weights!**