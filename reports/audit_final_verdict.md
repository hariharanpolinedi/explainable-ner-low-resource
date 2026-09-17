# Verification Audit of Phase 10/11 Patch: Final Consolidated Report & Verdict

**Audit Scope**: Verification audit of the "Phase 10/11 Patch: ORG Recall & Sentence-Length Robustness" (commit `e67db9c2753794dbc039e8b3e7ef79f87688584e`).  
**Auditor**: Senior ML/NLP Engineer (Independent Verification & Correction Audit).  
**Date**: September 17, 2026.  
**Constraint Adherence**: Strict from-scratch constraint (zero pretrained models, zero pretrained embeddings/tokenizers, zero external artifacts).

---

## 1. Executive Summary & Audit Motivation

The Phase 10/11 patch claimed a "PASS" across Konkani, Assamese, and Sinhala for eliminating ORG under-detection and middle-entity dropping in long sentences. However, anomalous artifacts were observed:
1. Konkani 30+ token long-sentence overall F1 showed a delta of exactly **0.00%**.
2. Assamese and Sinhala exhibited **0.00% deltas across every single metric in every single length bucket**.
3. "Missed middle entities" was reported as **0/0 in every cell**, indicating the failure mode was never tested by the evaluation set.
4. Silver-label precision was reported as **exactly 100.0%** for Konkani and Assamese.
5. All three checkpoints reported identical file sizes of **40.4 MB**.

This audit conducted an exhaustive, multi-dimensional verification into checkpoint integrity, raw counts, error cases, silver label precision methodology, and built a brand-new supplementary gold test set of long multi-entity sentences.

---

## 2. Synthesis of Audit Findings

### 2.1 Checkpoint Distinctness & Retraining Integrity (Finding 1)
- **File Size Plausibility**: All three checkpoints have 10,596,571 float32 parameters across 42 tensors. The unified multilingual vocabulary (89,196 words, 13,022 characters) dictates that any checkpoint using this architecture has a base parameter size of $10,596,571 \times 4 = 42,386,284$ bytes (~40.4 MB). The exact uncompressed sizes are 42,400,357 bytes to 42,401,269 bytes. Identical file sizes are an architectural property of the fixed parameter count, not an indicator of file cloning.
- **Cross-Language Distinctness**: Pairwise parameter diffs between models show $44.5\% - 47.1\%$ parameter divergence ($L_2$ norms 10.60 to 14.34). The models are genuine, language-distinct checkpoints, **not cross-language duplicates**.
- **Retraining Status**:
  - **Konkani**: $L_2$ norm diff = 4.9389; 2,404,319 of 10,596,571 parameters changed ($22.69\%$). **Konkani was genuinely retrained and updated**.
  - **Assamese**: $L_2$ norm diff = **0.0000**; 0 of 10,596,571 parameters changed ($0.00\%$).
  - **Sinhala**: $L_2$ norm diff = **0.0000**; 0 of 10,596,571 parameters changed ($0.00\%$).
- **Root Cause of 0.00% Deltas**: In `training/run_patch_pipeline.py`, the training loop initialized `best_dev_f1` to the pre-patch baseline dev F1 and only saved weights if `epoch_f1 > best_dev_f1`. Assamese tied its pre-patch dev F1 (98.92% on all epochs), and Sinhala regressed (96.30% vs 99.24% baseline). Both early-stopped and saved their untouched pre-patch checkpoints to `patched_assamese.pt` and `patched_sinhala.pt`. **Retraining failed to update Assamese and Sinhala weights.**

### 2.2 Raw-Count Reconstruction & Statistical Fragility (Finding 2)
- Reconstructing raw denominators for `reports/patch_org_length_results.md` revealed that the evaluation set was statistically inadequate for the claims being made:
  - **Konkani Long (30+ tokens)**: The entire bucket contains only **16 sentences** and **4 ORG entities**. Both pre-patch and post-patch had 4 TP and 0 FN ($100.0\%$ recall). A single false negative would drop recall from $100\%$ to $75\%$.
  - **Assamese Gold-Dev**: Contains only **15 ORG entities across all 200 sentences**. The Medium bucket has 6 ORGs, and the Long bucket has **1 single ORG entity**. Any single error causes a $16.7\%$ to $100.0\%$ percentage swing.
  - **Sinhala Gold-Dev**: Contains only **30 ORG entities across 200 sentences**. The Long bucket contains **2 ORG entities**.
- The reported high percentages masked extremely small sample counts that could not support statistically significant claims.

### 2.3 Konkani Long-Sentence Null Result Investigation (Finding 3)
- Token-by-token comparison across all 16 sentences in Konkani Gold-Dev (30+ tokens) revealed:
  - **15 out of 16 sentences (93.75%) yielded 100% token-for-token identical predictions** between pre-patch and post-patch models.
  - Across the 16 sentences, there were 4 ORG entities, and the pre-patch model had already detected all 4 ($100\%$ recall). There were zero ORG errors in this bucket for the patch to fix.
  - The only errors were 3 false negatives on PER/LOC (Sentence 10: PER "नारायण", Sentence 42: LOC "जागो", Sentence 118: LOC "कोंकणी भवन"). Both pre and post models made the exact same errors.
- **Verdict on Explanations**:
  - Explanation (a) [Identical errors made before and after]: **TRUE**.
  - Explanation (b) [Different errors coincidental netting to zero]: **FALSE**.
  - Explanation (c) [Bucket too small]: **TRUE** (16 sentences, 4 ORGs, zero ORG errors to correct).

### 2.4 Silver-Precision Methodology Re-Audit (Finding 4)
- **Methodological Flaw**: The original report claimed $100.0\%$ ORG precision for Konkani and Assamese. Inspection revealed that `weak_supervision/label_aggregator.py` calibrated its heuristic source weights on the 200 Gold-Dev sentences, and then evaluated silver label precision against the exact same 200 sentences (in-sample evaluation).
- **Corrected Out-of-Sample Precision** (evaluated on independent Gold-Train, 600 sentences, uncalibrated):
  - **Konkani ORG**: **86.90%** (73 TP, 11 FP) vs original claim of $100.0\%$.
  - **Assamese ORG**: **81.25%** (26 TP, 6 FP) vs original claim of $100.0\%$.
  - **Sinhala ORG**: **91.43%** (32 TP, 3 FP) vs original claim of $96.30\%$.
- While all three languages remain above the project's $\ge 70.0\%$ weak supervision quality threshold, the original $100.0\%$ claim was circular and invalid.

### 2.5 Supplementary Gold-Set Construction & Re-Measurement (Findings 5 & 6)
- Because Gold-Dev contained 0/0 missed middle entities and virtually zero multi-entity lists, a supplementary gold dataset was constructed from the raw corpus pool following Phase 3 annotation guidelines:
  - **Konkani**: 12 sentences (all 12 naturally-occurring $30+$ token multi-entity list sentences in the corpus; 43 entities, 4 middle targets). Shortfall from 35 target explicitly recorded.
  - **Assamese**: 35 sentences ($50.8$ avg length, 201 entities, 78 middle targets).
  - **Sinhala**: 28 sentences ($43.9$ avg length, 116 entities, 30 middle targets). Shortfall from 35 target explicitly recorded.
  - **Inter-Annotator Agreement**: Konkani $\kappa = 1.0000$ (Span agreement $1.0000$); Assamese $\kappa = 0.9579$ (Span agreement $0.9333$); Sinhala $\kappa = 1.0000$ (Span agreement $1.0000$).
- **Re-Measurement Results**:
  - **Konkani**: Pre F1 $92.50\% \to$ Post F1 $91.36\%$ ($\Delta -1.14\%$). Middle Recall: Pre $100.0\%$ ($4/4$) vs Post $100.0\%$ ($4/4$) ($\Delta 0.00\%$). The pre-patch baseline already had $100\%$ middle-entity recall.
  - **Assamese**: Pre F1 $94.61\% \to$ Post F1 $94.61\%$ ($\Delta 0.00\%$). Middle Recall: Pre $96.15\%$ ($75/78$) vs Post $96.15\%$ ($75/78$) ($\Delta 0.00\%$). Weights were never updated.
  - **Sinhala**: Pre F1 $98.25\% \to$ Post F1 $98.25\%$ ($\Delta 0.00\%$). Middle Recall: Pre $96.67\%$ ($29/30$) vs Post $96.67\%$ ($29/30$) ($\Delta 0.00\%$). Weights were never updated.

---

## 3. Honest Verdict Per Language

### 1. Konkani: **INCONCLUSIVE / SLIGHT REGRESSION**
- **Reasoning**: Konkani was genuinely retrained ($22.69\%$ weights updated), but the patch's intended benefits did not manifest on long sentences or lists.
  - In Gold-Dev, the long-sentence bucket had 0 ORG errors initially, producing a $0.00\%$ delta.
  - On the supplementary long multi-entity set, pre-patch baseline already resolved $100\%$ ($4/4$) of middle entities. Post-patch introduced 1 false positive ORG, dropping overall F1 from $92.50\%$ to $91.36\%$ ($\Delta -1.14\%$).
  - The patch did yield a modest ORG recall gain on medium sentences ($66.7\% \to 80.0\%$), but did not fix or improve long-sentence list processing.

### 2. Assamese: **DID NOT WORK (TRAINING ABORTED / UNCHANGED WEIGHTS)**
- **Reasoning**: The reported patch results were completely fictitious in execution.
  - The post-patch checkpoint `patched_assamese.pt` is byte-for-byte parameter identical ($L_2$ diff = $0.0000$) to the pre-patch checkpoint `augmented_assamese.pt`.
  - The patch training pipeline early-stopped when dev F1 tied $98.92\%$, re-saving the untouched baseline weights.
  - The reported $0.00\%$ deltas across all buckets occurred solely because the model was identical.

### 3. Sinhala: **DID NOT WORK (TRAINING REGRESSED / UNCHANGED WEIGHTS)**
- **Reasoning**: Like Assamese, the reported patch results represent an unapplied patch.
  - The post-patch checkpoint `patched_sinhala.pt` is byte-for-byte parameter identical ($L_2$ diff = $0.0000$) to `augmented_sinhala.pt`.
  - Retraining caused dev F1 to drop from $99.24\%$ to $96.30\%$, causing the early-stopping logic to reject the new weights and re-save the baseline.
  - The reported $0.00\%$ deltas were the consequence of evaluating the baseline against itself.

---

## 4. Testing & Validation Checklist

| Test # | Check Description | Audit Result | Key Metric / Evidence |
|---|---|---|---|
| **1** | Checkpoint-diff test | **PASS** | Konkani diff $L_2 = 4.9389$ ($22.7\%$ params); Assamese $L_2 = 0.0000$; Sinhala $L_2 = 0.0000$. |
| **2** | Cross-language checkpoint distinctness test | **PASS** | Distinct models ($44.5\% - 47.1\%$ parameter difference, $L_2 \ge 10.60$). Fixed $40.4$ MB size explained by unified architecture. |
| **3** | Raw-count reliability test | **PASS** | Denominators attached to all percentage cells. Flagged extreme sample fragility (e.g. 1 ORG in Assamese Long bucket). |
| **4** | Long-sentence null-result explanation test | **PASS** | Proven via token-by-token comparison: Explanations (a) and (c) confirmed; (b) rejected. 15/16 sentences identical. |
| **5** | Silver-precision re-audit test | **PASS** | Corrected on independent Gold-Train: Konkani $86.9\%$, Assamese $81.3\%$, Sinhala $91.4\%$ (original $100\%$ invalidated). |
| **6** | Supplementary-set annotation quality test | **PASS** | Dual-annotator agreement: $\kappa \ge 0.9579$, Span agreement $\ge 0.9333$. Sourced exclusively from raw corpus pool. |
| **7** | Supplementary-set re-measurement test | **PASS** | Evaluated pre vs post checkpoints on supplementary sets; middle-entity recall measured and reported. |

---

## 5. Absolute No-Pretrained Compliance Verification

- **Architectures**: Pure PyTorch custom Character CNN + Word Embedding + BiLSTM + CRF.
- **Embeddings/Tokenizers**: Pure whitespace tokenization + from-scratch vocabulary dictionary (`word2idx.json`, `char2idx.json`).
- **External Dependencies**: Strictly ZERO pretrained transformers (HuggingFace/BERT/mBERT), zero pretrained vectors (FastText/GloVe), zero external annotation models.
- **Status**: **PASS (100% Compliant)**.

---

## 6. Official Recommendation for Phase 12 Gate

**RECOMMENDATION**: **RETURN TO PATCH FOR FURTHER CORRECTION (NO-GO FOR PHASE 12)**

### Detailed Justification:
1. **Unpatched Checkpoints**: Two of the three target languages (Assamese and Sinhala) were never patched at all. Retraining failed to improve upon the pre-patch baseline dev F1, so the training script re-saved the pre-patch weights. Proceeding to Phase 12 (hyperparameter search and ablations) under the premise that all three languages are "patched" would mean running search over unpatched models for two languages.
2. **Untargeted In Konkani**: For Konkani, the patch was retrained, but middle-entity recall was already 100% in the baseline and overall F1 slightly regressed on long sentences (-1.14%). The patch did not solve the long-sentence robustness issue it targeted.
3. **Training Objective Correction Required**: The patch training script (`run_patch_pipeline.py`) must be modified so that checkpoint saving and early stopping evaluate on a balanced validation split or a composite loss that includes length robustness and ORG recall, rather than relying exclusively on a global dev F1 that is dominated by short sentences and non-ORG classes.

Until the patch training pipeline is corrected and genuinely updates all three models with measurable improvements, Phase 12 must remain closed.
