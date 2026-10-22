# Phase 7 Pre-Check Report: Baseline Reproduction Verification

**Date**: 2026-09-23  
**Auditor**: Senior ML/NLP Research Engineer  
**Objective**: Mandatory Pre-Check B — Fresh verification of frozen baseline checkpoints (`baseline_v1/checkpoints/final_{lang}.pt`) on locked Gold-Dev (`data/gold/{lang}/dev.conll`) against authoritative reference numbers in [`reports/baseline_v1_reference_card.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/baseline_v1_reference_card.md), and reconciliation of stress-set evaluation methodology.

---

## 1. Executive Summary: Pre-Check B Status

- **Pre-Check B Status**: **PASS (EXACT MATCH TO LOCKED REFERENCE CARD)**
- **Hardware Acceleration Used**: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Evaluation Dataset**: `data/gold/{lang}/dev.conll` (200 sentences per language; strictly isolated from train and test)
- **Cryptographic Test Lock Integrity**: All `data/gold/{lang}/test.conll` files confirmed untouched and unaccessed.

---

## 2. Gold-Dev Headline Baseline Reproduction Table

Fresh evaluation executed on `baseline_v1/checkpoints/final_{lang}.pt` vs. the locked benchmark ledger:

| Language | Checkpoint Evaluated | Locked Reference Dev F1 | Freshly Reproduced Dev F1 | Precision (P) | Recall (R) | Token Acc | Discrepancy ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `baseline_v1/checkpoints/final_konkani.pt` | **96.69%** | **96.69%** | 100.00% | 93.60% | 99.59% | **0.00 pp** | **EXACT MATCH** |
| **Assamese** | `baseline_v1/checkpoints/final_assamese.pt` | **98.92%** | **98.92%** | 99.28% | 98.56% | 99.89% | **0.00 pp** | **EXACT MATCH** |
| **Sinhala** | `baseline_v1/checkpoints/final_sinhala.pt` | **99.24%** | **99.24%** | 98.50% | 100.00% | 99.93% | **0.00 pp** | **EXACT MATCH** |
| **Macro Average** | — | **98.28%** | **98.28%** | **99.26%** | **97.39%** | **99.80%** | **0.00 pp** | **EXACT MATCH** |

---

## 3. Granular Per-Entity-Type Baseline Breakdown (Gold-Dev)

| Language | Entity Type | Reference Card F1 | Freshly Reproduced F1 | Precision | Recall | Match Confirmation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **PER** | **91.89%** | **91.89%** | 100.00% | 85.00% | **EXACT MATCH** |
| | **LOC** | **93.10%** | **93.10%** | 100.00% | 87.10% | **EXACT MATCH** |
| | **ORG** | **98.63%** | **98.63%** | 100.00% | 97.30% | **EXACT MATCH** |
| | **MISC** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |
| **Assamese** | **PER** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |
| | **LOC** | **98.73%** | **98.73%** | 98.73% | 98.73% | **EXACT MATCH** |
| | **ORG** | **96.55%** | **96.55%** | 100.00% | 93.33% | **EXACT MATCH** |
| | **MISC** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |
| **Sinhala** | **PER** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |
| | **LOC** | **97.50%** | **97.50%** | 95.12% | 100.00% | **EXACT MATCH** |
| | **ORG** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |
| | **MISC** | **100.00%** | **100.00%** | 100.00% | 100.00% | **EXACT MATCH** |

All per-type precision, recall, and F1 metrics match the authoritative ledger in [`reports/baseline_v1_reference_card.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/baseline_v1_reference_card.md) and [`reports/current_best_checkpoint_recommendation.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/current_best_checkpoint_recommendation.md) to the hundredth of a percent ($0.00\text{ pp}$ delta).

---

## 4. Root Cause Analysis: Prior Phase 7 Baseline Mismatch Discrepancy

A prior execution of Phase 7 recorded erroneous baseline dev numbers:
- Konkani: $92.70\%$ (vs. true $96.69\%$)
- Assamese: $98.54\%$ (vs. true $98.92\%$)
- Sinhala: $93.98\%$ (vs. true $99.24\%$)

### Technical Root Cause Identified
In `evaluation/phase5_targeted_interventions.py`, the evaluation helper `evaluate_model_metrics` processed tokens with a naive character loop:
```python
# BROKEN NAIVE LOOP in phase5_targeted_interventions.py:
for c_idx, ch in enumerate(tok):
    c_batch[b_idx, w_idx, c_idx] = char_to_id.get(ch, 1)
```
- The unified character vocabulary `data/processed/multilingual/char_vocab.json` was generated using [`preprocessing/tokenizer.py:ScriptTokenizer`](file:///C:/Users/harih/Desktop/NLP-MARK-3/preprocessing/tokenizer.py#L33-L75), which tokenizes into Unicode extended grapheme clusters (`\X`), treating Indic and Sinhala conjuncts and vowel marks (matras, ZWJ, virama) as atomic multi-character units.
- Standard Python string iteration (`for ch in tok`) iterates over individual Unicode code points rather than grapheme clusters. As a result, the code points in conjuncts failed to match the multi-character grapheme keys in `char_to_id` and were overwhelmingly resolved to `<UNK>` (index 1).
- This starved the Char-CNN of valid character features during inference, artificially depressing the baseline score by $-3.99\text{ pp}$ in Konkani, $-0.38\text{ pp}$ in Assamese, and $-5.26\text{ pp}$ in Sinhala.

### Fix Applied and Verified
When evaluated using the canonical dataset pipeline [`training/multilingual_dataset.py:MultilingualNERDataset`](file:///C:/Users/harih/Desktop/NLP-MARK-3/training/multilingual_dataset.py#L131-L220) with `ScriptTokenizer(lang).tokenize_characters(tok)`:
- Character tensors correctly index genuine grapheme clusters.
- The baseline scores replicate the locked reference benchmarks ($96.69\%$, $98.92\%$, $99.24\%$) exactly.
- All numbers in Phase 7 are now grounded in this verified, authoritative baseline.

---

## 5. Stress-Set Evaluation Methodology Consistency Check

Evaluated on `data/gold/{lang}/long_multientity_eval_v2.conll` (35 sentences per language):

| Language | Model Specification | Earlier Phase 6 Stress F1 | Freshly Reproduced Stress F1 | Middle Entity Recall | Collapse Rate | Methodology Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | Baseline `final_konkani.pt` | **17.39%** | **17.39%** | 0.00% | 75.76% | **CONSISTENT (0.00 pp delta)** |
| | Current Best `improved_konkani_v2.pt` | **17.39%** | **17.39%** | 0.00% | 75.76% | **CONSISTENT (0.00 pp delta)** |
| **Assamese** | Baseline `final_assamese.pt` | **22.99%** | **22.99%** | 7.41% | 70.97% | **CONSISTENT (0.00 pp delta)** |
| | Current Best `improved_assamese_v2.pt` | **22.99%** | **22.99%** | 7.41% | 70.97% | **CONSISTENT (0.00 pp delta)** |
| **Sinhala** | Baseline `final_sinhala.pt` | **36.36%** | **36.36%** | 0.00% | 60.00% | **CONSISTENT (0.00 pp delta)** |
| | Current Best `improved_sinhala_v2.pt` | **38.46%** | **38.46%** | 16.67% | 60.00% | **CONSISTENT (0.00 pp delta)** |

The stress-set evaluation methodology reproduces the recorded Phase 6 numbers identically. The hard gate has passed.
