# Honest Per-Entity-Type Baseline Report

**Date**: 2026-09-17  
**Evaluation Set**: Locked, untouched `data/gold/{lang}/dev.conll` (200 sentences per language).  
**Rule Enforced**: No test set peeking, no contaminated `long_multientity_eval.conll`.  

---

## Konkani Master Per-Type Baseline

| Checkpoint | Origin Phase | Overall F1 | Token Acc | PER F1 (P / R) | LOC F1 (P / R) | ORG F1 (P / R) | MISC F1 (P / R) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_konkani.pt` | Phase 9 Fine-Tuned | **96.69%** | 99.59% | **91.89%** (100.0/85.0) | **93.10%** (100.0/87.1) | **98.63%** (100.0/97.3) | **100.00%** (100.0/100.0) |
| `selftrained_konkani_best.pt` | Phase 10 Self-Trained (Baseline) | **96.69%** | 99.59% | **91.89%** (100.0/85.0) | **93.10%** (100.0/87.1) | **98.63%** (100.0/97.3) | **100.00%** (100.0/100.0) |
| `patched_konkani.pt` | Phase 10/11 Patch v1 (Rejected) | **96.72%** | 99.50% | **91.89%** (100.0/85.0) | **93.10%** (100.0/87.1) | **98.67%** (97.4/100.0) | **100.00%** (100.0/100.0) |
| `patched_v2_konkani.pt` | Patch v2 (Rejected) | **92.31%** | 98.93% | **91.89%** (100.0/85.0) | **93.10%** (100.0/87.1) | **89.19%** (89.2/89.2) | **94.87%** (90.2/100.0) |
| `final_konkani.pt` | Currently Deployed (final_konkani) | **96.69%** | 99.59% | **91.89%** (100.0/85.0) | **93.10%** (100.0/87.1) | **98.63%** (100.0/97.3) | **100.00%** (100.0/100.0) |

## Assamese Master Per-Type Baseline

| Checkpoint | Origin Phase | Overall F1 | Token Acc | PER F1 (P / R) | LOC F1 (P / R) | ORG F1 (P / R) | MISC F1 (P / R) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_assamese.pt` | Phase 9 Fine-Tuned | **98.91%** | 99.89% | **100.00%** (100.0/100.0) | **98.72%** (100.0/97.5) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |
| `selftrained_assamese_best.pt` | Phase 10 Self-Trained | **98.91%** | 99.89% | **100.00%** (100.0/100.0) | **98.72%** (100.0/97.5) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |
| `augmented_assamese.pt` | Phase 11 Augmented (Baseline) | **98.92%** | 99.89% | **100.00%** (100.0/100.0) | **98.73%** (98.7/98.7) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |
| `patched_assamese.pt` | Phase 10/11 Patch v1 (Rejected) | **98.92%** | 99.89% | **100.00%** (100.0/100.0) | **98.73%** (98.7/98.7) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |
| `patched_v2_assamese.pt` | Patch v2 (Candidate) | **98.22%** | 99.78% | **94.74%** (90.0/100.0) | **98.75%** (97.5/100.0) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |
| `final_assamese.pt` | Currently Deployed (final_assamese) | **98.92%** | 99.89% | **100.00%** (100.0/100.0) | **98.73%** (98.7/98.7) | **96.55%** (100.0/93.3) | **100.00%** (100.0/100.0) |

## Sinhala Master Per-Type Baseline

| Checkpoint | Origin Phase | Overall F1 | Token Acc | PER F1 (P / R) | LOC F1 (P / R) | ORG F1 (P / R) | MISC F1 (P / R) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_sinhala.pt` | Phase 9 Fine-Tuned | **98.13%** | 99.78% | **100.00%** (100.0/100.0) | **93.98%** (88.6/100.0) | **100.00%** (100.0/100.0) | **100.00%** (100.0/100.0) |
| `selftrained_sinhala_best.pt` | Phase 10 Self-Trained | **98.13%** | 99.78% | **100.00%** (100.0/100.0) | **93.98%** (88.6/100.0) | **100.00%** (100.0/100.0) | **100.00%** (100.0/100.0) |
| `augmented_sinhala.pt` | Phase 11 Augmented (Baseline) | **99.24%** | 99.93% | **100.00%** (100.0/100.0) | **97.50%** (95.1/100.0) | **100.00%** (100.0/100.0) | **100.00%** (100.0/100.0) |
| `patched_sinhala.pt` | Phase 10/11 Patch v1 (Rejected) | **99.24%** | 99.93% | **100.00%** (100.0/100.0) | **97.50%** (95.1/100.0) | **100.00%** (100.0/100.0) | **100.00%** (100.0/100.0) |
| `patched_v2_sinhala.pt` | Patch v2 (Rejected) | **96.65%** | 99.53% | **97.30%** (97.3/97.3) | **98.73%** (97.5/100.0) | **90.32%** (82.3/100.0) | **100.00%** (100.0/100.0) |
| `final_sinhala.pt` | Currently Deployed (final_sinhala) | **99.24%** | 99.93% | **100.00%** (100.0/100.0) | **97.50%** (95.1/100.0) | **100.00%** (100.0/100.0) | **100.00%** (100.0/100.0) |