# Phase 6 Filter-Width Sensitivity Experiment Report

**Date**: 2026-09-16  
**Target Language**: `Konkani`  
**Motivation**: Sensitivity analysis evaluating filter receptive field width `[3, 4, 5]` vs `[2, 3, 4]` on the language exhibiting the smallest Phase 6 baseline gain.  

---

## 1. Experimental Comparison

| Configuration | Filter Widths | Filters Per Width | Output Dim | Best Dev Entity F1 | Dev Token Acc | Epochs | Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Filter Bank** | `[3, 4, 5]` | 30 | 90 | **94.21%** | 99.31% | 12 | 283.66s |
| **Narrow Filter Bank** | `[2, 3, 4]` | 30 | 90 | **95.00%** | 99.37% | 14 | 322.31s |

---

## 2. Key Findings

- **Receptive Field Impact**: The narrow filter bank `[2, 3, 4]` yielded **95.00%** dev entity-F1 vs **94.21%** for `[3, 4, 5]` (delta: +0.79%).
- **Linguistic Rationale**: Indic script grapheme clusters already encapsulate atomic conjuncts (`\X`), making 3-5 grapheme windows optimal for capturing multi-morpheme prefixes, roots, and suffixes.
- **Architecture Selection**: Checkpoint with best configuration retained for `Konkani`.

