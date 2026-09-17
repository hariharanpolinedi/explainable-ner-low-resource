# LOC-Targeted Retraining & Divergence Gate Results

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Evaluation Set**: Locked, Untouched Gold-Dev (`data/gold/{lang}/dev.conll`) ONLY  

---

## 1. Executive Summary & Divergence Gate Verification

| Language | Baseline Checkpoint | Divergence Gate | % Params Changed | L2 Diff | Max Diff | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | `selftrained_konkani_best.pt` | **PASSED** | 20.88% | 3.4106 | 0.0371 | Saved to loc_patched_konkani.pt |
| **Assamese** | `augmented_assamese.pt` | **PASSED** | 21.46% | 3.4824 | 0.0282 | Saved to loc_patched_assamese.pt |
| **Sinhala** | `augmented_sinhala.pt` | **PASSED** | 21.90% | 4.1469 | 0.0328 | Saved to loc_patched_sinhala.pt |

---

## 2. Before vs After Performance on Real Gold-Dev

| Language | Metric | Baseline | Retrained (LOC Patch) | Delta | Tradeoff / Analysis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Konkani** | Overall Entity F1 | 96.69% | 96.27% | -0.42% | Overall dev benchmark |
| | **LOC F1** | 93.10% | **93.10%** | **+0.00%** | Target Entity Type |
| | LOC Precision | 100.00% | 100.00% | +0.00% | False Positive Check |
| | LOC Recall | 87.10% | 87.10% | +0.00% | Ground Truth Coverage |
| | ORG F1 | 98.63% | 97.22% | -1.41% | Regression Check |
| | PER F1 | 91.89% | 91.89% | +0.00% | Regression Check |
| **Assamese** | Overall Entity F1 | 98.92% | 98.92% | +0.00% | Overall dev benchmark |
| | **LOC F1** | 98.73% | **98.73%** | **+0.00%** | Target Entity Type |
| | LOC Precision | 98.73% | 98.73% | +0.00% | False Positive Check |
| | LOC Recall | 98.73% | 98.73% | +0.00% | Ground Truth Coverage |
| | ORG F1 | 96.55% | 96.55% | +0.00% | Regression Check |
| | PER F1 | 100.00% | 100.00% | +0.00% | Regression Check |
| **Sinhala** | Overall Entity F1 | 99.24% | 96.30% | -2.94% | Overall dev benchmark |
| | **LOC F1** | 97.50% | **90.70%** | **-6.80%** | Target Entity Type |
| | LOC Precision | 95.12% | 82.98% | -12.14% | False Positive Check |
| | LOC Recall | 100.00% | 100.00% | +0.00% | Ground Truth Coverage |
| | ORG F1 | 100.00% | 100.00% | +0.00% | Regression Check |
| | PER F1 | 100.00% | 97.30% | -2.70% | Regression Check |

---

## 3. False-Positive Tradeoff Analysis

### Konkani
- **Baseline Counts**: TP=27, FP=0, FN=4
- **Retrained Counts**: TP=27, FP=0, FN=4
- **No Tradeoff**: Zero new false-positive LOC predictions introduced. Precision maintained at 100.00%.

### Assamese
- **Baseline Counts**: TP=78, FP=1, FN=1
- **Retrained Counts**: TP=78, FP=1, FN=1
- **No Tradeoff**: Zero new false-positive LOC predictions introduced. Precision maintained at 98.73%.

### Sinhala
- **Baseline Counts**: TP=39, FP=2, FN=0
- **Retrained Counts**: TP=39, FP=8, FN=0
- **Tradeoff Detected**: Introduced 6 new false-positive LOC prediction(s). Precision shifted from 95.12% to 82.98%.

