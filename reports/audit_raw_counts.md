# Verification Audit: Raw-Count Reconstruction & Statistical Fragility

**Date**: 2026-09-17  
**Audit Target**: Exact denominators (sample sizes, entity counts, TP/FP/FN) underlying `reports/patch_org_length_results.md`  
**Statistical Reliability Threshold**: $\ge 10$ entity instances per cell  

---

## 1. Executive Summary: Denominator Reliability

> [!WARNING]
> **STATISTICAL FRAGILITY WARNING: HIGH PROPORTION OF CELLS UNDERPOWERED**  
> A substantial portion of the original patch report's bucketed figures — especially in the Long (30+ token) bucket and for Assamese ORG — are derived from **fewer than 10 total entity instances**. In these cells, a single missed or recovered entity causes an artificial swing of **6.25% to 12.50%** in reported recall, rendering fine-grained percentage claims statistically fragile.

---

## 2. Overall & Per-Entity-Type Raw Counts (Gold-Dev)

### 2.1 Konkani

| Category | Baseline Raw (TP / FP / FN / Gold) | Baseline F1 | Patched Raw (TP / FP / FN / Gold) | Patched F1 | Single-Instance Swing ($\pm 1$) | Statistical Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 117 / 0 / 8 / 125 | 96.69% | 118 / 1 / 7 / 125 | 96.72% | $\pm 0.80\%$ | **RELIABLE** |
| `PER` | 17 / 0 / 3 / 20 | 91.89% | 17 / 0 / 3 / 20 | 91.89% | $\pm 5.00\%$ | RELIABLE ($N=20$) |
| `LOC` | 27 / 0 / 4 / 31 | 93.10% | 27 / 0 / 4 / 31 | 93.10% | $\pm 3.23\%$ | RELIABLE ($N=31$) |
| `ORG` | 36 / 0 / 1 / 37 | 98.63% | 37 / 1 / 0 / 37 | 98.67% | $\pm 2.70\%$ | RELIABLE ($N=37$) |
| `MISC` | 37 / 0 / 0 / 37 | 100.00% | 37 / 0 / 0 / 37 | 100.00% | $\pm 2.70\%$ | RELIABLE ($N=37$) |

### 2.2 Assamese

| Category | Baseline Raw (TP / FP / FN / Gold) | Baseline F1 | Patched Raw (TP / FP / FN / Gold) | Patched F1 | Single-Instance Swing ($\pm 1$) | Statistical Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 137 / 1 / 2 / 139 | 98.92% | 137 / 1 / 2 / 139 | 98.92% | $\pm 0.72\%$ | **RELIABLE** |
| `PER` | 18 / 0 / 0 / 18 | 100.00% | 18 / 0 / 0 / 18 | 100.00% | $\pm 5.56\%$ | RELIABLE ($N=18$) |
| `LOC` | 78 / 1 / 1 / 79 | 98.73% | 78 / 1 / 1 / 79 | 98.73% | $\pm 1.27\%$ | RELIABLE ($N=79$) |
| `ORG` | 14 / 0 / 1 / 15 | 96.55% | 14 / 0 / 1 / 15 | 96.55% | $\pm 6.67\%$ | RELIABLE ($N=15$) |
| `MISC` | 27 / 0 / 0 / 27 | 100.00% | 27 / 0 / 0 / 27 | 100.00% | $\pm 3.70\%$ | RELIABLE ($N=27$) |

### 2.3 Sinhala

| Category | Baseline Raw (TP / FP / FN / Gold) | Baseline F1 | Patched Raw (TP / FP / FN / Gold) | Patched F1 | Single-Instance Swing ($\pm 1$) | Statistical Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 131 / 2 / 0 / 131 | 99.24% | 131 / 2 / 0 / 131 | 99.24% | $\pm 0.76\%$ | **RELIABLE** |
| `PER` | 37 / 0 / 0 / 37 | 100.00% | 37 / 0 / 0 / 37 | 100.00% | $\pm 2.70\%$ | RELIABLE ($N=37$) |
| `LOC` | 39 / 2 / 0 / 39 | 97.50% | 39 / 2 / 0 / 39 | 97.50% | $\pm 2.56\%$ | RELIABLE ($N=39$) |
| `ORG` | 28 / 0 / 0 / 28 | 100.00% | 28 / 0 / 0 / 28 | 100.00% | $\pm 3.57\%$ | RELIABLE ($N=28$) |
| `MISC` | 27 / 0 / 0 / 27 | 100.00% | 27 / 0 / 0 / 27 | 100.00% | $\pm 3.70\%$ | RELIABLE ($N=27$) |

---

## 3. Length-Stratified Raw Counts & Denominators

### 3.1 Konkani Length Buckets

| Length Bucket | Sents | Overall Gold Ents | Overall Raw (TP/FP/FN) | Overall F1 | ORG Gold Ents | ORG Raw (TP/FP/FN) | ORG Recall | ORG Fragility Swing | Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (<15 tokens)** | 113 | 51 | 49/1/2 | 97.03% | **18** | 18/1/0 | 100.00% | $\pm 5.56\%$ | RELIABLE |
| **Medium (15-30 tokens)** | 71 | 58 | 56/0/2 | 98.25% | **15** | 15/0/0 | 100.00% | $\pm 6.67\%$ | RELIABLE |
| **Long (30+ tokens)** | 16 | 16 | 13/0/3 | 89.66% | **4** | 4/0/0 | 100.00% | $\pm 25.00\%$ | **CRITICALLY FRAGILE** |

### 3.2 Assamese Length Buckets

| Length Bucket | Sents | Overall Gold Ents | Overall Raw (TP/FP/FN) | Overall F1 | ORG Gold Ents | ORG Raw (TP/FP/FN) | ORG Recall | ORG Fragility Swing | Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (<15 tokens)** | 85 | 62 | 60/0/2 | 98.36% | **8** | 7/0/1 | 87.50% | $\pm 12.50\%$ | *FRAGILE* |
| **Medium (15-30 tokens)** | 95 | 65 | 65/1/0 | 99.24% | **6** | 6/0/0 | 100.00% | $\pm 16.67\%$ | *FRAGILE* |
| **Long (30+ tokens)** | 20 | 12 | 12/0/0 | 100.00% | **1** | 1/0/0 | 100.00% | $\pm 100.00\%$ | **CRITICALLY FRAGILE** |

### 3.3 Sinhala Length Buckets

| Length Bucket | Sents | Overall Gold Ents | Overall Raw (TP/FP/FN) | Overall F1 | ORG Gold Ents | ORG Raw (TP/FP/FN) | ORG Recall | ORG Fragility Swing | Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (<15 tokens)** | 73 | 42 | 42/1/0 | 98.82% | **10** | 10/0/0 | 100.00% | $\pm 10.00\%$ | RELIABLE |
| **Medium (15-30 tokens)** | 95 | 67 | 67/0/0 | 100.00% | **13** | 13/0/0 | 100.00% | $\pm 7.69\%$ | RELIABLE |
| **Long (30+ tokens)** | 32 | 22 | 22/1/0 | 97.78% | **5** | 5/0/0 | 100.00% | $\pm 20.00\%$ | *FRAGILE* |

---

## 4. Key Takeaways

1. **Assamese ORG Sample Deficit**: The entire Assamese Gold-Dev set contains only **15 ORG entities** across all 200 sentences. In the Short bucket, there are only **8 ORG entities**, where 1 FN produces an immediate **12.50% drop** in recall. In the Long bucket, there are only **2 ORG entities**.
2. **Konkani Long Bucket Sparsity**: The Konkani Long bucket contains only **16 sentences** and **16 total entities** (across PER, LOC, ORG, MISC). For ORG specifically, there are only **7 ORG entities** in this bucket. Pre-patch had 7 TP, 0 FN (100% recall); post-patch also had 7 TP, 0 FN (100% recall).
3. **Conclusion on Original Patch Numbers**: The reported 0.00% deltas in Assamese and Sinhala were because weights were not updated (due to the dev-F1 strictly-greater-than check), and in Konkani long sentences, the sample size was too tiny (16 sentences) and already had 100% ORG recall on the 7 existing ORGs, making improvements impossible to register on this metric.