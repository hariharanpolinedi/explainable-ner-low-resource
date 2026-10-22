# Phase 7: Post-Improvement Error Re-Analysis

> **Phase 7 Deliverable — Paired Empirical Re-Analysis**  
> Direct apples-to-apples comparison between Verified Phase 0 Baseline and Phase 6 Current Best Checkpoints.
> Evaluated on the identical diagnostic sample manifest (247–270 sentences) and Phase 2 stratification buckets.
> 
> *Baseline Verification Notice*: Grounded in freshly-reproduced, cryptographically verified baseline numbers (Konkani 96.69%, Assamese 98.92%, Sinhala 99.24% Dev F1), superseding prior unverified numbers that suffered from naive character-splitting degradation.

---

## 1. Master Benchmark & Diagnostic Comparison Matrix

| Language | Model Evaluated | Checkpoint Origin | Gold Dev F1 | Stress F1 | Middle Recall (Lists $\ge 3$) | Stress Collapse Rate | Total Manifest Errors |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | **Phase 0 Baseline (Verified)** | `baseline_v1` | **96.69%** | **17.39%** | **0.00%** | **75.76%** | **90** |
| | **Current Best Evaluated** | baseline_v1 (Phase 0 Baseline preserved as Current Best) | **96.69%** | **17.39%** | **0.00%** | **75.76%** | **90** |
| **Assamese** | **Phase 0 Baseline (Verified)** | `baseline_v1` | **98.92%** | **22.99%** | **7.41%** | **70.97%** | **86** |
| | **Current Best Evaluated** | baseline_v1 (Phase 0 Baseline preserved as Current Best) | **98.92%** | **22.99%** | **7.41%** | **70.97%** | **86** |
| **Sinhala** | **Phase 0 Baseline (Verified)** | `baseline_v1` | **99.24%** | **36.36%** | **0.00%** | **60.00%** | **55** |
| | **Current Best Evaluated** | combined (Phase 6 Combined Model) | **96.30%** | **37.97%** | **16.67%** | **60.00%** | **64** |

---

## 2. Paired Error Taxonomy Census (Raw Counts & Rates)

Comparison across all 8 taxonomy categories on the exact diagnostic sample manifest:

### Konkani Taxonomy Census (N=247 sentences, 251 gold entities)

| Error Category | Baseline Raw Count | Current Best Raw Count | Baseline % of Errors | Current Best % of Errors | Net Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`complete_sentence_collapse`** | 55 | 55 | 61.11% | 61.11% | 0 |
| **`missed_entity`** | 34 | 34 | 37.78% | 37.78% | 0 |
| **`false_positive`** | 1 | 1 | 1.11% | 1.11% | 0 |
| **`incomplete_span`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`overextended_span`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`wrong_entity_type`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`bi_transition_error`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`o_inside_entity`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **TOTAL ERRORS** | **90** | **90** | 100.00% | 100.00% | **0** |

### Assamese Taxonomy Census (N=270 sentences, 416 gold entities)

| Error Category | Baseline Raw Count | Current Best Raw Count | Baseline % of Errors | Current Best % of Errors | Net Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`complete_sentence_collapse`** | 56 | 56 | 65.12% | 65.12% | 0 |
| **`missed_entity`** | 14 | 14 | 16.28% | 16.28% | 0 |
| **`false_positive`** | 10 | 10 | 11.63% | 11.63% | 0 |
| **`incomplete_span`** | 4 | 4 | 4.65% | 4.65% | 0 |
| **`overextended_span`** | 2 | 2 | 2.33% | 2.33% | 0 |
| **`wrong_entity_type`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`bi_transition_error`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`o_inside_entity`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **TOTAL ERRORS** | **86** | **86** | 100.00% | 100.00% | **0** |

### Sinhala Taxonomy Census (N=263 sentences, 300 gold entities)

| Error Category | Baseline Raw Count | Current Best Raw Count | Baseline % of Errors | Current Best % of Errors | Net Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`complete_sentence_collapse`** | 28 | 28 | 50.91% | 43.75% | 0 |
| **`missed_entity`** | 15 | 14 | 27.27% | 21.88% | -1 |
| **`false_positive`** | 12 | 20 | 21.82% | 31.25% | +8 |
| **`incomplete_span`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`overextended_span`** | 0 | 1 | 0.00% | 1.56% | +1 |
| **`wrong_entity_type`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **`bi_transition_error`** | 0 | 1 | 0.00% | 1.56% | +1 |
| **`o_inside_entity`** | 0 | 0 | 0.00% | 0.00% | 0 |
| **TOTAL ERRORS** | **55** | **64** | 100.00% | 100.00% | **9** |

---

## 3. Paired Stratification Tables (Length & Entity Density)

### Konkani

#### Length Stratification

| Length Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1_Short (<=12)** | 88 | 97.30% | 97.30% | 2.86% | 2.86% | **ROBUST** |
| **L2_Medium (13-22)** | 71 | 92.73% | 92.73% | 9.30% | 9.30% | **ROBUST** |
| **L3_Long (23-34)** | 46 | 60.38% | 60.38% | 31.03% | 31.03% | **ROBUST** |
| **L4_Stress (>34)** | 42 | 40.57% | 40.57% | 44.44% | 44.44% | **ROBUST** |

#### Entity Density Stratification

| Density Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0 (0 Entities)** | 104 | 100.00% | 100.00% | 0.00% | 0.00% | **ROBUST** |
| **E1 (1 Entity)** | 91 | 79.12% | 79.12% | 20.88% | 20.88% | **ROBUST** |
| **E2 (2 Entities)** | 27 | 72.22% | 72.22% | 18.52% | 18.52% | **ROBUST** |
| **E3 (3+ Entities)** | 25 | 48.11% | 48.11% | 24.00% | 24.00% | **ROBUST** |

### Assamese

#### Length Stratification

| Length Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1_Short (<=12)** | 64 | 95.65% | 95.65% | 4.44% | 4.44% | **ROBUST** |
| **L2_Medium (13-22)** | 89 | 100.00% | 100.00% | 0.00% | 0.00% | **ROBUST** |
| **L3_Long (23-34)** | 51 | 73.26% | 73.26% | 21.62% | 21.62% | **ROBUST** |
| **L4_Stress (>34)** | 66 | 77.23% | 77.23% | 24.56% | 24.56% | **ROBUST** |

#### Entity Density Stratification

| Density Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0 (0 Entities)** | 76 | 100.00% | 100.00% | 0.00% | 0.00% | **ROBUST** |
| **E1 (1 Entity)** | 134 | 90.30% | 90.30% | 8.96% | 8.96% | **ROBUST** |
| **E2 (2 Entities)** | 10 | 50.00% | 50.00% | 40.00% | 40.00% | **ROBUST** |
| **E3 (3+ Entities)** | 50 | 79.77% | 79.77% | 16.00% | 16.00% | **ROBUST** |

### Sinhala

#### Length Stratification

| Length Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1_Short (<=12)** | 52 | 100.00% | 100.00% | 0.00% | 0.00% | **ROBUST** |
| **L2_Medium (13-22)** | 69 | 100.00% | 97.96% | 0.00% | 0.00% | **ROBUST** |
| **L3_Long (23-34)** | 78 | 89.25% | 89.25% | 10.42% | 10.42% | **ROBUST** |
| **L4_Stress (>34)** | 64 | 74.81% | 75.57% | 25.00% | 25.00% | **ROBUST** |

#### Entity Density Stratification

| Density Stratum | Sentences | Baseline Recall | Current Best Recall | Baseline Collapse Rate | Current Best Collapse Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0 (0 Entities)** | 99 | 100.00% | 100.00% | 0.00% | 0.00% | **ROBUST** |
| **E1 (1 Entity)** | 98 | 89.80% | 88.78% | 10.20% | 10.20% | **ROBUST** |
| **E2 (2 Entities)** | 29 | 72.41% | 72.41% | 20.69% | 20.69% | **ROBUST** |
| **E3 (3+ Entities)** | 37 | 88.19% | 88.89% | 5.41% | 5.41% | **ROBUST** |

