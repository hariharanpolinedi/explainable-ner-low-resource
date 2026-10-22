# Kokborok Final Result Integrity Verification Audit Report

**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Audit Target**: Kokborok Final Evaluation & Optimization Progression (Rounds 1–4)  
**Auditor**: Senior ML/NLP Research Engineer  
**Date**: 2026-09-30  
**Device**: `NVIDIA GeForce RTX 3050 Laptop GPU` (CUDA 12.8 / PyTorch 2.6.0)  
**Status**: **CRITICAL METHODOLOGICAL BREACH CONFIRMED — 90.32% F1 RESULT REJECTED**

---

## Executive Summary

A mandatory verification audit was conducted to examine the claim that Kokborok achieved **90.32% entity-F1 ("TARGET MET")** following a four-round optimization progression. This audit evaluated:
1. Cryptographic test-set integrity
2. Adherence to the project's single-official-evaluation-run protocol
3. Test-set sample size viability and statistical sensitivity
4. Model generalization to genuinely fresh, never-seen Kokborok text
5. Round-by-round reproducibility on `gold-dev` (zero test exposure)
6. Cross-language negative transfer / regression on Konkani, Assamese, and Sinhala

### Key Findings
1. **Test-Set Integrity**: `data/gold/kokborok/test.conll` is cryptographically intact and matches its Phase 11 lockfile record (`de8ccf63...`) exactly.
2. **Single-Evaluation-Run Violation**: The official `test.conll` was accessed and evaluated against **over 15 times across 4 rounds**, directly violating the project's single-official-evaluation-run discipline. The reported 90.32% F1 was obtained by hill-climbing against the 16 test entities.
3. **Statistical Fragility**: The test set contains only **16 total entities across 200 sentences** (0.08 entities/sentence), with $\text{ORG}$ represented by literally $N=1$ instance. A single token flip moves class F1 by 100.0 percentage points.
4. **Generalization Failure**: On a verified zero-overlap set of 14 fresh Kokborok sentences containing 19 real entities, the model achieved **0.00% entity-F1**, proving that the 90.32% score was a memorization/tuning artifact.
5. **False Cross-Language Assertion**: The claim of "Zero Cross-Language Regression under Option C" was empirically refuted. Unfreezing the shared BiLSTM backbone during Kokborok retraining degraded **Sinhala by -19.40 pp** (97.38% $\rightarrow$ 77.98%) and **Konkani by -6.65 pp** (96.69% $\rightarrow$ 90.04%).

---

## 1. Test-Set Cryptographic Integrity Check

| Metric | Recorded in Lockfile ([`kokborok_gold_test_lockfile.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_gold_test_lockfile.md)) | Independently Recomputed from Raw File | Status |
|---|---|---|:---:|
| **File Path** | `data/gold/kokborok/test.conll` | `data/gold/kokborok/test.conll` | — |
| **SHA-256** | `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799` | `de8ccf63e6df8bf461508a12ebe2b5734ad1096da815e556a84146c1597a0799` | **EXACT MATCH (PASS)** |
| **Total Sentences** | 200 | 200 | **MATCH** |
| **Total Tokens** | 1,542 | 1,542 | **MATCH** |

> **Verdict**: The physical file `data/gold/kokborok/test.conll` has not been altered or corrupted.

---

## 2. Independent Gold Entity Census & Sample-Size Adequacy

An exhaustive, token-by-token parse of `data/gold/kokborok/test.conll` was performed independently:

### 2.1 Entity Distribution Census

| Entity Type | Instances in Kokborok Test Set ($N$) | Instances in Konkani Test Set | Instances in Assamese Test Set | Instances in Sinhala Test Set | Kokborok vs Other Languages |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **PER** | **2** | 20 | 19 | 38 | **~10x smaller** |
| **LOC** | **10** | 30 | 79 | 39 | **~4-8x smaller** |
| **ORG** | **1** | 37 | 14 | 28 | **~14-37x smaller** |
| **MISC** | **3** | 37 | 26 | 27 | **~9x smaller** |
| **TOTAL** | **16** | **124** | **138** | **132** | **8.2x smaller** |

### 2.2 Complete Manifest of All 16 Test Entities in Kokborok
* **Sentence 2**: `Christmas` (MISC)
* **Sentence 28**: `Mohan` (PER)
* **Sentence 44**: `Agartala` (LOC)
* **Sentence 63**: `Tripura` (LOC)
* **Sentence 64**: `Tripura` (LOC), `Tripura` (LOC)
* **Sentence 82**: `Rahul` (PER)
* **Sentence 109**: `Christmas` (MISC)
* **Sentence 116**: `Udaipur` (LOC)
* **Sentence 128**: `Tripura` (LOC), `Tripura` (LOC)
* **Sentence 141**: `Tripura board` (ORG) — *The sole organization mention in the entire split*
* **Sentence 152**: `Christmas` (MISC)
* **Sentence 163**: `Tripura` (LOC)
* **Sentence 190**: `Agartala` (LOC)
* **Sentence 199**: `Tripura` (LOC)

### 2.3 Instance-Flip Sensitivity Calculation
* **ORG Sensitivity ($N=1$)**:
  * 1 TP $\rightarrow$ Recall = 100%, F1 = 100.0%
  * 1 FN $\rightarrow$ Recall = 0%, F1 = 0.0%
  * **Sensitivity: 100.0 percentage points per single instance.**
* **PER Sensitivity ($N=2$)**:
  * Missing 1 instance swings recall by **50.0 pp** and F1 by **33.3 pp**.
* **Overall F1 Sensitivity**:
  * A single False Positive moves overall precision by **5.83 pp** and F1 by **2.82 pp**.
  * A single False Negative moves overall recall by **6.25 pp** and F1 by **3.65 pp**.
* **Comparison**: In Konkani ($N=124$) and Sinhala ($N=132$), a single instance flip shifts F1 by merely $\sim 0.6$–$0.7$ pp. Kokborok is **$5\times$ to $150\times$ more fragile**.

---

## 3. Single-Evaluation-Run Discipline Violation Audit

A forensic reconstruction was executed using script logs, execution history, and artifact traces:

### Evidence of Multiple Test-Set Invocations
1. **Round 1**: `kokborok_improvement_pipeline.py` invoked `evaluate_on_gold_test(ckpt_path)` on `test.conll` (reported 80.00%).
2. **Error Diagnosis**: `scratch/diagnose_errors.py` explicitly loaded `test.conll`, ran forward inference, and printed every individual false positive and false negative.
3. **Round 2**: `kokborok_round2_fixes.py` evaluated on `test.conll` to check if recall improved (reported 75.00%).
4. **Round 3**: `kokborok_round3_balanced.py` evaluated on `test.conll` across epochs (reported 71.79%).
5. **Round 4 (Surgical)**: `kokborok_surgical_finetune.py` ran test evaluations after **every single epoch (Epochs 1–6)**.
6. **Round 4 (Calibrated)**: `test_boundary_negative_tuning.py` evaluated on `test.conll` after **every single epoch (Epochs 1–8)** and picked Epoch 8 as "best" based on `test.conll` metrics.
7. **Direct Test Sentence Ingestion**: `test_boundary_negative_tuning.py` ingested verbatim strings from `test.conll` (e.g. `Rahul bike accident wngwi thwikha`, `Cherai no beginning simi...`, `Jones sakha je Swrwngthairok Los Angeles...`) as calibration targets.

> **Audit Finding**: The protocol of a single official evaluation on a cryptographically locked test set was completely breached. The 90.32% figure is the result of direct iterative parameter and data selection against `test.conll`. Under the scientific standards of this project, **the 90.32% result is null, void, and scientifically invalid**.

---

## 4. Round-by-Round Dev-Set Re-Verification (Zero Test Touch)

To measure actual learning trajectory without further test-set exposure, all available checkpoints were independently re-evaluated on `data/gold/kokborok/dev.conll` ($N=17$ entities):

| Round | Checkpoint | Dev Entity F1 | Dev Precision | Dev Recall | Token Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline (Phase 13)** | `staged_optionC_kokborok.pt` | 50.00% | 85.71% | 35.29% | 99.07% |
| **Round 1** | `kokborok_improved_v2.pt` | **87.18%** | 77.27% | **100.00%** | 99.50% |
| **Round 2** | `kokborok_improved_v3.pt` | 79.07% | 65.38% | 100.00% | 99.38% |
| **Round 3** | `kokborok_improved_v4.pt` | 77.27% | 62.96% | 100.00% | 99.32% |
| **Round 4** | `kokborok_calib_best.pt` | 88.24% | 88.24% | 88.24% | 99.69% |

* Dev evaluation confirms the exact oscillation pattern seen in the test results: Round 1 achieved solid generalization (+37.18 pp over baseline), while Rounds 2 and 3 degraded precision due to aggressive synthetic overfitting.

---

## 5. Cross-Language Regression Verification

The previous report claimed: *"Zero Cross-Language Regression: Option-C guarantee preserved — Konkani, Assamese, and Sinhala heads remained strictly frozen."*

An empirical audit was executed evaluating the checkpoints on each language's official `gold-dev` split:

| Checkpoint | Konkani Dev F1 | Assamese Dev F1 | Sinhala Dev F1 | Verdict |
|---|:---:|:---:|:---:|---|
| **Phase 14 Reference Baseline** | **96.69%** | **98.55%** | **97.38%** | Official Protected Baseline |
| **Round 1 (`kokborok_improved_v2.pt`)** | 90.16% (**-6.53 pp**) | 97.08% (**-1.47 pp**) | 91.70% (**-5.68 pp**) | **Substantial Negative Transfer** |
| **Round 4 (`kokborok_calib_best.pt`)** | 90.04% (**-6.65 pp**) | 94.30% (**-4.25 pp**) | 77.98% (**-19.40 pp**) | **Catastrophic Sinhala Regression** |

### Mechanism of Cross-Language Interference
* Although the language-specific classification heads (`fc_heads.{0,1,2}`) and CRF transitions (`crf_heads.{0,1,2}`) were frozen, the **shared BiLSTM encoder weights and shared token embeddings were NOT frozen**.
* Kokborok's Latin-script backpropagation updated the shared sequence representations, severely destabilizing Sinhala's non-Latin conjunct representations and Konkani's Devanagari feature space.
* **The "Zero Regression" claim was factually false.**

---

## 6. Fresh-Sentence Generalization Check

Fourteen verified zero-overlap sentences (19 gold entities: 13 PER, 4 LOC, 1 ORG, 1 MISC) were sourced from open Kokborok text (`sdmy/kokborok`):

* **Overlap with all training/tuning sets**: **0.00%** (Verified).
* **Evaluation Results**:
  * **Phase 13 Baseline**: **0.00% F1** (0/19 recalled)
  * **Round 1 Model**: **0.00% F1** (0/19 recalled, 3 FP)
  * **Round 4 Model**: **0.00% F1** (0/19 recalled, 0 FP)
  * **Final Deployed Model**: **0.00% F1** (0/19 recalled, 0 FP)

The model was unable to recognize authentic Kokborok named entities out-of-domain. While token-level accuracy remained at ~83.75% on negative sentences, entity-level recall collapsed completely.

---

## 7. Audit Verdict & Corrective Actions

1. **Rejection of 90.32% Claim**: The "90.32% TARGET MET" headline is rejected.
2. **Defensible Historical Benchmark**: The only valid single-run benchmark for Kokborok is the original **Phase 13 single-run evaluation: 68.97% F1** (or Round 1's exploratory 80.00%), which must be explicitly qualified by $N=16$ sample size.
3. **Mandatory Reporting Policy**: Phase 16 reporting must describe Kokborok as an **exploratory low-resource prototype constrained by severe corpus and test-set sparsity**, rather than a verified 90%+ production system.
