# Kokborok Final Verdict (Corrected Scientific Assessment)

**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Document**: Authoritative Corrected Verdict & Scientific Remediation  
**Auditor**: Senior ML/NLP Research Engineer  
**Date**: 2026-09-30  
**Status**: **FINAL AUDIT COMPLETE — CLAIMS CORRECTED & STANDARDS RESTORED**

---

## 1. Executive Verdict on the "90.32% F1, TARGET MET" Claim

The claim that Kokborok achieved **90.32% Entity F1 ("TARGET MET")** is **categorically rejected as scientifically invalid and indefensible**. 

This verdict is supported by four independent lines of empirical proof:

### Proof 1: Catastrophic Evaluation Under-Powering ($N=16$)
* The locked gold-test split contains only **16 total entities across 200 sentences** ($0.08$ entities/sentence).
* Category representation is severely sparse:
  * **ORG**: Exactly **1 instance** (`Tripura board`). A single token boundary determines whether ORG is 0.0% or 100.0%.
  * **PER**: Exactly **2 instances** (`Mohan`, `Rahul`).
  * **MISC**: Exactly **3 instances** (all three are the word `"Christmas"`).
  * **LOC**: Exactly **10 instances** (7 are `"Tripura"`, 2 are `"Agartala"`, 1 is `"Udaipur"`).
* In comparison, the test sets for Konkani ($N=124$), Assamese ($N=138$), and Sinhala ($N=132$) are **8x to 9x larger** with balanced distributions across all four classes.
* Reporting an exact metric like "90.32%" on 16 instances presents a false precision that is statistically indistinguishable from noise.

### Proof 2: Violation of Single-Evaluation-Run Discipline & Test Contamination
* The official `data/gold/kokborok/test.conll` was loaded, evaluated, and inspected **over 15 times** across four optimization rounds.
* In Round 4, exact verbatim sentences and tokens from `test.conll` were ingested into training scripts as calibration targets.
* The reported 90.32% F1 was the direct product of **hill-climbing against the test set's exact vocabulary**, violating the project's single-official-run guarantee.

### Proof 3: Zero Generalization on Genuinely Fresh Kokborok Sentences
* On a verified zero-overlap set of 14 authentic Kokborok sentences containing 19 real named entities sourced from public corpora, the model achieved **0.00% Entity F1** (0 out of 19 entities recognized).
* The model did not learn generalized neural entity extraction; it memorized the specific lexical patterns of the 16 test instances.

### Proof 4: Unreported Negative Transfer on Existing Languages
* The claim of "Zero Cross-Language Regression under Option C" was empirically refuted.
* Because the shared BiLSTM sequence encoder and word embeddings remained trainable during Kokborok retraining, backpropagation of Latin-script gradients caused severe cross-lingual interference:
  * **Sinhala Dev F1 plunged by -19.40 pp** (97.38% $\rightarrow$ **77.98%**)
  * **Konkani Dev F1 dropped by -6.65 pp** (96.69% $\rightarrow$ **90.04%**)
  * **Assamese Dev F1 dropped by -4.25 pp** (98.55% $\rightarrow$ **94.30%**)

---

## 2. What Is the Actually Defensible Reported Result for Kokborok?

Under the scientific standards of this repository, the only defensible reported benchmark for Kokborok is:

### The Defensible Baseline: Phase 13 Single Official Run (68.97% F1)

| Language | Test Entity F1 | Token Accuracy | Gold Test Entities ($N$) | Evaluation Integrity | Research Target Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Konkani** | **96.27%** | 99.47% | 124 | Single Locked Run (Verified) | **TARGET MET (+6.27 pp)** |
| **Assamese** | **97.87%** | 99.64% | 138 | Single Locked Run (Verified) | **TARGET MET (+7.87 pp)** |
| **Sinhala** | **97.76%** | 99.70% | 132 | Single Locked Run (Verified) | **TARGET MET (+7.76 pp)** |
| **Kokborok** | **68.97%** | 99.55% | 16 | Phase 13 Single Run *(Statistically Under-Powered)* | **SUB-TARGET (-21.03 pp)** |

### Mandatory Reporting Requirements in Phase 16 / Research Publications
1. **Explicit Sample-Size Disclosure**: Every mention of Kokborok's test performance must disclose the sample size: *(N=16 entities: 10 LOC, 3 MISC, 2 PER, 1 ORG in 200 sentences)*.
2. **Honest Characterization**: Kokborok must be framed as an **exploratory feasibility study of a fourth, typologically distant low-resource language under severe data scarcity**, rather than a verified >90% production pipeline.
3. **No Conflation with High-Resource Baselines**: Kokborok's numbers must never be presented side-by-side with Konkani, Assamese, and Sinhala without explicitly highlighting the 8x sample-size disparity and the absence of institutional organizational corpora.

---

## 3. Remediation & Action Log

| Component | Remediation Action Taken |
|---|---|
| **Deployment Bundle** | Checkpoint restored to pre-contamination state. False "TARGET MET" labels removed. |
| **Web Application (`app.py`)** | Benchmark telemetry corrected to reflect honest exploratory status with explicit N=16 caveat. |
| **Option-C Integrity** | Established that any future Kokborok fine-tuning must strictly freeze `bilstm.*`, `char_cnn.*`, `word_embedding.*`, and `attention_blocks.*`. |
| **Audit Trail** | All evidence, checksums, logs, and comparative tables preserved in `reports/kokborok_result_integrity_audit.md`. |

---

## 4. Final Scientific Conclusion

This audit successfully protected the academic and scientific integrity of the research paper and project deliverables. By catching the test-set leakage, empirical cross-language regression, and extreme sample-size fragility before submission to reviewers or faculty, we preserve the credibility of the primary findings on Konkani, Assamese, and Sinhala, and present an honest, rigorous account of Kokborok.
