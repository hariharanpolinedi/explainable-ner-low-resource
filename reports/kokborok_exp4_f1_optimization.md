# Kokborok Named Entity Recognition: Experiment 4 — F1 Optimization Report

**Project:** NLP-MARK-3  
**Scope:** Kokborok-Only Optimization  
**Target Language:** Kokborok (`trp` — Latin Script / Extended Graphemes)  
**Evaluation Constraint:** Single Official Evaluation on Locked Gold-Test Split ([`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll)) conducted strictly once at cycle close.  
**Hardware Device:** NVIDIA GeForce RTX 3050 Laptop GPU (`cuda`, CUDA 12.6, PyTorch 2.1.2+cu121)  
**Date:** 2026-09-30  

---

## 1. Executive Summary & Progression

| Model Milestone | Checkpoint Path | Dev F1 | Official Test Precision | Official Test Recall | **Official Test F1** | Test TP / FP / FN | Net F1 vs Benchmark (71.79%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 13 Baseline** | [`checkpoints/final_kokborok.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/final_kokborok.pt) | 50.00% | 76.92% | 62.50% | **68.97%** | 10 / 3 / 6 | -2.82 pp | Original baseline |
| **Option C Baseline** | [`checkpoints/updated_kokborok_optionC.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/updated_kokborok_optionC.pt) | 56.00% | — | — | — | 7 / 1 / 10 | — | Staged modular baseline |
| **Exp 2 Best** | [`checkpoints/kokborok_improved_v4.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_improved_v4.pt) | 77.27% | 60.87% | 87.50% | **71.79%** | 14 / 9 / 2 | Reference Benchmark | High recall, elevated FPs |
| **Exp 3 (Hard-Negatives)** | [`checkpoints/kokborok_exp3_precision_recovered.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp3_precision_recovered.pt) | 68.97% | — | — | — | 10 / 2 / 7 | — | High precision, lower recall |
| **Exp 4 Final Candidate** | [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) | **94.44%** | **77.78%** | **87.50%** | **82.35%** | **14 / 4 / 2** | **+10.56 pp** | **ACCEPTED (NEW BEST)** |

---

## 2. Step 1: Audit of Existing Kokborok Candidates on Gold-Dev

Evaluated strictly on [`data/gold/kokborok/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/dev.conll) ($N=200$ sentences, 1,612 tokens, 17 gold entities):

| Checkpoint Name | Precision | Recall | Entity F1 | TP | FP | FN | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `final_kokborok.pt` | 85.71% | 35.29% | 50.00% | 6 | 1 | 11 | Phase 13 baseline |
| `updated_kokborok_optionC.pt` | 87.50% | 41.18% | 56.00% | 7 | 1 | 10 | Option C baseline |
| `kokborok_improved_v2.pt` | **77.27%** | **100.00%** | **87.18%** | **17** | **5** | **0** | **Strongest recall-preserving base** |
| `kokborok_improved_v4.pt` | 62.96% | 100.00% | 77.27% | 17 | 10 | 0 | Elevated false positives |
| `kokborok_exp3_precision_recovered.pt` | 83.33% | 58.82% | 68.97% | 10 | 2 | 7 | Excessive recall penalty |

---

## 3. Step 2 & 3: Complete Kokborok Error Matrix on Gold-Dev

Evaluated on `kokborok_improved_v2.pt` vs `kokborok_improved_v4.pt`:

```
================================================================================
KOKBOROK ERROR MATRIX (GOLD-DEV)
================================================================================
Entity Type | True Positives | False Positives | False Negatives | Boundary Errors | Type Confusions
--------------------------------------------------------------------------------
PER         | 2              | 0               | 0               | 0               | 0
LOC         | 9              | 3               | 0               | 0               | 0
ORG         | 2              | 1               | 0               | 0               | 0
MISC        | 4              | 1               | 0               | 0               | 0
TOTAL       | 17             | 5               | 0               | 0               | 0
================================================================================
```

### Analysis of the 5 "False Positives" in `kokborok_improved_v2.pt`:
1. `agartala` (Sent 6) — Actual capital of Tripura, unannotated in Gold Dev.
2. `agartala` (Sent 102) — Actual capital of Tripura, unannotated in Gold Dev.
3. `Mizoram` (Sent 124) — Actual Indian state, unannotated in Gold Dev.
4. `Sensor Signal-o` (Sent 55) — Technical loan noun with locative clitic `-o`.
5. `Social mediani lenglamung` (Sent 152) — Loan expression with genitive clitic `ni`.

**Key Finding:** 3 out of the 5 apparent FPs on Dev were genuine geographic entities omitted in the small dev set, while only 2 were genuine non-entity loanword false triggers.

---

## 4. Step 5 & 6: Experiment 4 Multi-Seed Optimization

- **Hypothesis**: Warm-starting from `kokborok_improved_v2.pt` and applying conservative fine-tuning ($\text{lr}_{\text{head}} = 5 \times 10^{-5}, \text{lr}_{\text{emb}} = 3 \times 10^{-5}$) with hard-negative loanword regularization suppresses non-entity triggers (`Sensor Signal-o`, `Social media`) while preserving 100% recall.
- **Multi-Seed Stability Sweep (Dev F1)**:
  - Seed 42:   **94.44% F1** (P: 89.47%, R: 100.00%, TP: 17, FP: 2, FN: 0)
  - Seed 123:  **94.44% F1** (P: 89.47%, R: 100.00%, TP: 17, FP: 2, FN: 0)
  - Seed 2026: **94.44% F1** (P: 89.47%, R: 100.00%, TP: 17, FP: 2, FN: 0)
  - **Mean Dev F1**: **$94.44\% \pm 0.00\%$** (Perfect stability across seeds).

---

## 5. Step 12: Official Locked-Test Evaluation

Conducted strictly once on [`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll) ($N=200$ sentences, 1,542 tokens, 16 gold entities):

| Metric | Previous Official Best (`v4`) | Experiment 4 Candidate (`kokborok_exp4_best.pt`) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Entity Precision** | 60.87% | **77.78%** | **+16.91 pp** |
| **Entity Recall** | 87.50% | **87.50%** | **0.00 pp (Preserved)** |
| **Entity F1 (Strict)** | **71.79%** | **82.35%** | **+10.56 pp** |
| **Token Accuracy** | 99.16% | **99.55%** | **+0.39 pp** |
| **True Positives (TP)** | 14 | **14** | 0 |
| **False Positives (FP)** | 9 | **4** | **-5 (-55.6% reduction)** |
| **False Negatives (FN)** | 2 | **2** | 0 |

### Per-Entity-Type Performance on Locked Test:
| Entity Class | Precision | Recall | Entity F1 | TP | FP | FN | Gold Support | Pred Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PER`** | 100.00% | 50.00% | **66.67%** | 1 | 0 | 1 | 2 | 1 |
| **`LOC`** | 76.92% | 100.00% | **86.96%** | 10 | 3 | 0 | 10 | 13 |
| **`ORG`** | 0.00% | 0.00% | **0.00%** | 0 | 1 | 1 | 1 | 1 |
| **`MISC`** | 100.00% | 100.00% | **100.00%** | 3 | 0 | 0 | 3 | 3 |
| **OVERALL** | **77.78%** | **87.50%** | **82.35%** | **14** | **4** | **2** | **16** | **18** |

---

## 6. Conclusion & Acceptance Decision

1. **Target Achieved**: Official locked-test Entity F1 advanced from **$71.79\% \to 82.35\%$** (**+10.56 percentage points**), driven by slashing false positives from $9 \to 4$ while maintaining **$87.50\%$ Recall**.
2. **Final Checkpoint**: [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) is designated as the authoritative best Kokborok NER model.
3. **Artifact Protection**: All baseline and intermediate checkpoints (`updated_kokborok_optionC.pt`, `kokborok_improved_v4.pt`, `kokborok_exp3_precision_recovered.pt`) remain archived and intact.
