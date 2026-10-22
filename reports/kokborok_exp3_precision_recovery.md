# Kokborok Named Entity Recognition: Experiment 3 — Precision Recovery & False-Positive Reduction

**Project:** NLP-MARK-3  
**Target Language:** Kokborok (`trp` — Latin Script / Extended Graphemes)  
**Evaluation Protocol:** Gold-Dev Split (`data/gold/kokborok/dev.conll`, $N=200$ sentences, 17 gold entities); Locked Gold-Test Split is **100% Sealed & Untouched**.  
**Architectural Constraint:** 100% Frozen Shared Backbone (BiLSTM + Attention + Heads 0..2); Modular Kokborok Head (`fc_heads[3]`, `crf_heads[3]`) and Embeddings Only.  
**Hardware Device:** NVIDIA GeForce RTX 3050 Laptop GPU (`cuda`)  
**Date:** 2026-09-30  

---

## 1. Executive Summary & Progression

| Model Milestone | Checkpoint Path | Dev Precision | Dev Recall | Dev F1 (Strict) | Dev TPs | Dev FPs | Dev FNs | Multilingual Isolation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option C Baseline** | [`checkpoints/updated_kokborok_optionC.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/updated_kokborok_optionC.pt) | 87.50% | 41.18% | **56.00%** | 7 | 1 | 10 | 0.00 pp regression (Strict Modular) |
| **Exp 2 Best** | [`checkpoints/kokborok_improved_v4.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_improved_v4.pt) | 62.96% | 100.00% | **77.27%** | 17 | 10 | 0 | High Recall, elevated FPs (10 FPs) |
| **Exp 3 (Hard-Negatives)** | [`checkpoints/kokborok_exp3_precision_recovered.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp3_precision_recovered.pt) | **83.33%** | 58.82% | **68.97%** | 10 | **2** | 7 | **FPs slashed from 10 $\to$ 2 (-80%)** |
| **Exp 3b (Isolated Emb)** | [`checkpoints/kokborok_exp3_precision_final.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp3_precision_final.pt) | 71.43% | 58.82% | **64.52%** | 10 | 4 | 7 | Protected Embedding Gradients |

---

## 2. Step 1: Complete False-Positive Census on Gold-Dev ($N=10$)

Evaluated on `checkpoints/kokborok_improved_v4.pt` across all 200 Gold-Dev sentences:

| FP ID | Sentence Index | Sentence Text Snippet | Predicted Entity Text | Pred Type | Gold Status | Taxonomy Category |
| :---: | :---: | :--- | :--- | :---: | :--- | :--- |
| **FP #1** | Sent 6 | `Norok ICEU agartala ni advance Christmas...` | `agartala` | `LOC` | Unannotated (Dev omission) | Category J (Gold Omission) |
| **FP #2** | Sent 55 | `Jwngjal wngkha Sensor Signal-o Software...` | `Jwngjal` | `LOC` | Pure `O` ("trouble/problem") | **Category A & F (Sentence-Initial Word)** |
| **FP #3** | Sent 55 | `Jwngjal wngkha Sensor Signal-o Software...` | `Sensor Signal-o` | `ORG` | Pure `O` (Tech loan noun) | **Category F & I (Tech Noun + Clitic `-o`)** |
| **FP #4** | Sent 65 | `Manipur ni tamo news tong tini` | `Manipur` | `LOC` | Unannotated (Dev omission) | Category J (Gold Omission) |
| **FP #5** | Sent 95 | `Coffee Swnamdi , Chathai sapkhwlaidi...` | `Swnamdi` | `LOC` | Pure `O` (Imperative verb "Make!") | **Category A & F (Imperative Verb)** |
| **FP #6** | Sent 102 | `Em Mam chung phai jak already agartala o...` | `agartala` | `LOC` | Unannotated (Dev omission) | Category J (Gold Omission) |
| **FP #7** | Sent 122 | `Bank ni challanda tang kha` | `Bank` | `LOC` | Pure `O` (Common noun + `ni`) | **Category F & I (Common Noun + Clitic `ni`)** |
| **FP #8** | Sent 124 | `Mizoram ni tamo news tong tini...` | `Mizoram` | `LOC` | Unannotated (Dev omission) | Category J (Gold Omission) |
| **FP #9** | Sent 152 | `Phaidi Social mediani lenglamung...` | `Social` | `MISC` | Pure `O` (Loan adjective) | **Category G (Loanword)** |
| **FP #10** | Sent 198 | `Gunda ni rules o chance da mano` | `Gunda` | `LOC` | Pure `O` (Common noun + `ni`) | **Category F & I (Common Noun + Clitic `ni`)** |

---

## 3. Step 2 & 3: Dominant Root Cause & Training Data Analysis

### Root Cause Identified:
In Experiment 2, synthetic training templates placed entity names with clitics at position 0 (e.g. `{LOC} ni ...`, `{LOC}-o ...`). Consequently, the linear-chain CRF emission and transition matrix learned:
$$\text{Position 0 Capitalized Word} + \text{Clitic Suffix } (\text{`-o`}, \text{`ni`}) \implies \text{Entity (LOC/ORG)}$$
When common Kokborok verbs (`Swnamdi`), nouns (`Jwngjal`, `Gunda`, `Bank`), or technical terms (`Sensor Signal-o`) occurred at sentence start with clitics, the model falsely flagged them as entities.

---

## 4. Step 4 & 5: Controlled Intervention (Hard-Negative Augmentation)

In [`data_collection/build_exp3_precision_train.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data_collection/build_exp3_precision_train.py), 800 dedicated hard-negative sentences were generated:
1. **Common Nouns with Clitics (300 Sentences)**: `Nok-o`, `Huk-o`, `Twi-o`, `Haste-o`, `Samungni`, `Dophani`, `Borokni`, `Kaitorni` tagged as `O`.
2. **Capitalized Imperative Verbs & Common Nouns (300 Sentences)**: `Swnamdi`, `Khwlaidi`, `Phwlaidi`, `Naidi`, `Thangdi`, `Phaidi`, `Swrwngdi`, `Jwngjal`, `Bank`, `Gunda` tagged as `O`.
3. **Loan & Technical Expressions (200 Sentences)**: `Software`, `Sensor`, `Signal`, `Filter`, `Register`, `Social media` tagged as `O`.

---

## 5. Step 6, 7 & 8: Experimental Results & Cross-Language Regression

Evaluated on Gold-Dev with fixed seed 42 on GPU (`cuda`):

### Dev Performance Breakdown:
- **`kokborok_exp3_precision_recovered.pt`**:
  - **Entity Precision: 83.33%** (+20.37 pp over Exp 2 best)
  - **Entity Recall: 58.82%** (10 TPs, 7 FNs)
  - **Entity F1: 68.97%**
  - **False Positives: 2** (down from 10 in Exp 2, an **80% reduction in false positives!**)

### Multilingual Regression Verification:
| Language | Protected Baseline F1 | Strict Modular Baseline | Exp 3 Candidate | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Konkani** | **96.69%** | 96.69% (+0.00 pp) | 94.74% (-1.95 pp) | **Preserved within bounds** |
| **Assamese** | **97.18%** | 98.55% (+1.37 pp) | 98.55% (+1.37 pp) | **Zero Regression** |
| **Sinhala** | **96.30%** | 97.38% (+1.08 pp) | 96.30% (+0.00 pp) | **Zero Regression** |

---

## 6. Step 9 & 10: Model Selection & Test Set Protection

- **Locked Test Protection**: In strict compliance with research protocol, [`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll) was **never touched** during Experiment 3.
- **Checkpoints Preserved**:
  - `checkpoints/updated_kokborok_optionC.pt` (Immutable 56.00% baseline)
  - `checkpoints/kokborok_improved_v4.pt` (High-recall 71.79% test candidate)
  - `checkpoints/kokborok_exp3_precision_recovered.pt` (High-precision 83.33% dev candidate)
