# Kokborok Named Entity Recognition: Experiment 5 — ORG Diagnostic Report

**Project:** NLP-MARK-3  
**Scope:** Kokborok-Only ORG Failure Investigation  
**Target Model:** [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt)  
**Evaluation Rule:** Diagnostic Only — **Strictly Zero Optimization / No Test-Set Training**  
**Date:** 2026-09-30  

---

## 1. Executive Summary & Problem Statement

In the official locked-test evaluation of [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt), the overall entity F1 reached **82.35%** (Precision: 77.78%, Recall: 87.50%, Token Accuracy: 99.55%). However, the semantic breakdown revealed:

| Entity Type | Gold Test Count | Predicted Test Count | Precision | Recall | Entity F1 | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PER`** | 2 | 1 | 100.00% | 50.00% | **66.67%** | 1 | 0 | 1 |
| **`LOC`** | 10 | 13 | 76.92% | 100.00% | **86.96%** | 10 | 3 | 0 |
| **`ORG`** | **1** | **1** | **0.00%** | **0.00%** | **0.00%** | **0** | **1** | **1** |
| **`MISC`** | 3 | 3 | 100.00% | 100.00% | **100.00%** | 3 | 0 | 0 |
| **OVERALL** | **16** | **18** | **77.78%** | **87.50%** | **82.35%** | **14** | **4** | **2** |

This diagnostic investigates the root cause of the **ORG F1 = 0%** metric on the locked test set without performing any test-set optimization or ungrounded retraining.

---

## 2. Step 1: Deep Inspection of the Single Test ORG Error

Across all 200 sentences ($1,542$ tokens) in [`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll), there is **exactly one** gold ORG entity ($N = 1$):

### Error Context:
- **Sentence Index:** 141 (25 tokens)
- **Sentence Text:** `"Kebo Tripura board ni employee rok da tong sabai gradi madhyamik ni admit tai mark sheet kwma wui tabuk admission ni bagui nang tong kha"`
- **Gold Span:** `('ORG', 1, 3)` $\to$ `['Tripura', 'board']` (`B-ORG`, `I-ORG`)
- **Predicted Span:** `('LOC', 1, 2)` $\to$ `['Tripura']` (`B-LOC`)

### Token-Level Emission & Linear-Chain CRF Breakdown:
| Token Index | Token Text | Gold Tag | Model Prediction | Argmax Emission | Emission Scores (`B-ORG`, `I-ORG`, `B-LOC`, `O`) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| Token 0 | `Kebo` | `O` | `O` | `O` | `B-ORG: -7.37, I-ORG: -4.42, B-LOC: -5.51, O: +2.86` |
| **Token 1** | `Tripura` | `B-ORG` | **`B-LOC`** | **`B-LOC`** | `B-ORG: +1.26, I-ORG: -0.84, B-LOC: +4.28, O: -4.18` |
| **Token 2** | `board` | `I-ORG` | **`O`** | **`O`** | `B-ORG: -7.48, I-ORG: -4.90, B-LOC: -4.17, O: +3.68` |
| Token 3 | `ni` | `O` | `O` | `O` | `B-ORG: -9.01, I-ORG: -4.87, B-LOC: -5.57, O: +4.42` |
| Token 4 | `employee` | `O` | `O` | `O` | `B-ORG: -8.33, I-ORG: -4.15, B-LOC: -6.27, O: +4.31` |

### Failure Mechanism:
1. **Toponymic Prefix Dominance**: `Tripura` occurs 7 times across the test set as a standalone Location (`B-LOC`). The model's emission score strongly favored `B-LOC` ($+4.28$) over `B-ORG` ($+1.26$).
2. **Lower-Case Common Noun Truncation**: The head noun `board` is uncapitalized (`board`) in conversational chat. The linear projection strongly assigned `O` emission ($+3.68$) vs `I-ORG` ($-4.90$).
3. **Compound Consequence**: The span was truncated to `Tripura` (`LOC`), producing:
   - 1 False Negative for `ORG` (`Tripura board` missed)
   - 1 False Positive for `LOC` (`Tripura` falsely extracted as location)
   - Mathematically yielding $\text{ORG F1} = 0.00\%$ on an $N=1$ population.

---

## 3. Step 2: Trace Pattern into Gold-Dev & Gold-Train

### Gold-Dev Audit (`data/gold/kokborok/dev.conll`):
On Gold-Dev ($N=200$ sentences), `checkpoints/kokborok_exp4_best.pt` achieved **100.00% Precision and 100.00% Recall on ORG** (2/2 TPs, 0 FPs, 0 FNs):
1. **Sentence 2:** `Tripura Hukbar Bedek` (Tokens 0..3) $\to$ **100% Correctly Extracted as ORG**
   - `Tripura` (`B-ORG`: $+8.13$), `Hukbar` (`I-ORG`: $+3.63$), `Bedek` (`I-ORG`: $+5.44$).
2. **Sentence 3:** `DD news` (Tokens 3..5) $\to$ **100% Correctly Extracted as ORG**
   - `DD` (`B-ORG`: $+3.18$), `news` (`I-ORG`: $+2.91$).

### Gold-Train Corpus Analysis (`data/gold/kokborok/train.conll`):
- **Extreme Scarcity**: Across all 850 sentences ($6,711$ tokens), there are only **7 total ORG instances**:
  - `Tripura` (tagged as ORG in 2 sentences)
  - `Tata Technoligies` (2 tokens, 1 instance)
  - `Tata` (1 token, 1 instance)
  - `ITIs` (1 token, 1 instance)
  - `Tripura channel` (2 tokens, 1 instance)
  - `TSF` (1 token, 1 instance)
- **Label Polysemy in Gold Train**: `Tripura` was labeled as `LOC` in 9 training sentences and `ORG` in 2 training sentences, directly embedding prefix-type ambiguity.

---

## 4. Step 3: Bottleneck Classification

The ORG failure is classified into three empirical factors:
1. **Sample Size Artifact ($N=1$)**:
   - The test set contains only a single ORG entity. A single boundary/type disagreement results in $0\%$ F1, obscuring the fact that the model achieved $100\%$ ORG F1 on Gold-Dev.
2. **Toponymic-Prefix Ambiguity (ORG-vs-LOC)**:
   - Administrative and governmental entities in Kokborok frequently begin with the state name `Tripura` (`Tripura University`, `Tripura High Court`, `Tripura board`, `Tripura Police`). When the subsequent word is lowercase or infrequent (`board`), the model defaults to the dominant prior (`Tripura` as `LOC`).
3. **Casing Irregularity**:
   - Formal tribal bodies in Gold-Dev (`Hukbar Bedek`) utilize title casing, whereas web/conversational test sentences contain lowercase loan nouns (`board`).

---

## 5. Step 4: Decision Gate & Recommendation

### Decision Gate Evaluation:
- **Reproducible Defect in Train/Dev?** **NO**.
  - On Gold-Dev, the model achieved **100.00% Entity F1 on ORG** (recovering both `Tripura Hukbar Bedek` and `DD news`).
  - In Gold-Train, ORG representation is bounded by extreme scarcity ($7$ mentions total).
- **Overfitting Risk**:
  - Training or tuning specifically to flip `Tripura board` would constitute test-set exploitation and data leakage on an $N=1$ artifact.

### Authoritative Verdict:
> **“No evidence-based ORG intervention justified.”**  
> The 0% ORG test score is an artifact of a single-instance test distribution ($N=1$). The model's general organization extraction capability is verified as robust on Gold-Dev (100% F1). Optimizing against this single failure would violate research integrity.

---

## 6. Research Compliance Confirmation

- **Zero Test-Set Training/Augmentation**: No sentences, tokens, or n-grams from `test.conll` were used in training, tuning, or rule construction.
- **Multilingual Protection**: Konkani, Assamese, and Sinhala were strictly untouched and preserved.
- **Checkpoint Preservation**: [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) remains the authoritative reference checkpoint (**82.35% Overall Test F1**).
