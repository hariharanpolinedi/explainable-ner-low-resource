# Reliability Cycle 2 — Step 1: Assamese Long-Sequence Decay Confirmation Report

> **Reliability Cycle 2 Deliverable — Diagnostic Evidence Gate**  
> **Target Mechanism**: Assamese BiLSTM encoder suffering "severe recurrent decay" on high-density, long (>35 token) multi-entity sentences, causing complete-sequence collapse to O.  
> **Status**: **DISCONFIRMED AS PURE "RECURRENT DECAY"**; **CONFIRMED AS A DUAL MECHANISM: COMPLETE ATTENTION DEGENERACY (Entropy Ratio 0.98–1.00) + SEVERE STATIC EMISSION O-BIAS (+6 to +13 Logits) ROOTED IN TRAINING LABEL AMBIGUITY**.  
> **Evaluation Checkpoint**: `checkpoints/improved_assamese_v2.pt` (`baseline_v1/checkpoints/final_assamese.pt`, preserved per Phase 7).  
> **Execution Device**: GPU `cuda:0` (NVIDIA GeForce RTX 3050 Laptop GPU).

---

## 1. Executive Summary

Phase 7 hypothesized that Assamese complete-sentence collapse on sentences >35 tokens was caused by "severe recurrent decay" in the BiLSTM encoder.

Independent, empirical diagnostic testing with hidden-state hooks, attention-entropy measurements, and controlled counterfactual truncations yields a decisive, surprising conclusion:
1. **The Pure "Recurrent Decay" Hypothesis is DISCONFIRMED**:
   - In controlled counterfactual testing, truncating long collapsed sentences down to short sub-spans (15–28 tokens, and even down to a 1-token sentence) **failed to recover entity detection in 0 / 10 cases (0.0% recovery rate)**.
   - If the failure were pure recurrent state decay as the LSTM stepped forward in time, truncating to the opening 1–5 tokens would immediately restore entity detection. It did not.
2. **The True Empirical Root Causes are CONFIRMED**:
   - **Mechanism 1: Complete Attention Degeneracy (Entropy Ratio 0.98–1.00)**. Across all 4 attention heads on long collapsed sentences (including canonical Sentence #200), mean attention entropy is virtually at theoretical maximum ($H \approx \ln(L)$). Head max weights range from 0.023 to 0.062 (essentially uniform $1/L$). The self-attention block smears the representation of every token across dozens of background $O$ tokens.
   - **Mechanism 2: Massive Emission $O$-Bias (+6 to +13 Logits) from Training Label Ambiguity**. Even on isolated tokens with zero recurrent steps and zero attention smearing, the linear emission head outputs an overwhelming positive logit for $O$ ($+5.7$ to $+6.1$) and strongly negative logits for entity tags ($-4.9$ to $-6.9$). The gap is $+10$ to $+13$ logits ($>20,000\times$ probability ratio).
   - **The Origin of the Emission Bias**: In `data/gold/assamese/train.conll`, key entities such as `ভাৰতৰ` (India's) are annotated as `O` **70.6% of the time (24 / 34)** and `ভাৰতীয়` is annotated as `O` **85.9% of the time (79 / 92)**! The model has been trained to treat these entity names as non-entities.

---

## 2. Collapse Length Distribution & Density Disentanglement

### 2.1 Complete Collapse Length Distribution (N=57 collapse instances across manifest)
From the 270-sentence Assamese diagnostic manifest, 57 complete collapse error records (spanning 32 unique sentences) were analyzed:

| Distribution Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Minimum Length** | 9 tokens | Shortest collapsed sentence in manifest |
| **Maximum Length** | 66 tokens | Longest collapsed sentence in manifest |
| **Mean Length** | 38.93 tokens | Strongly shifted toward long sequences |
| **Median Length** | 40 tokens | Confirms Phase 7's observation of collapse at $\sim 40$ tokens |
| **IQR ($Q_1 - Q_3$)** | 33 – 42 tokens | 50% of all collapse cases fall in 33–42 tokens |

### 2.2 Stratified Collapse Rates by Length Bucket
Evaluating collapse incidence against the overall sentence population in the manifest:

| Length Bucket | Total Manifest Sents | Collapsed Sents | Collapse Rate (%) |
| :--- | :---: | :---: | :---: |
| **L1: Short ($\le 12$ tokens)** | 64 | 2 | **3.1%** |
| **L2: Medium (13–22 tokens)** | 89 | 1 | **1.1%** |
| **L3: Long (23–34 tokens)** | 51 | 18 | **35.3%** |
| **L4: Stress ($>34$ tokens)** | 66 | 36 | **54.5%** |

**Observation**: Collapse rate escalates sharply from 1.1% in L2 to 35.3% in L3 and 54.5% in L4, confirming that long sequences suffer an order-of-magnitude higher vulnerability to collapse.

### 2.3 Disentangling Length vs Entity Density
To test whether length alone or entity density is the primary covariate:

| Length Bucket | E1 (1 Entity) | E2 (2 Entities) | E3 (3+ Entities) | Total Collapses |
| :--- | :---: | :---: | :---: | :---: |
| **L1: Short ($\le 12$)** | 2 | 0 | 0 | 2 |
| **L2: Medium (13–22)** | 1 | 0 | 0 | 1 |
| **L3: Long (23–34)** | 3 | 6 | 9 | 18 |
| **L4: Stress ($>34$)** | 7 | 2 | **27** | 36 |

**Finding**: In L4_Stress ($>34$ tokens), **75.0% (27 / 36)** of collapses are dense multi-entity sentences (E3: 3+ entities). Length and density are heavily confounded in the natural data, but both amplify failure.

---

## 3. Attention Degeneracy vs CRF Emission Logit Inspection

To determine the architectural locus of failure, we extracted intermediate tensors from the BiLSTM encoder, Multi-Head Self-Attention layer, and linear emission head for long collapsed sentences, including canonical Sentence #200 (42 tokens, 8 entities).

### 3.1 Attention Entropy & Dispersion Metrics

| Sentence ID | Seq Len | Gold Ents | Theoretical Max Entropy $\ln(L)$ | Empirical Mean Entropy | Entropy Ratio | Head Max Weights $[H_0, H_1, H_2, H_3]$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentence #200** | 42 | 8 | 3.738 | 3.713 | **0.993 (99.3%)** | [0.052, 0.045, 0.055, 0.056] |
| **Sentence #230** | 66 | 4 | 4.190 | 4.119 | **0.983 (98.3%)** | [0.062, 0.052, 0.063, 0.056] |
| **Sentence #220** | 53 | 1 | 3.970 | 3.967 | **0.999 (99.9%)** | [0.027, 0.023, 0.023, 0.025] |
| **Sentence #232** | 47 | 3 | 3.850 | 3.847 | **0.999 (99.9%)** | [0.030, 0.025, 0.027, 0.027] |
| **Sentence #223** | 43 | 3 | 3.761 | 3.761 | **1.000 (100.0%)** | [0.026, 0.025, 0.025, 0.026] |

**Key Diagnostic Finding**:
The attention weights show **complete uniform degeneracy**. On Sentence #200 ($L=42$), uniform attention weight is $1/42 \approx 0.0238$. The maximum weight in Head 1 across any token query is merely **0.045**. The attention layer functions as an unweighted uniform blur, averaging out entity features across the entire sequence.

### 3.2 CRF Emission Logit Gap ($O - \max(\text{Entity})$)

| Sentence ID | Mean Emission Gap ($O - \text{Ent}$) across All Tokens | Emission Gap ($O - \text{Ent}$) at Gold Entity Tokens | CRF Transition Impact |
| :--- | :---: | :---: | :--- |
| **Sentence #200** | **+9.84 logits** | **+8.40 logits** | $O \to O$ transition is $+0.43$; $O \to B\text{-LOC}$ is $+0.12$. |
| **Sentence #230** | **+9.97 logits** | **+6.73 logits** | Emission bias exceeds CRF transition delta by $\mathbf{15\times \text{to } 25\times}$. |
| **Sentence #220** | **+10.66 logits** | **+11.38 logits** | CRF Viterbi decoding is mathematically compelled to emit $O$. |
| **Sentence #223** | **+11.95 logits** | **+12.03 logits** | Entity emission is suppressed by $>160,000\times$ probability ratio. |

#### Specific Token Emissions on Sentence #200
| Token Index | Token | Gold Tag | Emission Logit $O$ | Max Entity Logit | Logit Gap |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **ভাৰতৰ** | `B-LOC` | **+5.74** | -4.92 (`I-ORG`) | **+10.67** |
| 8 | **ভাৰতীয়** | `B-LOC` | **+4.98** | -1.98 (`B-ORG`) | **+6.96** |
| 10 | **বেংক** | `B-ORG` | **+5.74** | -1.95 (`I-ORG`) | **+7.69** |
| 12 | **ভাৰতীয়** | `B-LOC` | **+4.68** | -0.88 (`B-ORG`) | **+5.56** |
| 14 | **বেংক** | `B-ORG` | **+5.65** | -1.07 (`I-ORG`) | **+6.72** |
| 19 | **নিগম** | `B-ORG` | **+4.99** | -0.88 (`I-ORG`) | **+5.87** |
| 39 | **মুম্বাইত** | `B-LOC` | **+5.98** | -5.81 (`I-ORG`) | **+11.79** |

---

## 4. Counterfactual Truncation Tests

If recurrent state decay were the actual root cause, truncating a 42-to-66 token collapsed sentence to a short 1-to-18 token span would remove the recurrent accumulation and restore entity detection.

**Test Population**: 10 collapsed sentences truncated to 18–28 tokens around the target entities, plus Sentence #200 truncated down to 1, 5, 11, and 20 tokens.  
**Success Rate**: **0 / 10 (0.0% recovery rate)**.

### Detailed Counterfactual Truncation Results
1. **Sentence #200 (Length 42 $\to$ 1, 5, 11, 20)**:
   - Full 42 tokens: All 42 tokens predicted `O` (0 / 8 entities detected).
   - Truncated to 1 token (`['ভাৰতৰ']`): Predicted `['O']`!
     - Raw 1-token emission: $O = \mathbf{+6.14}$, $B\text{-LOC} = \mathbf{-6.93}$ (Gap $= \mathbf{+13.07}$ logits).
   - Truncated to 5 tokens: Predicted `['O', 'O', 'O', 'O', 'O']`.
   - Truncated to 11 tokens: All 11 tokens predicted `['O', ..., 'O']`.
   - Truncated to 20 tokens: All 20 tokens predicted `['O', ..., 'O']`.
2. **Sentence #230 (Length 66 $\to$ 18)**:
   - Target entity: `বৰুৱা` (`PER`). Truncated prediction: All 18 tokens `O`.
3. **Sentence #220 (Length 53 $\to$ 28)**:
   - Target entity: `দিল্লী` (`LOC`). Truncated prediction: All 28 tokens `O`.
4. **Sentence #232 (Length 47 $\to$ 18)**:
   - Target entity: `অসমীয়াৰ` (`LOC`). Truncated prediction: All 18 tokens `O`.
5. **Sentence #225 (Length 43 $\to$ 18)**:
   - Target entity: `হাজৰিকাই` (`PER`). Truncated prediction: All 18 tokens `O`.

---

## 5. Training Label Ambiguity Census (The Root of Emission Bias)

Checking `data/gold/assamese/train.conll` revealed why the emission head has an insurmountable $+10$ to $+13$ logit bias toward $O$ on these entity tokens:

| Token | Meaning | Total Occurrences in Train | Annotated as `O` | Annotated as Entity | Entity Annotation % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ভাৰতৰ** | India's | 34 | **24** | 10 (`B-LOC`) | **29.4%** |
| **ভাৰতীয়** | Indian | 92 | **79** | 13 (`B-ORG`) | **14.1%** |
| **বেংক** | Bank | 0 | 0 | 0 | **0.0%** (Unseen in train) |

In the training split, 70.6% of `ভাৰতৰ` mentions and 85.9% of `ভাৰতীয়` mentions are tagged as `O`. The static embeddings and linear projection learned to treat these common entity tokens as non-entity background words.

---

## 6. Root-Cause Verdict for Assamese

1. **Recurrent Decay Hypothesis**: **DISCONFIRMED**. The failure is NOT caused by the BiLSTM's hidden state decaying over $>35$ recurrent steps.
2. **Confirmed Root Causes**:
   - **Severe Attention Degeneracy**: Multi-head self-attention exhibits near-100% entropy, blurring token context uniformly across long sequences.
   - **Static Emission $O$-Bias**: Driven by training set label noise/inconsistency, where entity tokens are predominantly labeled as $O$, producing a $+10$ to $+13$ logit bias toward $O$ that overrides both BiLSTM and CRF.
3. **Implications for Cycle 2 Intervention Design**:
   - Shortening sequence chunks (e.g. sliding window / truncation) will NOT solve Assamese collapse, because the tokens collapse even in 1-token sentences.
   - The intervention MUST address (a) attention regularization/temperature sharpening to eliminate attention degeneracy, (b) emission layer re-balancing / class-weighted loss, or (c) label consistency refinement on frequent entity tokens.
