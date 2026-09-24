# Reliability Cycle 2 — Step 2b: Assamese Intervention Strengthening Sweep Report

> **Task**: Systematic strengthening sweep and data reinforcement for Assamese linear emission $O$-bias intervention.  
> **Starting Baseline**: `checkpoints/improved_assamese_v2.pt` (Phase 7 baseline, 0 diffs from baseline_v1).  
> **Evaluated Benchmarks**: Gold Dev (`dev.conll`), Stress Multi-Entity Set (`long_multientity_eval_v2.conll`), Canonical Target (`Sentence #200`).  
> **Primary Question**: Does pushing the validated intervention direction further (stronger loss weight $2\times, 4\times, 8\times$, targeted genuine data reinforcement) flip Sentence #200 detection ($0/8 \to \ge 1/8$), or does a data/loss-only approach encounter a hard empirical ceiling?

---

## 1. Executive Summary & Full Sweep Comparison Table

| Candidate / Level | Training Data | Loss Weighting | Dev F1 | Dev Prec | Dev Rec | Stress F1 | Stress Collapse Rate | S200 Entities Detected | S200 Hits | S200 Mean $O$-Bias Gap | S200 Token 0 Gap | Status / Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`baseline`** | `train.conll` | None ($1.0\times$) | **98.54%** | **100.00%** | 97.12% | **22.99%** | **70.97% (22/31)** | 0 / 8 | 0 / 8 | **+9.43** | **+10.63** | Unpatched Baseline |
| **`step2_combined`** | `train_c2_corrected` | Baseline ($1.0\times$) | **95.34%** | **95.00%** | 95.68% | **22.22%** | **67.74% (21/31)** | 0 / 8 | 0 / 8 | **+0.85** | **+0.75** | Step 2 Reference |
| **`weight_2x`** | `train_c2_corrected` | Sweep $2.0\times$ | **95.44%** | 93.15% | 97.84% | **21.74%** | **67.74% (21/31)** | 2 / 8 | 0 / 8 | **+2.05** | **+2.74** | Evaluated (Best Dev F1 Epoch) |
| **`weight_4x`** | `train_c2_corrected` | Sweep $4.0\times$ | **96.80%** | 95.77% | 97.84% | **22.22%** | **67.74% (21/31)** | 0 / 8 | 0 / 8 | **+2.67** | **+3.38** | Evaluated (Best Dev F1 Epoch) |
| **`weight_8x`** | `train_c2_corrected` | Sweep $8.0\times$ | **97.49%** | 97.14% | 97.84% | **22.22%** | **67.74% (21/31)** | 0 / 8 | 0 / 8 | **+3.10** | **+3.83** | Evaluated (Best Dev F1 Epoch) |
| **`reinforced_data`** | `train_c2_corrected_v2` | Baseline ($1.0\times$) | **93.57%** | 92.91% | 94.24% | **22.47%** | **67.74% (21/31)** | 2 / 8 | 0 / 8 | **+0.60** | **+0.61** | **Lowest Mean Gap (+0.60)** |
| **`reinforced_2x`** | `train_c2_corrected_v2` | Sweep $2.0\times$ | **95.10%** | 92.52% | 97.84% | **21.51%** | **67.74% (21/31)** | 2 / 8 | 0 / 8 | **+2.00** | **+2.71** | Evaluated (Best Dev F1 Epoch) |

---

## 2. Decoded Outcome Audit on Target Sentence #200

Sentence #200 (42 tokens, 8 gold entity tokens) is the canonical collapse case. Gold annotations:
- `ভাৰতৰ` [0:1] $\to$ `B-LOC`
- `ভাৰতীয়` [8:9] $\to$ `B-LOC`, `বেংক` [10:11] $\to$ `B-ORG`
- `ভাৰতীয়` [12:13] $\to$ `B-LOC`, `বেংক` [14:15] $\to$ `B-ORG`
- `ভাৰতীয়` [16:17] $\to$ `B-LOC`, `নিগম` [19:20] $\to$ `B-ORG`
- `মুম্বাইত` [39:40] $\to$ `B-LOC`

### Per-Candidate Decoded Tag Sequence & Span Output:
1. **`baseline`**:
   - Decoded sequence: 42 tokens all predicted as `O` (100% false-negative collapse).
   - Mean $O$-bias gap: **$+9.43$ logits**; Token 0 gap: **$+10.63$ logits**.
2. **`step2_combined`**:
   - Decoded sequence: 42 tokens all predicted as `O`.
   - Mean $O$-bias gap: **$+0.85$ logits**; Token 0 gap: **$+0.75$ logits**.
   - Net reduction in bias: $-91.0\%$. Residual positive gap remains above zero.
3. **`weight_2x`** (Saved checkpoint, Epoch 1):
   - Decoded sequence: Predicted `[('ORG', 12, 14)]` (`ভাৰতীয়` [12] $\to$ `B-ORG`, `ষ্টেট` [13] $\to$ `I-ORG`).
   - Hits against gold: **0 / 8** (Gold tagged `ভাৰতীয়` as `B-LOC` and `ষ্টেট` as `O`).
4. **`weight_4x` & `weight_8x`** (Saved checkpoints):
   - Decoded sequence: 42 tokens all predicted as `O` (0/8 detected).
   - Early-stopping on Dev F1 selects Epoch 1 where weight scaling has not yet forced entity prediction.
5. **`reinforced_data`** (`cycle2_assamese_reinforced_data.pt`):
   - Mean $O$-bias gap: **$+0.60$ logits** (lowest across all candidates).
   - Token 0 gap: **$+0.61$ logits** ($O$: $-0.47$, $B\text{-LOC}$: $-1.09$).
   - Predicted span: `[('ORG', 12, 14)]` (`ভাৰতীয়/B-ORG ষ্টেট/I-ORG`). Hits: 0 / 8.
   - Tokens [8:9] (`ভাৰতীয়`): $O$-bias gap flipped negative ($-0.26$ logits), but decoded to $O$ due to transition matrix penalty ($O \to O$ transition bonus $+0.43$).

---

## 3. Precision-Tradeoff Boundary & Epoch Trajectory Analysis

Tracking training across all 5 epochs under scaled loss weights reveals the precise mathematical and empirical boundary where precision degrades:

### Epoch-by-Epoch Dynamics for `weight_2x`:
* **Epoch 1**: Dev F1 = **95.44%**, Precision = **93.15%**, Recall = 97.84% | Stress Collapse = 67.7% | S200 Gap = +2.05 | S200 Ents: 2/8 (Hits: 0)
* **Epoch 2**: Dev F1 = **92.41%**, Precision = **88.74%**, Recall = 96.40% | Stress Collapse = 61.3% | S200 Gap = +0.76 | S200 Ents: 2/8 (Hits: 0)
* **Epoch 3**: Dev F1 = **90.39%**, Precision = **89.44%**, Recall = 91.37% | Stress Collapse = 61.3% | S200 Gap = +0.45 | S200 Ents: 0/8 (Hits: 0)
* **Epoch 4**: Dev F1 = **83.06%**, Precision = **77.16%**, Recall = 89.93% | Stress Collapse = 45.2% | S200 Gap = +0.20 | S200 Ents: 10/8 (Hits: 0)
* **Epoch 5**: Dev F1 = **65.10%**, Precision = **51.02%**, Recall = 89.93% | Stress Collapse = 3.2% | S200 Gap = +0.04 | S200 Ents: 42/8 (Hits: 0)

### Epoch-by-Epoch Dynamics for `weight_8x`:
* **Epoch 1**: Dev F1 = **97.49%**, Precision = **97.14%**, Recall = 97.84% | Stress Collapse = 67.7% | S200 Gap = +3.10 | S200 Ents: 0/8 (Hits: 0)
* **Epoch 2**: Dev F1 = **96.75%**, Precision = **97.10%**, Recall = 96.40% | Stress Collapse = 67.7% | S200 Gap = +1.51 | S200 Ents: 0/8 (Hits: 0)
* **Epoch 3**: Dev F1 = **96.35%**, Precision = **97.78%**, Recall = 94.96% | Stress Collapse = 67.7% | S200 Gap = +0.94 | S200 Ents: 0/8 (Hits: 0)
* **Epoch 4**: Dev F1 = **93.43%**, Precision = **94.81%**, Recall = 92.09% | Stress Collapse = 71.0% | S200 Gap = +0.51 | S200 Ents: 0/8 (Hits: 0)
* **Epoch 5**: Dev F1 = **89.86%**, Precision = **90.51%**, Recall = 89.21% | Stress Collapse = 71.0% | S200 Gap = +0.34 | S200 Ents: 0/8 (Hits: 0)

### Critical Empirical Insight:
1. **Unacceptable Precision Collapse**: At $2\times$ weight, allowing training to continue to Epoch 5 drives the mean emission gap down to $+0.04$ logits and Token 0 gap to $-0.16$ logits, but at a **catastrophic precision cost: Dev Precision drops from 100.0% to 51.02%**! The model enters runaway false-positive hallucination, predicting all 42 tokens of Sentence #200 as entity (`ORG`), destroying system usability.
2. **The 90.0% Precision Boundary**: If precision is constrained to $\ge 90.0\%$ (our established threshold of acceptability), the emission gap cannot be driven far enough negative to overcome the CRF transition barrier ($O \to O$ bonus of $+0.43$).
3. **Compound Annotation Asymmetry**: When entity tokens are decoded (`ভাৰতীয় ষ্টেট`), the model predicts the standard real-world compound entity `[ORG: ভাৰতীয় ষ্টেট]`, which conflicts with the benchmark's idiosyncratic inner token segmentation (`ভাৰতীয়/B-LOC ষ্টেট/O বেংক/B-ORG`).

---

## 4. Divergence Gate Verification Audit

Every candidate was verified against `checkpoints/improved_assamese_v2.pt` via `save_patched_checkpoint_with_gate`:

| Candidate Checkpoint | Status | Parameters Changed | Pct Changed (Gate: $\ge 1.0\%$) | L2 Diff (Gate: $\ge 0.01$) | Max Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cycle2_assamese_weight2x.pt` | **PASSED** | 1,378,980 / 10,596,571 | **13.01%** | **1.9909** | 0.0063 |
| `cycle2_assamese_weight4x.pt` | **PASSED** | 1,346,731 / 10,596,571 | **12.71%** | **1.9049** | 0.0063 |
| `cycle2_assamese_weight8x.pt` | **PASSED** | 1,318,698 / 10,596,571 | **12.44%** | **1.8622** | 0.0063 |
| `cycle2_assamese_reinforced_data.pt` | **PASSED** | 1,470,683 / 10,596,571 | **13.88%** | **5.0592** | 0.0249 |
| `cycle2_assamese_reinforced_weight2x.pt` | **PASSED** | 1,380,681 / 10,596,571 | **13.03%** | **1.9822** | 0.0063 |

All runs represent genuine retrained weights and pass the divergence threshold with $>12\%$ parameters modified and $L_2 \ge 1.86$.
