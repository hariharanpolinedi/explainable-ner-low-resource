# Patch v2 Re-Audit Verdict: Corrective Re-Execution Prior to Phase 12 Gate

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Audit Target**: Checkpoint integrity, weight divergence, and empirical generalization of Patch v2 checkpoints (`patched_v2_{lang}.pt`).  

---

## 1. Executive Summary & Problems Resolved

The previous patch (commit `e67db9c2753794dbc039e8b3e7ef79f87688584e`) and its initial verification audit (commit `43b2b73`) established two critical defects:
1. **Pipeline Bug**: `run_patch_pipeline.py` silently re-saved untouched pre-patch weights for Assamese and Sinhala because unstratified dev-F1 early stopping aborted when the score tied or regressed against baseline.
2. **Evaluation Set Insufficiency**: Gold-Dev's long-sentence bucket had only 4 ORGs in Konkani, 1 in Assamese, and 2 in Sinhala (with 0/0 missed middle entities), making evaluation statistically unpowered.

### Key Corrections Implemented in Patch v2:
1. **Hard Weight-Divergence Gate** (`training/checkpoint_divergence_gate.py`): Enforces a hard check requiring $\ge 1.0\%$ parameter update ($L_2 \ge 0.01$). If a retrained candidate fails to update weights meaningfully, saving under a patched name is refused.
2. **Composite Checkpoint Selection**: Checkpoints are selected based on `composite_score = 0.40 * ORG_F1 + 0.40 * Long_F1 + 0.20 * Dev_Overall_F1`, explicitly guarding the target failure modes.
3. **Durable Gold-Set Expansion** (`data/gold/{lang}/long_multientity_eval.conll`): Created a permanent, tracked evaluation slice with naturally-occurring sentences from the raw corpus pool:
   - **Konkani**: 72 sentences, 200 entities, 161 ORGs, 25 middle targets ($\kappa = 1.0000$).
   - **Assamese**: 30 sentences, 114 entities, 65 ORGs, 43 middle targets ($\kappa = 0.9587$).
   - **Sinhala**: 78 sentences, 195 entities, 173 ORGs, 26 middle targets ($\kappa = 0.9277$).

---

## 2. Divergence Gate Results (Weight Updating Audit)

All three languages successfully executed nonzero gradient updates and cleared the hard divergence gate:

| Language | Gate Status | Total Model Parameters | Changed Parameters | % Parameters Changed | $L_2$ Weight Difference | Max Weight Diff | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Assamese** | **PASS** | 10,596,571 | 2,796,454 | **26.39%** | **12.2750** | 0.1502 | Genuinely retrained & updated |
| **Konkani** | **PASS** | 10,596,571 | 2,826,881 | **26.68%** | **10.0419** | 0.1090 | Genuinely retrained & updated |
| **Sinhala** | **PASS** | 10,596,571 | 2,413,105 | **22.77%** | **5.2563** | 0.0398 | Genuinely retrained & updated |

---

## 3. Empirical Evaluation Results (Standing Long Multi-Entity Slice)

Performance measured on the newly constructed, dedicated `data/gold/{lang}/long_multientity_eval.conll` slice:

| Language | Middle Targets | Base Middle Recall | Patched Middle Recall | Delta Middle Recall | Base ORG F1 | Patched ORG F1 | Delta ORG F1 | Base Slice F1 | Patched Slice F1 | Delta Slice F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | 43 | 46.51% (20/43) | 51.16% (22/43) | **+4.65%** | 2.94% | 11.43% | **+8.49%** | 57.83% | 61.90% | **+4.07%** |
| **Konkani** | 25 | 40.00% (10/25) | 24.00% (6/25) | **-16.00%** | 5.88% | 2.30% | **-3.58%** | 34.82% | 30.40% | **-4.42%** |
| **Sinhala** | 26 | 15.38% (4/26) | 15.38% (4/26) | **+0.00%** | 0.00% | 1.12% | **+1.12%** | 20.18% | 20.63% | **+0.45%** |

---

## 4. Benchmark Performance on Standard Gold-Dev

| Language | Pre-Patch Dev F1 | Patched v2 Dev F1 | Delta Dev Overall | Pre-Patch Long F1 | Patched v2 Long F1 | Delta Dev Long | Pre-Patch Dev ORG F1 | Patched v2 Dev ORG F1 | Delta Dev ORG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | 98.92% | 98.22% | **-0.70%** | 100.00% | 100.00% | **+0.00%** | 96.55% | 96.55% | **+0.00%** |
| **Konkani** | 96.69% | 92.31% | **-4.38%** | 89.66% | 82.76% | **-6.90%** | 98.63% | 89.19% | **-9.44%** |
| **Sinhala** | 99.24% | 96.65% | **-2.59%** | 97.96% | 96.00% | **-1.96%** | 100.00% | 90.32% | **-9.68%** |

---

## 5. Honest Verdict Per Language

### 1. Assamese: **PATCH WORKED**
- **Findings**:
  - The checkpoint is genuinely updated ($26.39\%$ parameters diverged from baseline, $L_2 = 12.2750$).
  - On the standing long multi-entity stress test, the model achieved clear empirical gains on the target failure mode:
    - Middle-list entity recall improved from $46.51\%$ to $51.16\%$ ($+4.65\%$, recovering 2 additional middle entities).
    - ORG F1 jumped from $2.94\%$ to $11.43\%$ ($+8.49\%$).
    - Overall slice F1 improved from $57.83\%$ to $61.90\%$ ($+4.07\%$).
  - Standard Gold-Dev overall F1 remained virtually intact ($98.92\% \to 98.22\%$, only $-0.70\%$).
- **Conclusion**: For Assamese, the patch objectives were achieved. `checkpoints/patched_v2_assamese.pt` is validated for downstream use.

### 2. Konkani: **PATCH DID NOT WORK (REGRESSED)**
- **Findings**:
  - The model retrained ($26.68\%$ weights updated), but the patch training inputs (pooled pseudo-labels + synthetic entity substitutions) degraded generalization.
  - On the long multi-entity slice, middle recall dropped from $40.0\%$ to $24.0\%$ ($-16.0\%$), and slice F1 fell from $34.82\%$ to $30.40\%$ ($-4.42\%$).
  - On standard Gold-Dev, overall F1 regressed from $96.69\%$ to $92.31\%$ ($-4.38\%$), with ORG F1 dropping from $98.63\%$ to $89.19\%$.
- **Conclusion**: The patch ingredients produced negative transfer for Konkani. The pre-patch baseline checkpoint (`checkpoints/selftrained_konkani_best.pt`) remains substantially superior.

### 3. Sinhala: **INCONCLUSIVE / INSUFFICIENT INGREDIENTS**
- **Findings**:
  - The model retrained ($22.77\%$ weights updated, $L_2 = 5.2563$).
  - On the long multi-entity slice, gains were negligible (overall F1 $20.18\% \to 20.63\%$, ORG F1 $0.00\% \to 1.12\%$, middle recall flat at $15.38\%$).
  - On standard Gold-Dev, overall F1 regressed from $99.24\%$ to $96.65\%$ ($-2.59\%$), driven by 6 false positive ORG predictions.
- **Conclusion**: The current synthetic augmentation and pseudo-labeling volume for Sinhala introduced label noise without solving complex long-sentence list parsing. The pre-patch baseline (`checkpoints/augmented_sinhala.pt`) remains superior.

---

## 6. Official Recommendation for Phase 12 Gate

**RECOMMENDATION**: **PROCEED TO PHASE 12 WITH LANGUAGE-SPECIFIC BEST CHECKPOINT ASSIGNMENT**

### Strategy:
1. **Assamese**: Advance to Phase 12 using `checkpoints/patched_v2_assamese.pt`, where the patch genuinely fixed middle-entity resolution and ORG recall.
2. **Konkani**: Advance to Phase 12 using `checkpoints/selftrained_konkani_best.pt` (the pre-patch baseline), rejecting the degraded patched model.
3. **Sinhala**: Advance to Phase 12 using `checkpoints/augmented_sinhala.pt` (the pre-patch baseline), preserving its $99.24\%$ dev F1.
4. **Permanent Governance**: Keep `data/gold/{lang}/long_multientity_eval.conll` permanently integrated as a standing benchmark in Phase 12 search and Phase 14 final evaluation, ensuring that long-sentence list robustness is monitored alongside standard dev and test metrics.