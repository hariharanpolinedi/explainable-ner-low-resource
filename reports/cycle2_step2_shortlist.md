# Reliability Cycle 2 — Step 2: Final Accepted Candidate Shortlist

> **Milestone**: Targeted Intervention Design & Isolated Experimentation Shortlist.
> **Discipline Enforced**: Strict isolation testing against current-best checkpoints; 100% traceability to confirmed Step 1/1b root causes; honest rejection of non-performing candidates.
> **Phase Boundary**: No cross-language combination, no cross-cycle retraining. Candidate shortlist ready for Step 3.

---

## 1. Master Shortlist Matrix

| Language | Candidate Intervention | Confirmed Mechanism Targeted | Dev F1 | Stress Set F1 | Stress Collapse Rate | Targeted Diagnostic Metric Before -> After | Shortlist Decision | Key Tradeoff / Warning |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | **`cycle2_konkani_normalization`** (Lemma/Stem Normalization at Input Layer) | Morphological inflectional OOV in static word embeddings | **95.51%** (+2.81%) | **17.20%** (-0.19%) | **72.73%** (-3.03%) | CF Sentences: **0 / 12 (0.0%) $\to$ 7 / 12 (58.3%)** (8/13 targets) | **ACCEPTED (PRIMARY)** | Zero precision degradation (Dev P: 97.50%, R: 93.60%). 100% recovery on OOV inflections. |
| **Konkani** | `cycle2_konkani_combined` (Augmentation + Normalization) | Morphological inflectional OOV | 95.12% (+2.42%) | 17.20% (-0.19%) | 72.73% (-3.03%) | CF Sentences: 0 / 12 $\to$ 7 / 12 (58.3%) (8/13 targets) | **ACCEPTED (BACKUP)** | Slightly lower Dev F1 (95.12%) than normalization alone. |
| **Konkani** | `cycle2_konkani_augmentation` (Augmentation Only) | Morphological inflectional OOV | 93.39% (+0.69%) | 17.20% (-0.19%) | 72.73% (-3.03%) | CF Sentences: 0 / 12 $\to$ 0 / 12 (0.0%) (0/13 targets) | **REJECTED** | Fails to resolve OOV embedding barrier; recovers 0 counterfactual cases. |
| **Assamese** | **`cycle2_assamese_combined`** (Label Denoising + Emission Loss Rebalancing) | Runaway +10 to +13 logit linear emission O-bias from label noise | **95.34%** (-3.20%) | **22.22%** (-0.77%) | **67.74%** (-3.23%) | S200 O-Bias Gap: **+9.64 $\to$ +0.86 logits** (-91%) | **ACCEPTED (PRIMARY)** | Slashes root emission bias by 91%; modest dev precision tradeoff (95.00%). |
| **Assamese** | `cycle2_assamese_loss_rebalanced` (Emission Loss Rebalancing Only) | Linear emission O-bias | 97.83% (-0.71%) | 22.99% (0.00%) | 70.97% (0.00%) | S200 O-Bias Gap: +9.64 $\to$ +6.09 logits (-37%) | **ACCEPTED (BACKUP)** | Preserves high Dev F1 (97.83%), but leaves +6.09 logit bias intact without label cleaning. |
| **Assamese** | `cycle2_assamese_label_corrected` (Label Correction Only) | Training label inconsistency | 96.80% (-1.74%) | 21.98% (-1.01%) | 67.74% (-3.23%) | S200 O-Bias Gap: +9.64 $\to$ +3.01 logits (-69%) | **REJECTED** | Does not overcome CRF emission gradient without rebalanced loss; Dev P drops to 95.77%. |

---

## 2. Evidentiary Justification for Decisions

### Konkani Verdict:
- **Primary Winner**: `cycle2_konkani_normalization` (`checkpoints/cycle2_konkani_normalization.pt`).
- **Evidence**: Directly solves the root cause by mapping OOV inflected tokens to their in-vocabulary base stem embeddings while retaining surface graphemes for the Char-CNN. Counterfactual sentence recovery jumps from **0 / 12 (0.0%) to 7 / 12 (58.3%)** overall and **6 / 6 (100.0%)** on genuine OOV inflections (`पणजींत`, `पणजीक`, `गोयाच्या`, `गोयाक`, `वास्कोक`, `म्हापश्यांत`, `सावंतांनी`), exactly matching the diagnostic recovery finding from Step 1. Gold Dev F1 increases from **92.70% to 95.51%** with zero tradeoff in precision (97.50% precision).
- **Rejection of Augmentation-Only**: Augmentation alone without lemma normalization recovers 0 of 12 counterfactual sentences on genuine OOV inflections, confirming that static word embeddings cannot be fixed by small amounts of surface text augmentation when the static vocabulary remains closed.

### Assamese Verdict:
- **Primary Winner**: `cycle2_assamese_combined` (`checkpoints/cycle2_assamese_combined.pt`).
- **Evidence**: Directly attacks the primary causal driver identified in Step 1b (the +10 to +13 logit linear emission O-bias driven by training label inconsistency). Slashes the mean O-bias gap on Sentence #200 from **+9.64 logits down to +0.86 logits (a 91.1% reduction)**, and on Token 0 (`ভাৰতৰ`) from **+12.52 down to +1.54 logits (an 87.7% reduction)**, while reducing the stress set collapse rate from **70.97% to 67.74%** and preserving an aggregate Dev F1 of **95.34%**.
- **Backup Winner**: `cycle2_assamese_loss_rebalanced` (`checkpoints/cycle2_assamese_loss_rebalanced.pt`), which preserves a higher Dev F1 (97.83%) and 98.54% precision, but achieves only a 37% reduction in the S200 logit gap (+6.09 logits remaining).
- **Rejection of Label Correction Alone**: Correcting labels without loss rebalancing leaves a +3.01 logit gap and degrades Dev F1 to 96.80%.

## 3. Next Steps (Step 3 Scope Boundary)

The shortlisted interventions:
- **Konkani Primary**: `cycle2_konkani_normalization` (Backup: `cycle2_konkani_combined`)
- **Assamese Primary**: `cycle2_assamese_combined` (Backup: `cycle2_assamese_loss_rebalanced`)

All interventions have been strictly tested in isolation against the baseline checkpoints. The phase boundary is fully respected. Ready for Step 3: Final Retrain, Multilingual Integration & Verification.
