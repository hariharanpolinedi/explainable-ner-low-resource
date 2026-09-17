# Current Best Checkpoint Recommendation & Starting Point for Remediation

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Objective**: Establish the empirical ground truth for checkpoint selection based strictly on honest measurements on locked Gold-Dev.

---

## 1. Executive Summary: Current Best Checkpoint Per Language

Based strictly on evaluation against the untouched, immutable `data/gold/{lang}/dev.conll`, the pre-patch baselines are the genuine best models across all three languages:

| Language | Recommended Checkpoint | Origin Phase | Gold-Dev Overall F1 | PER F1 | LOC F1 | ORG F1 | MISC F1 | Token Acc |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `checkpoints/selftrained_konkani_best.pt` | Phase 10 Baseline | **96.69%** | 91.89% | 93.10% | 98.63% | 100.00% | 98.71% |
| **Assamese** | `checkpoints/augmented_assamese.pt` | Phase 11 Baseline | **98.92%** | 100.00% | 98.73% | 96.55% | 100.00% | 99.41% |
| **Sinhala** | `checkpoints/augmented_sinhala.pt` | Phase 11 Baseline | **99.24%** | 100.00% | 97.50% | 100.00% | 100.00% | 99.68% |

---

## 2. Why the Patched Checkpoints Must Be Rejected

### 1. Konkani (`patched_v2_konkani.pt`):
- Overall Gold-Dev F1 dropped from **96.69% to 92.31%** ($-4.38\%$).
- ORG F1 plummeted from **98.63% to 89.19%** ($-9.44\%$).
- On manual testing, it failed identically to baseline (2/5 detected, missing `पणजींत`, `गोयाच्या`, and middle entity `श्रीपाद नाईक`).

### 2. Assamese (`patched_v2_assamese.pt`):
- Overall Gold-Dev F1 dropped from **98.92% to 98.22%** ($-0.70\%$).
- PER F1 dropped from **100.00% to 94.74%** ($-5.26\%$).
- Its apparent $+8.49\%$ gain in Patch v2 was measured on a contaminated slice where 98.5% of ORGs were injected by `DedicatedORGRuleTagger`. On real gold data, it produced zero ORG gain and degraded person entity detection.

### 3. Sinhala (`patched_v2_sinhala.pt`):
- Overall Gold-Dev F1 dropped from **99.24% to 96.65%** ($-2.59\%$).
- ORG F1 dropped from **100.00% to 90.32%** ($-9.68\%$) due to false-positive ORG predictions.

---

## 3. Honest Weakness Diagnosis: Gaps Below the >90% Floor

While unstratified aggregate scores exceed 90% across all categories, targeted stratification reveals severe localized failure modes:

1. **Konkani Location Inflection / OOV Collapse**:
   - In-Vocab LOC F1: **100.00%** (Recall: 100.0%).
   - OOV / Inflected LOC F1: **50.00%** (Recall: **33.3%**).
   - *Failure Mechanism*: When place names take locative (`-ंत`) or genitive (`-चो/-च्या`) case inflections, both gazetteers and static suffix rules fail. The model misses 2 out of 3 inflected place names in real sentences.
2. **Long Multi-Entity List Middle Recall**:
   - In 30+ token sentences with comma/conjunction lists, middle entities are systematically missed across all three languages:
     - Konkani middle recall: **40.0%** (missing 60% of middle entities).
     - Assamese middle recall: **46.5%** (missing 53.5% of middle entities).
     - Sinhala middle recall: **15.4%** (missing 84.6% of middle entities).

---

## 4. Starting Point for Remediation & Phase 12

1. **Baseline Anchoring**:
   All remediation and hyperparameter tuning in Phase 12 should branch from:
   - Konkani: `checkpoints/selftrained_konkani_best.pt`
   - Assamese: `checkpoints/augmented_assamese.pt`
   - Sinhala: `checkpoints/augmented_sinhala.pt`
2. **Remediation Priorities**:
   - **Priority 1 (LOC Case Morphology)**: Add rule-based and synthetic augmentation for case inflections (`-ंत`, `-च्या`, `-चो`) on location names to fix the 33.3% OOV recall bottleneck.
   - **Priority 2 (List-Aware Attention / Augmentation)**: Address middle-entity list parsing using list-level syntactic permutations rather than naive weak supervision heuristics.
