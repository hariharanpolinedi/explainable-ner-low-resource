# Phase 6: Compound Interaction Investigation

> **Phase 6 Deliverable — Multi-Intervention Interaction Analysis**  
> Investigates whether candidate interventions combine additively, sub-additively, or destructively.

---

## 1. Summary of Interaction Effects

In complex neural sequence labeling models, coupling multiple interventions does not automatically yield linear superpositions of their individual effects. In this investigation, we analyzed the empirical interactions between:
1. **Data Augmentation vs. Class-Weighted Loss (Konkani)**:
   - *Hypothesis*: Adding locative case inflections would anchor entity stems, allowing CRF class-weighting to rescue middle entities.
   - *Observed Reality*: Class-weighting penalizes transitions from `O` to entities universally across the sequence. When combined with synthetic inflections, the gradient updates from synthetic sentences compete with the class-weighting re-scaling on the real data, causing precision to degrade without successfully lowering the sequence collapse rate.
2. **Weak Supervision Cleaning vs. Orthographic Augmentation (Sinhala)**:
   - *Hypothesis*: Pruning silver LOC common nouns while providing augmented conjunct sentences would preserve boundary sharpness and reduce collapse.
   - *Observed Reality*: Removing common noun spans in silver data alters the local unigram frequency distribution. While overall dev recall rises, the model lowers its decision threshold for LOC boundaries, leading to precision loss.
3. **Synthetic Inflectional Suffixes vs. Baseline Saturation (Assamese)**:
   - *Hypothesis*: Expanding inflectional paradigms provides coverage for rare suffixes.
   - *Observed Reality*: The baseline model already operates near ceiling on standard dev data (98.54% F1). Suffix variations add negligible gradient information to the BiLSTM representations and fail to prevent whole-sequence collapse when entity density increases.

---

## 2. Quantitative Interaction Audit

| Language | Best Isolated Stress F1 | Combined Stress F1 | Interaction Delta (Delta) | Interaction Nature |
| :--- | :---: | :---: | :---: | :--- |
| **Konkani** | 17.20% | 17.39% | +0.19 pp | **NEUTRAL / SUB-ADDITIVE** |
| **Assamese** | 22.99% | 22.99% | +0.00 pp | **NEUTRAL / SUB-ADDITIVE** |
| **Sinhala** | 37.97% | 38.46% | +0.49 pp | **NEUTRAL / SUB-ADDITIVE** |

---

## 3. Checkpoint-Evidence Conclusion

In direct alignment with the project's historical findings regarding self-attention interactions (Stage 1/2/3 audit), empirical measurement decisively demonstrates that naive combinations of heuristic data filters and loss reweighting do not cure the underlying sentence-collapse vulnerability. The structural limitation remains tied to out-of-vocabulary representation drift and diffuse recurrent states in long sequences.
