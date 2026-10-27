# Unsanitized Chronological Project History & Audit Ledger

**Paper Section Target**: Limitations / Methodological Audits / Negative Results  
**Auditor**: Senior ML/NLP Research Engineer  
**Scope**: Full lifecycle record of all patch, audit, remediation, and verification cycles conducted across the project  
**Principle**: Raw, unsanitized scientific accounting of what worked, what failed, and which evaluation artifacts were found contaminated and discarded.

---

## 1. Chronological Overview of Engineering & Audit Interventions

```
[Phase 10/11 Patch v1]
        │
        ▼
[Audit Cycle 1: Weight Divergence & Circularity Discovery]
        │ (Unchanged weights in as/si; in-sample silver precision exposed)
        ▼
[Patch v2: Synthetic Rule Augmentation Attempt]
        │
        ▼
[Audit Cycle 2: Selection Bias & Pseudo-Annotator Circularity]
        │ (Contaminated long_multientity_eval.conll discarded; Patch v2 rejected)
        ▼
[Long Multi-Entity Benchmark v2 Rebuild]
        │ (Unbiased length-only sampling, true dual annotation)
        ▼
[Specialized Recovery Cycles: LOC Expansion, Sinhala ZWJ, Inflectional Retraining]
        │ (All speculative patches failed to beat core baselines)
        ▼
[Master 50-Checkpoint Re-Evaluation & Ground-Truth Reversion]
        │ (Reverted all 3 languages to genuine Phase 10/11 baselines)
        ▼
[Phase 14 Single-Run Locked Test Evaluation]
```

---

## 2. Granular Audit Ledger by Intervention Cycle

### 2.1 Cycle 1: Phase 10/11 Patch v1 (ORG Recall & Length Robustness)
- **Claim**: The patch claimed to eliminate ORG under-detection and middle-entity dropping across all three languages (commit `e67db9c2`).
- **Trigger for Audit**:
  1. Reported deltas of exactly $0.00\%$ across every length bucket for Assamese and Sinhala.
  2. Reported "Missed middle entities" was $0/0$ across all evaluation buckets.
  3. Silver-label precision was reported as an implausible $100.0\%$.
- **Audit Findings** (*Source: [`reports/audit_final_verdict.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/audit_final_verdict.md)*):
  - **Weight Divergence**: In `training/run_patch_pipeline.py`, early-stopping logic re-saved the baseline checkpoint if the new epoch did not strictly exceed baseline dev F1. Assamese tied baseline ($98.92\%$) and Sinhala regressed ($96.30\%$ vs $99.24\%$). Both saved untouched baseline weights ($L_2\text{ diff} = 0.0000$). The $0.00\%$ deltas occurred solely because the baseline was evaluated against itself.
  - **Konkani Result**: Konkani was genuinely retrained ($22.69\%$ weights updated), but Gold-Dev long sentences had zero ORG errors initially. On a newly constructed supplementary set, the pre-patch baseline already had $100\%$ middle recall ($4/4$), and post-patch introduced a false positive, regressing F1 from $92.50\%$ to $91.36\%$ ($-1.14\%$).
  - **In-Sample Silver Precision**: The claimed $100.0\%$ silver precision was evaluated on the identical 200 Gold-Dev sentences used to calibrate rule weights. Out-of-sample evaluation on Gold-Train revealed true precision of **$86.90\%$** (Konkani), **$81.25\%$** (Assamese), and **$91.43\%$** (Sinhala).
- **Verdict**: **FAILED / INCONCLUSIVE**. Patch v1 rejected.

---

### 2.2 Cycle 2: Patch v2 & Evaluation Set Contamination
- **Attempt**: Patch v2 generated an expanded evaluation set `data/gold/{lang}/long_multientity_eval.conll` claiming large ORG gains.
- **Audit Findings** (*Source: [`reports/verify_final_verdict.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/verify_final_verdict.md)*):
  - **Severe Selection Bias**: The evaluation set entity distribution was artificially skewed: Konkani was **$80.5\%$ ORG**, Sinhala was **$88.7\%$ ORG**, and Assamese was **$57.0\%$ ORG** (compared to ~20–30% in natural gold splits).
  - **Rule-Tagger Injection**: In `annotation/expand_long_multientity_set.py`, candidates were screened for ORG keywords and lines 120–125 directly injected predictions from `DedicatedORGRuleTagger` into the gold tags. **$98.8\%$** of Konkani ORGs, **$98.5\%$** of Assamese ORGs, and **$99.4\%$** of Sinhala ORGs were synthetically created by the rule tagger.
  - **Circularity in Inter-Annotator Agreement**: The reported perfect agreement ($\kappa = 1.0000$) was an artifact of two automated scripts calling the same function without activating perturbation rules. True agreement against the rule tagger was only $\kappa = 0.1053$.
  - **Performance on Real Gold-Dev**: When evaluated on untouched Gold-Dev, `patched_v2_konkani.pt` suffered severe regression (**$92.31\%$ vs $96.69\%$**, $-4.38$ pp; ORG F1 dropped $-9.4$ pp) and `patched_v2_sinhala.pt` regressed (**$96.65\%$ vs $99.24\%$**, $-2.59$ pp). Assamese gained $+8.49\%$ only on the synthetic ORG test set, but regressed $-0.70\%$ on real gold data.
- **Discarded Artifacts**: **`data/gold/{lang}/long_multientity_eval.conll` (v1) was declared scientifically contaminated and permanently discarded.**
- **Verdict**: **CRITICAL REJECTION**. Patch v2 discarded.

---

### 2.3 Cycle 3: Bias-Free Long Multi-Entity Benchmark v2
- **Intervention** (commit `d6818c5`):
  - Rebuilt `data/gold/{lang}/long_multientity_eval_v2.conll` from the raw corpus pool using **length-only criteria** ($30 \le \text{tokens} \le 80$, delimiter density $\ge 2$).
  - Completely banned `DedicatedORGRuleTagger` and gazetteer keyword pre-filtering.
  - Conducted genuine dual annotation with adjudicated dispute resolution.
- **Outcome** (*Source: [`reports/long_multientity_eval_v2_agreement.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/long_multientity_eval_v2_agreement.md)*):
  - Validated balanced entity distributions matching natural linguistic frequency.
  - Established cryptographically locked benchmark `long_multientity_eval_v2_lockfile.md`.

---

### 2.4 Cycle 4: Location (LOC) Recovery & Diagnostic Retraining
- **Diagnosis** (*Source: [`reports/loc_diagnostic_findings.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/loc_diagnostic_findings.md)*):
  - Investigated why inflected location names (e.g., Konkani locative `पणजींत` [in Panaji] and genitive `गोयाच्या` [of Goa]) failed in manual testing.
  - Gazetteers and rules contained only nominative base lemmas (`पणजी`, `गोवा`), leaving inflected forms as OOV tokens that the Char-CNN could not reliably resolve alone.
- **Retraining Attempt** (*Source: [`reports/loc_patch_retraining_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/loc_patch_retraining_results.md)*):
  - Retrained models with expanded toponymic gazetteers and suffix rules (`loc_patched_{lang}.pt`).
  - **Results**: Konkani Gold-Dev F1 dropped slightly ($96.69\% \to 96.27\%$), Assamese was identical ($98.92\%$), and Sinhala suffered severe LOC regression ($97.50\% \to 90.70\%$, overall F1 $99.24\% \to 96.30\%$).
- **Verdict**: **FAILED**. `loc_patched_{lang}.pt` rejected.

---

### 2.5 Cycle 5: Sinhala Zero-Width Joiner (ZWJ) Orthography Verification
- **Diagnosis** (*Source: [`reports/zwj_hypothesis_verification.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/zwj_hypothesis_verification.md)*):
  - In Sinhala, complex consonant conjuncts require an explicit Unicode Zero-Width Joiner (`\u200D`, ZWJ) between the virama (hal kirima `\u0DCA`) and the following consonant (e.g., `ශ්` + `\u200D` + `රී` = `ශ්‍රී`).
  - When user input or web text strips ZWJ (decomposed form `ශ්රී`), the grapheme cluster tokenizer decomposes the akshara into distinct byte components, causing a 100% entity detection collapse on base named entities (`ශ්‍රී ලංකාව` detected, `ශ්රී ලංකාව` collapsed to `O`).
- **Remediation Attempt** (*Source: [`reports/conjunct_repair_fix_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/conjunct_repair_fix_results.md)*):
  - Retrained with ZWJ perturbation augmentation (`conjunct_fixed_sinhala.pt`).
  - On Gold-Dev, overall F1 regressed from **$99.24\%$ to $98.13\%$** (LOC F1 dropped from $97.5\%$ to $94.0\%$).
- **Verdict**: **FAILED AS RETRAINING; CONFIRMED AS PREPROCESSING REQUIREMENT**. Checkpoint rejected; ZWJ normalization added to input preprocessing pipeline.

---

### 2.6 Cycle 6: Cross-Language Inflectional Augmentation
- **Attempt** (*Source: [`reports/inflectional_retraining_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/inflectional_retraining_results.md)*):
  - Applied cross-language morphological suffix augmentation to train sets to teach the models nominal case markers (`inflected_{lang}.pt`).
- **Results on Locked Gold-Dev**:
  - Konkani: $96.69\% \to 96.69\%$ (identical F1, lower token accuracy $99.59\% \to 99.56\%$).
  - Assamese: $98.92\% \to 98.57\%$ ($-0.35$ pp regression).
  - Sinhala: $98.13\% \to 95.62\%$ ($-2.51$ pp regression; LOC collapsed from $93.98\%$ to $87.64\%$).
- **Verdict**: **FAILED**. `inflected_{lang}.pt` rejected.

---

### 2.7 Cycle 7: Master 50-Checkpoint Cross-Comparison & Reversion to Ground Truth
- **Action** (commit `0ea0e46`, *Source: [`reports/final_cross_checkpoint_comparison.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/final_cross_checkpoint_comparison.md)*):
  - Evaluated every checkpoint produced across all 16 phases and 6 patch cycles under identical conditions on locked Gold-Dev.
  - Demonstrated conclusively that every speculative post-hoc patch regressed model accuracy or introduced spurious false positives.
  - Reverted official project checkpoints to the genuine best baselines:
    - Konkani: `selftrained_konkani_best.pt` (Phase 10) $\to$ `final_konkani.pt`
    - Assamese: `augmented_assamese.pt` (Phase 11) $\to$ `final_assamese.pt`
    - Sinhala: `augmented_sinhala.pt` (Phase 11) $\to$ `final_sinhala.pt`

---

## 3. Methodological Takeaways for the Limitations Section

1. **Weak Supervision Rule Leaks into Evaluation**: Automatically generating or augmenting evaluation sets with heuristic rule taggers fatally corrupts test independence. Independent manual curation is non-negotiable.
2. **Small-Sample Fragility in Long-Sentence Buckets**: When slicing small evaluation sets by sentence length, individual buckets often contain $< 5$ entities, rendering percentage deltas volatile and uninformative.
3. **In-Sample Calibration Fallacy**: Calibration of weak supervision weights on Dev sets produces deceptive $100\%$ precision estimates that deflate by 10–20 pp on out-of-sample data.
4. **Subword Conjunct Brittleness**: In Brahmic scripts, minor Unicode encoding variations (e.g. ZWJ vs non-ZWJ conjuncts) severely disrupt character-level convolutions if canonical normalization is not enforced prior to embedding.
