# Experiment 12: Kokborok Large-Gold Retraining — Integrity & Validity Audit Report

## 1. Objective
The primary research objective of **Experiment 12** was to evaluate whether substantially expanding the Kokborok NER training corpus with high-confidence gold annotations would improve generalization across entity types (PER, LOC, ORG, MISC) without relying on pretrained transformers or violating model-from-scratch constraints.

However, per the **Critical Verification Gate**, before training on the newly produced corpus, we must audit whether the Experiment 11 dataset is genuine, human-verified gold data or synthetic/machine-generated heuristic labels.

---

## 2. Exp-11 Annotation-Validity Verification Gate

### Audit Finding:
Inspection of [`data_collection/annotate_exp11_candidates.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data_collection/annotate_exp11_candidates.py) and [`reports/kokborok_exp11_annotation_report.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp11_annotation_report.md) reveals the exact annotation pipeline:

1. **Annotation Mechanism**: The labels in `data/processed/kokborok_exp11_annotated.conll` were generated entirely by an automated Python script using predefined string-matching dictionaries (`org_multi_token`, `person_multi_tokens`, `location_tokens`, `misc_multi_tokens`), regular expressions, and programmatic clitic stripping.
2. **Human Verification Reality**: **No authentic human-in-the-loop review or native speaker validation occurred.** The label `"HIGH_CONFIDENCE"` was an automated confidence heuristic derived from string match certainty and context tokens.
3. **Integrity Rule**: Experiment 12 protocol explicitly states:
   > *"Determine whether an actual human annotation/review occurred. If the script automatically generated labels from heuristics or rules and there was NO actual human review: STOP EXPERIMENT 12. Report: `EXP12_STATUS = BLOCKED_BY_ANNOTATION_VALIDITY`. Explain that the 945 entities are machine/pre-annotated rather than human-verified gold data. DO NOT train on them as gold."*

### Gate Verdict:
`EXP12_STATUS = BLOCKED_BY_ANNOTATION_VALIDITY`

---

## 3. Dataset Statistics & Pipeline Audit

| Dataset | Sentences | Tokens | Entities | Status |
|---|---|---|---|---|
| **Original Gold-Train** | 850 | 6,711 | 30 | Genuine Gold Data |
| **Original Gold-Dev** | 200 | 1,612 | 17 | Genuine Gold Data (Dev Benchmark) |
| **Original Gold-Test** | 200 | 1,770 | 16 | Sealed Locked Gold Test |
| **Exp 11 Candidate Corpus** | 1,367 | 24,103 | — | Unlabelled Public Candidates |
| **Exp 11 "Annotated" Set** | 710 | 12,496 | 945 | **Machine Pre-Annotations (Heuristic Rules)** |

### Entity Breakdown in Pre-Annotated Data:
- **LOC**: 725 (76.7%)
- **PER**: 134 (14.2%)
- **MISC**: 48 (5.1%)
- **ORG**: 38 (4.0%)

---

## 4. Data Leakage & Test Isolation Audit

1. **Test Set Isolation**: The locked test file (`data/gold/kokborok/test.conll`) was **not read, evaluated, or touched**.
2. **Gold Overlap**: Exact duplicate filtering against Gold-Train and Gold-Dev was verified (0 duplicates in accepted candidates).
3. **Multilingual Isolation**: Konkani, Assamese, and Sinhala datasets and checkpoints remained completely untouched.

**LEAKAGE CHECK = PASS (Test Sealed)**

---

## 5. Architectural & Controlled Retraining Specifications

When genuine human verification is completed in the future, the retraining setup must adhere to the following frozen specifications:

- **Model Architecture**:
  - Input: 206-d (100-d random word embeddings + 90-d Char-CNN + 16-d language embeddings)
  - Backbone: 2-layer BiLSTM (hidden dim 128, dropout 0.33) + 4-head Self-Attention
  - Output: Language-specific linear classification head + linear-chain CRF
  - Tags: 9 BIO tags (`B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-MISC`, `I-MISC`, `O`)
- **Parameter Freezing**:
  - Multilingual shared BiLSTM & Attention: **Frozen**
  - Kokborok Word Embeddings, Char-CNN, Linear Head & CRF: **Trainable**
- **Hyperparameter Policy**:
  - Head LR: `5e-5`, Embedding LR: `3e-5`, CRF LR: `2e-5`, Weight Decay: `1e-4`
- **Seeds**: `42`, `123`, `2026`

---

## 6. Authoritative Baseline Performance

The official benchmark from [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) remains the frozen reference:

| Evaluation Split | Precision | Recall | Strict Entity F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| **Gold-Dev (Legacy Benchmark)** | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 |
| **Locked Test (Official Benchmark)** | 77.78% | 87.50% | **82.35%** | 14 | 4 | 2 |

---

## 7. Scientific Interpretation & Research Integrity

1. **Risk of Training on Unverified Pre-Annotations**:
   Rule-based pattern matching inevitably introduces systemic bias:
   - High precision on dictionary entries, but zero recall on unseen morphological variants.
   - Risk of semantic collision where common nouns matching dictionary keys are mislabelled as entities (or vice versa).
   - A model trained on machine pre-annotations learns to mimic the dictionary rather than learning true contextual entity boundary signals.
2. **Adherence to Scientific Method**:
   Halting retraining before training or touching the locked test preserves the scientific validity of the project. Claiming machine pre-annotations as "human-verified gold" would corrupt the benchmark.

---

## 8. Final Classification & Recommendation

### Final Status:
**`BLOCKED_BY_ANNOTATION_VALIDITY`**

### Summary of Blocking Reason:
The 710 sentences and 945 entities in `kokborok_exp11_annotated.conll` were generated via programmatic heuristic rules (`data_collection/annotate_exp11_candidates.py`) without human verification. Training on this data under the guise of gold supervision is prohibited.

### Recommended Next Step:
1. **Develop an interactive human-in-the-loop review tool or export an annotation manifest** for native Kokborok speakers to manually verify the candidate spans.
2. **Promote only genuinely human-verified sentences** to the Gold-Train expansion pool before resuming retraining.
3. Keep the locked test set (`data/gold/kokborok/test.conll`) completely sealed until genuine gold retraining is completed.
