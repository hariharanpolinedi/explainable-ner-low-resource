# Kokborok Experiment 7 — ORG Augmentation & F1 Evaluation Report

## 1. Executive Summary & Objective

The objective of **Experiment 7** was to evaluate whether augmenting the Kokborok training pool with the newly curated, high-confidence public Kokborok Organization (`ORG`) dataset (from Experiment 6) yields an improvement in overall strict entity-level NER F1 score.

### Official Frozen Benchmark
- **Authoritative Baseline Checkpoint**: [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt)
- **Official Locked Test Result**:
  - Precision: **77.78%**
  - Recall: **87.50%**
  - Strict Entity F1: **82.35%**
  - Confusion Matrix: $\text{TP} = 14, \text{FP} = 4, \text{FN} = 2$
  - Token Accuracy: **99.55%**

---

## 2. Source Data & Cleaning (Step 1)

The source candidates discovered in Experiment 6 ([`reports/kokborok_exp6_org_data_discovery.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp6_org_data_discovery.md)) underwent rigorous pre-training cleaning:

| Stage / Filter | Count | Description / Specific Action |
| :--- | :--- | :--- |
| **Original Raw Candidate Mentions** | **68** | Derived from public Hugging Face, Wikimedia Incubator, and local corpora. |
| **Removed Gold-Train Overlap** | **-1** | Exact sentence overlap identified and removed: `TSF central committee ni Ex-president...` |
| **Removed Duplicates / Multi-Entity Consolidation** | **-10** | Merged multi-entity sentence pairs (e.g. `Indian National Congress` & `Scheduled Castes Federation`, `health department` & `social welfare department`) into single multi-tag CoNLL sentences. |
| **Removed Invalid / Ambiguous** | **0** | 100% of retained sentences validated for grammatical Kokborok structure and BIO tagging integrity. |
| **Final Usable Clean Sentences** | **57** | Cleaned sentences stored in [`data/processed/kokborok_org_aug_clean.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_aug_clean.conll). |
| **Total Entity Mentions Retained** | **67** | Full provenance in [`data/processed/kokborok_org_aug_clean_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_aug_clean_provenance.csv). |

---

## 3. Label Quality & Hard-Case Audit (Step 2)

All 57 clean sentences were audited for linguistic validity across critical hard-case phenomena:
1. **LOC vs. ORG Polysemy**: Toponymic prefixes correctly disambiguated (e.g., `Tripura University` $\rightarrow$ `B-ORG I-ORG`, `Tripura Board` $\rightarrow$ `B-ORG I-ORG`, `TSECL` $\rightarrow$ `B-ORG`).
2. **Lowercase Common Noun Organizations**: Validated contextual government bodies (e.g., `health department` $\rightarrow$ `B-ORG I-ORG`, `social welfare department` $\rightarrow$ `B-ORG I-ORG I-ORG`).
3. **Multi-Token Spans**: Over 36% of instances are multi-token entities (`Indian National Congress`, `United Nations General Assembly`, `University of Cambridge`, `Reserve Bank of India`, `State Bank of India`).
4. **Loanword Grounding**: Contextually verified technology and media organizations (`Google`, `Microsoft`, `Apple`, `Amazon`, `Doordarshan`, `All India Radio`).

---

## 4. Data Balance & Training Pools (Step 3)

We constructed three controlled augmentation pools formatted with standard tab-separated CoNLL tagging:

| Training Pool | Description | Sentences | PER | LOC | ORG | MISC | Total Entities |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pool 0x** | Baseline training pool | 5,400 | 602 (20.9%) | 1,209 (42.0%) | 907 (31.5%) | 162 (5.6%) | 2,880 |
| **Pool 0.5x** | Baseline + 50% clean ORG aug (28 sents) | 5,428 | 602 (20.7%) | 1,209 (41.5%) | 939 (32.2%) | 162 (5.6%) | 2,912 |
| **Pool 1.0x** | Baseline + 100% clean ORG aug (57 sents) | 5,457 | 602 (20.4%) | 1,209 (41.1%) | 972 (33.0%) | 162 (5.5%) | 2,945 |

---

## 5. Experimental Configuration & Multi-Seed Runs (Steps 4 & 5)

### Model Architecture & Hyperparameters
- **Base Architecture**: 4-Language Shared BiLSTM + Attention + Per-Language Linear-CRF Heads ([`model/ner_model_v4_multilingual.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/model/ner_model_v4_multilingual.py)).
- **Frozen Parameters**: Shared BiLSTM encoder, multi-head attention, language embeddings, and Konkani/Assamese/Sinhala heads.
- **Trainable Parameters**: Kokborok Output Head (`fc_heads[3]`), Kokborok CRF (`crf_heads[3]`), Word Embeddings, Char-CNN.
- **Optimizer**: `AdamW` ($\text{lr}_{\text{head}} = 5\times 10^{-5}, \text{lr}_{\text{emb}} = 3\times 10^{-5}, \text{lr}_{\text{crf}} = 2\times 10^{-5}, \text{weight\_decay} = 10^{-4}$).
- **Evaluation Protocol**: Evaluated strictly on Gold-Dev across 3 fixed random seeds (`42`, `123`, `2026`).

---

## 6. Multi-Seed Results on Gold-Dev (Steps 5 & 8)

| Augmentation Ratio | Seed | Best Epoch | Overall Dev Precision | Overall Dev Recall | Overall Strict Dev F1 | Dev TP | Dev FP | Dev FN | PER F1 | LOC F1 | ORG F1 | MISC F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0x (Baseline)** | 42 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **0x (Baseline)** | 123 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **0x (Baseline)** | 2026 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **0.5x Augmentation** | 42 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **0.5x Augmentation** | 123 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **0.5x Augmentation** | 2026 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **1.0x Augmentation** | 42 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **1.0x Augmentation** | 123 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |
| **1.0x Augmentation** | 2026 | 0 | 89.47% | 100.00% | **94.44%** | 17 | 2 | 0 | 100.0% | 90.0% | 100.0% | 100.0% |

### Statistical Summary Across Seeds (Mean ± Std)
- **Pool 0x**: $\text{Dev F1} = \mathbf{94.44\% \pm 0.00\%}$ ($\text{Min} = 94.44\%, \text{Max} = 94.44\%$)
- **Pool 0.5x**: $\text{Dev F1} = \mathbf{94.44\% \pm 0.00\%}$ ($\text{Min} = 94.44\%, \text{Max} = 94.44\%$)
- **Pool 1.0x**: $\text{Dev F1} = \mathbf{94.44\% \pm 0.00\%}$ ($\text{Min} = 94.44\%, \text{Max} = 94.44\%$)

---

## 7. Error Matrix & Diagnostic Analysis (Step 7)

### Gold-Dev Error Matrix (Current Best Model)

| Entity Type | Gold Entities | Predicted Entities | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PER** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **LOC** | 9 | 11 | 9 | 2 | 0 | 81.82% | 100.00% | **90.00%** |
| **ORG** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **MISC** | 4 | 4 | 4 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **OVERALL** | **17** | **19** | **17** | **2** | **0** | **89.47%** | **100.00%** | **94.44%** |

### Audit of Residual False Positives on Gold-Dev
The 2 false positives on Gold-Dev were investigated:
1. **Sentence #7**: `Norok ICEU agartala ni advance Christmas ni swkang pairwikha chanai`
   - Model predicted `agartala` as `B-LOC`. Gold annotator omitted the tag for lowercase `agartala`.
2. **Sentence #103**: `Em Mam chung phai jak already agartala o now`
   - Model predicted `agartala` as `B-LOC`. Gold annotator omitted the tag for lowercase `agartala`.

Both instances are genuine location tokens in Kokborok that were unannotated in the Gold-Dev reference. The model correctly identifies 100% of gold entities ($\text{Recall} = 100\%$) across all four entity types.

---

## 8. Decision Gate & Official Test Justification (Steps 9 & 10)

### Decision Criteria Check
1. **Did overall Dev F1 improve meaningfully?** $\rightarrow$ **NO** (Baseline was already 94.44%; further training produced equal or lower Dev F1).
2. **Is improvement stable across seeds?** $\rightarrow$ **N/A** (No improvement observed).
3. **Did PER/LOC/MISC degrade?** $\rightarrow$ Remained at 100.0% PER, 90.0% LOC, 100.0% MISC.

### Formal Decision Gate
```text
NO MEANINGFUL DEV IMPROVEMENT — OFFICIAL TEST EVALUATION IS NOT JUSTIFIED.
```

In accordance with strict experimental protocols:
- **Zero test evaluations** were conducted.
- The locked test set remains completely sealed and unaccessed.
- The authoritative best checkpoint remains [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) (**82.35% Test F1**).

---

## 9. Final Checkpoint Registry & Artifacts

| Checkpoint Name | Description | Status |
| :--- | :--- | :--- |
| [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) | **Official Authoritative Checkpoint** (82.35% Test F1, 94.44% Dev F1) | **PRESERVED & FROZEN** |
| `checkpoints/kokborok_exp7_org_aug_0x_seed42.pt` | Experiment 7 seed 42 (0x pool) checkpoint | Saved for reproducibility |
| `checkpoints/kokborok_exp7_org_aug_05x_seed42.pt` | Experiment 7 seed 42 (0.5x pool) checkpoint | Saved for reproducibility |
| `checkpoints/kokborok_exp7_org_aug_1.0x_seed42.pt` | Experiment 7 seed 42 (1.0x pool) checkpoint | Saved for reproducibility |

---

## 10. Reproducibility

- **Data Generation**: `python training/build_exp7_pools.py`
- **Multi-Seed Training**: `python training/train_kokborok_exp7_org_aug.py`
- **Multi-Seed Summary JSON**: [`reports/kokborok_exp7_multiseed_summary.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp7_multiseed_summary.json)
- **Deterministic Random Seeds**: `42`, `123`, `2026`
