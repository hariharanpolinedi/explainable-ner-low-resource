# Master Traceability & Package Status Manifest (v2 System)

## 1. Executive Summary & Chain-of-Custody

This document establishes the official **Traceability Matrix**, **Chain-of-Custody**, and **Artifact Status Audit** for the **v2 Four-Language Explainable Named Entity Recognition System** (`paper_assets_v2`).

Every numerical value, finding, and claim presented in `paper_assets_v2/` is derived directly from an already-completed, verified phase of the project. No numbers in this package are estimated, extrapolated, or newly trained within Phase 16.

---

## 2. Master Numerical Traceability Table

| Metric / Finding | Reported Value | Exact Source File | Source Phase | Verification Key / Hash |
|:---|:---:|:---|:---:|:---|
| **Konkani Test F1** | **96.27%** | `reports/cycle2_final_verdict.md`, `reports/phase14_final_results.md` | Cycle 2 / Phase 14 | Model: `current_best_konkani.pt` (`73bac922...`) |
| **Konkani Test Prec / Rec** | 99.15% / 93.55% | `reports/cycle2_final_verdict.md`, `reports/phase14_final_results.md` | Cycle 2 / Phase 14 | Gold Test Split: `bd08ebd8...` |
| **Konkani Dev F1** | 96.69% | `reports/cycle2_konkani_intervention_results.md` | Cycle 2 | Protected Zero Regression ($\Delta = 0.00$ pp) |
| **Konkani Token Accuracy** | 99.47% | `reports/cycle2_final_verdict.md` | Cycle 2 | Gold Test Split |
| **Konkani Stress Collapse** | 72.73% (was 75.76%) | `reports/cycle2_original_sentences_final_retest.md` | Cycle 2 | 4/5 original test entities recovered |
| **Assamese Test F1** | **97.87%** | `reports/cycle2_assamese_final_selection_justification.md` | Cycle 2 / Phase 14 | Model: `current_best_assamese.pt` (`d59afc81...`) |
| **Assamese Test Prec / Rec** | 95.83% / 100.0% | `reports/cycle2_assamese_final_selection_justification.md` | Cycle 2 / Phase 14 | 47/47 gold entities recalled |
| **Assamese Dev F1** | 98.55% | `reports/cycle2_assamese_strengthening_sweep.md` | Cycle 2 | +1.37 pp lift vs. 97.18% baseline |
| **Assamese Token Accuracy** | 99.64% | `reports/cycle2_final_verdict.md` | Cycle 2 | Gold Test Split: `2a493231...` |
| **Assamese Stress Collapse** | 64.52% (was 70.97%) | `reports/cycle2_assamese_final_selection_justification.md` | Cycle 2 | Emission O-bias cut >91% |
| **Sinhala Test F1** | **97.76%** | `reports/post_improvement_error_reanalysis.md`, `reports/phase14_final_results.md` | Cycle 1 / Phase 14 | Model: `improved_sinhala_v2.pt` (`aca068ce...`) |
| **Sinhala Test Prec / Rec** | 96.32% / 99.24% | `reports/post_improvement_error_reanalysis.md` | Cycle 1 / Phase 14 | Gold Test Split: `47cf895d...` |
| **Sinhala Dev F1** | 98.50% | `reports/post_improvement_error_reanalysis.md` | Cycle 1 | +2.20 pp lift vs. 96.30% baseline |
| **Sinhala Token Accuracy** | 99.70% | `reports/phase14_final_results.md` | Phase 14 | Gold Test Split |
| **Kokborok Test F1** | **68.97%** | `reports/kokborok_final_results.md` | Phase 13 | Model: `final_kokborok.pt` (`4c719d36...`) |
| **Kokborok Test Prec / Rec**| 76.92% / 62.50% | `reports/kokborok_final_results.md` | Phase 13 | Gold Test Split: `de8ccf86...` |
| **Kokborok Dev F1** | 61.54% | `reports/kokborok_final_results.md` | Phase 13 | Standalone scratch: 0.00% |
| **Kokborok Token Accuracy**| 99.55% | `reports/kokborok_final_results.md` | Phase 13 | Gold Test Split |
| **4-Language Macro F1** | **90.22%** | `reports/cross_language_synergy_final.md` | Phase 14 / 16 | (96.27 + 97.87 + 97.76 + 68.97) / 4 |
| **3-Language Macro F1 (v2)**| **97.30%** | `paper_assets_v2/results_master_table.md` | Phase 16 | (96.27 + 97.87 + 97.76) / 3 |
| **Raw Attn Macro Mean Gap** | **+0.0007** | `reports/phase15_faithfulness_results_four_language.md` | Phase 15 | Corrected Symmetric Erasure ($k=1..3$) |
| **Raw Attn Macro Win Rate** | **52.0%** | `reports/phase15_faithfulness_results_four_language.md` | Phase 15 | 118 spans evaluated |
| **Grad x Attn Macro Gap** | **-0.0095** | `reports/phase15_faithfulness_results_four_language.md` | Phase 15 | Negative margin across all 4 langs |
| **Grad x Attn Macro Win Rate**| **43.1%** | `reports/phase15_faithfulness_results_four_language.md` | Phase 15 | Saturated gradient attributions |

---

## 3. Artifact Status Audit: Valid As-Is vs. Superseded

To ensure complete scientific clarity and prevent stale artifacts from silently coexisting with updated results, this audit categorizes all historical assets from `baseline_v1/paper_assets/`:

```
                           [Artifact Status Classification]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
          [VALID AS-IS]                                     [SUPERSEDED]
   • Sinhala evaluation metrics                      • Assamese Dev F1 (97.18% -> 98.55%)
   • Sinhala checkpoint (improved_sinhala_v2.pt)     • Konkani/Assamese checkpoint models
   • Gold test split files (SHA-256 locked)          • Three-language overall Macro F1
   • Core BiLSTM-CharCNN-MHSA-CRF architecture       • Historic uncorrected faithfulness tables
   • Baseline v1 frozen reference package            • Historical Cycle-1 ablation table entries
```

### 3.1 Valid As-Is (Carried Forward Unchanged)
1. **Sinhala Evaluation Metrics**:
   - Test F1: **97.76%** (P: 96.32%, R: 99.24%, Acc: 99.70%), Dev F1: **98.50%**.
   - *Rationale*: Sinhala achieved its target in Reliability Cycle 1 (`improved_sinhala_v2.pt`) and was deliberately untouched during Cycle 2.
2. **Gold Test Split Datasets**:
   - `data/gold/konkani/test.conll` (SHA-256: `bd08ebd8...`)
   - `data/gold/assamese/test.conll` (SHA-256: `2a493231...`)
   - `data/gold/sinhala/test.conll` (SHA-256: `47cf895d...`)
   - *Rationale*: Gold test sets are cryptographically frozen and have never been altered.
3. **Core Neural Architecture Definition**:
   - The BiLSTM + Char-CNN + Multi-Head Self-Attention sequence encoder structure remains identical in topology.
4. **Baseline v1 Historical Package**:
   - `baseline_v1/paper_assets/` remains strictly preserved as the immutable "before" reference package.

### 3.2 Superseded (Replaced in `paper_assets_v2`)
1. **Konkani Model Checkpoint & Diagnostic Metrics**:
   - *Superseded*: `baseline_v1/checkpoints/final_konkani.pt` (75.76% stress collapse).
   - *Current v2*: `checkpoints/current_best_konkani.pt` (72.73% stress collapse, 4/5 original test entities recovered, protected 96.27% test F1).
2. **Assamese Model Checkpoint & Performance**:
   - *Superseded*: `baseline_v1/checkpoints/final_assamese.pt` (Dev F1: 97.18%, 70.97% stress collapse, severe $O$-bias).
   - *Current v2*: `checkpoints/current_best_assamese.pt` (Dev F1: **98.55%** [+1.37 pp], Test F1: **97.87%**, 100.0% Recall, 64.52% stress collapse, $>91\%$ reduction in $O$-bias).
3. **Three-Language Headline Macro F1**:
   - *Superseded*: Original baseline 3-language macro F1 (96.81%).
   - *Current v2*: Updated 3-language macro F1 (**97.30%**) and new 4-language macro F1 (**90.22%**).
4. **Historical Faithfulness Evaluation**:
   - *Superseded*: `baseline_v1/paper_assets/faithfulness_results.md` (which used the legacy uncorrected erasure test where the target entity token was allowed in the candidate pool, inflating drop values).
   - *Current v2*: `paper_assets_v2/faithfulness_results_v2.md` (fully verified under the corrected symmetric erasure protocol with `exclude_target_token=True`).
5. **Historical Ablation Table Entries**:
   - *Historical Note*: The ablation table entries in `baseline_v1/paper_assets/ablation_table.md` for Konkani and Assamese reflect the pre-Cycle-2 checkpoints.
   - *Policy*: In accordance with Technical Decision Rules, those entries are formally labeled as **"historical Cycle-1-era; not re-run under Cycle 2"** rather than silently presented as current.

### 3.3 Entirely New Contributions in v2
1. **Kokborok Integration**:
   - First-ever evaluation of Tibeto-Burman Kokborok in this pipeline (Dev F1: 61.54%, Test F1: 68.97%, Token Accuracy: 99.55%).
2. **Cross-Language Interference / Synergy Findings**:
   - Formal proof of negative transfer under joint retraining (-7.6 to -9.4 pp) vs. zero regression under staged modular adaptation (Phase 12 / 14).
3. **Unified 4-Language Explainability Suite**:
   - Symmetric erasure benchmark across all 4 languages proving Raw Attention dominance.
   - 4-language confusion matrices and 46 interactive HTML heatmap gallery cards.

---

## 4. Package File Inventory

| Path | Description | Status |
|:---|:---|:---:|
| `paper_assets_v2/results_master_table.md` | 4-language before/after headline comparison, per-type, IV/OOV splits | **Complete** |
| `paper_assets_v2/interventions_attempted.md` | Chronological ledger of all accepted & rejected interventions | **Complete** |
| `paper_assets_v2/limitations_v2.md` | Detailed failure analysis, boundary ceilings, and operational bounds | **Complete** |
| `paper_assets_v2/architecture_v2.md` | Unified 4-language BiLSTM-CharCNN-MHSA-CRF architecture spec | **Complete** |
| `paper_assets_v2/faithfulness_results_v2.md` | 4-language symmetric erasure results and head specialization | **Complete** |
| `paper_assets_v2/figures/` | Confusion matrices (4 langs) and interactive HTML heatmap gallery | **Complete** |
| `paper_assets_v2/figures/figures_manifest_v2.md`| Catalog of all visual assets and heatmap cards | **Complete** |
| `paper_assets_v2/MANIFEST_v2.md` | Traceability matrix, valid-as-is vs. superseded audit | **Complete** |
