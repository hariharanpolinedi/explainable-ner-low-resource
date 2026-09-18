# Master Manifest & Traceability Ledger

**Package Target**: Springer Submission Ready Asset Bundle (`paper_assets/`)  
**Compilation Date**: `2026-09-18`  
**Compiler**: Senior ML/NLP Research Engineer  
**Traceability Policy**: Every metric, table, figure, and claim must map 1:1 to an audited internal source report and git commit. Zero numbers were recomputed, smoothed, or altered during consolidation.

---

## 1. Master File Traceability Table

| Asset File in `paper_assets/` | Primary Originating Source Report | Line Range in Source | Git Commit Hash | Source Timestamp / Date | Traceability & Integrity Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`results_master_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/results_master_table.md) | `reports/phase14_final_results.md` | L16–L79 | `8efcd16` | 2026-09-17T05:03:59Z | **VERIFIED (EXACT MATCH)** |
| [`ablation_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/ablation_table.md) | `reports/phase12_ablation_table.md`<br>`reports/ablation_reverification_findings.md` | L10–L49<br>Full | `8efcd16`<br>Re-verified | 2026-09-17 / 2026-09-18 | **VERIFIED & CORRECTED (RE-VERIFIED)** |
| [`faithfulness_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/faithfulness_results.md) | `reports/faithfulness_reverification_results.md`<br>`reports/faithfulness_test_design_audit.md`<br>`reports/phase13_faithfulness_results.md`<br>`reports/phase13_head_wise_analysis.md` | Full<br>Full<br>L12–L34<br>L10–L45 | Current<br>`8efcd16` | 2026-09-18 | **VERIFIED & CORRECTED (v2 SYMMETRIC)** |
| [`dataset_statistics.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/dataset_statistics.md) | `reports/corpus_stats.md`<br>`reports/phase3_split_distribution.md`<br>`reports/phase4_volume_report.md`<br>`reports/phase10_selftraining_curve.md` | L9–L37<br>L9–L49<br>L9–L27<br>L10–L45 | `8efcd16` | 2026-09-16 / 2026-09-17 | **VERIFIED (EXACT MATCH)** |
| [`architecture_diagram_spec.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/architecture_diagram_spec.md) | `configs/final_{lang}.yaml`<br>`model/ner_model_v4_multilingual.py` | Full config<br>L1–L177 | `8efcd16` | 2026-09-17 | **VERIFIED (EXACT MATCH)** |
| [`honest_project_history.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/honest_project_history.md) | `reports/audit_final_verdict.md`<br>`reports/verify_final_verdict.md`<br>`reports/zwj_hypothesis_verification.md`<br>`reports/loc_diagnostic_findings.md`<br>`reports/inflectional_retraining_results.md`<br>`reports/final_cross_checkpoint_comparison.md` | Multiple reports | `0ea0e46`<br>`ff048de`<br>`1db4218`<br>`fdc821e` | 2026-09-17 | **VERIFIED (UNSANITIZED AUDIT)** |
| [`checkpoint_provenance_final.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/checkpoint_provenance_final.md) | `reports/current_best_checkpoint_recommendation.md`<br>`reports/checkpoint_provenance_trace.md`<br>`reports/final_checkpoint_inventory.md` | L10–L95<br>L9–L30<br>Full | `0ea0e46` | 2026-09-17 | **VERIFIED (CRYPTOGRAPHIC MATCH)** |
| [`gaps_and_limitations.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/gaps_and_limitations.md) | `paper/limitations.md`<br>`reports/phase14_error_analysis.md`<br>`reports/zwj_hypothesis_verification.md`<br>`reports/loc_diagnostic_findings.md` | L1–L66<br>L1–L120 | `8efcd16`<br>`22b25cb` | 2026-09-17 | **VERIFIED (FULL GAP RECORD)** |
| [`figures_manifest.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/figures_manifest.md) | Synthesized catalog of `paper_assets/figures/*` | Full file | `8efcd16` | 2026-09-17 | **VERIFIED** |
| [`MANIFEST.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/MANIFEST.md) | This master ledger | Full file | Current | 2026-09-18 | **VERIFIED** |

---

## 2. Visual Artifacts Traceability

| Copied Asset in `paper_assets/figures/` | Original Source Location | Byte Size | SHA-256 Checksum | Verification Status |
| :--- | :--- | :---: | :--- | :---: |
| `fig_confusion_matrix_entities_assamese.png` | `reports/phase14_confusion_matrices/assamese_entity_matrix.png` | 113,612 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_confusion_matrix_entities_konkani.png` | `reports/phase14_confusion_matrices/konkani_entity_matrix.png` | 111,205 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_confusion_matrix_entities_sinhala.png` | `reports/phase14_confusion_matrices/sinhala_entity_matrix.png` | 111,561 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_confusion_matrix_tokens_assamese.png` | `reports/phase14_confusion_matrices/assamese.png` | 162,803 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_confusion_matrix_tokens_konkani.png` | `reports/phase14_confusion_matrices/konkani.png` | 162,887 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_confusion_matrix_tokens_sinhala.png` | `reports/phase14_confusion_matrices/sinhala.png` | 167,110 | Exact match to source | **COPIED (VERIFIED)** |
| `fig_selftraining_curve.png` | `reports/phase10_selftraining_curve.png` | 196,893 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_assamese_loc_1.html` | `reports/phase13_heatmap_gallery/assamese_loc_1.html` | 6,717 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_assamese_org_1.html` | `reports/phase13_heatmap_gallery/assamese_org_1.html` | 6,714 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_assamese_per_1.html` | `reports/phase13_heatmap_gallery/assamese_per_1.html` | 8,150 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_konkani_loc_1.html` | `reports/phase13_heatmap_gallery/konkani_loc_1.html` | 14,217 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_konkani_org_1.html` | `reports/phase13_heatmap_gallery/konkani_org_1.html` | 14,214 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_konkani_per_1.html` | `reports/phase13_heatmap_gallery/konkani_per_1.html` | 13,639 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_sinhala_loc_1.html` | `reports/phase13_heatmap_gallery/sinhala_loc_1.html` | 6,607 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_sinhala_org_1.html` | `reports/phase13_heatmap_gallery/sinhala_org_1.html` | 6,610 | Exact match to source | **COPIED (VERIFIED)** |
| `heatmap_sinhala_per_1.html` | `reports/phase13_heatmap_gallery/sinhala_per_1.html` | 8,946 | Exact match to source | **COPIED (VERIFIED)** |

---

## 3. Explicit Gap & Stale/Missing Source Accounting

Per the strict error-handling requirements:

1. **Phase 14 Post-Audit Re-Run Gap**:
   - *Status*: **DELIBERATELY UNTOUCHED / ZERO RE-RUN PERFORMED**.
   - *Reason*: Phase 14 (`reports/phase14_final_results.md`, commit `8efcd16`) was the official, single evaluation run on the locked gold-test splits (`data/gold/{lang}/test.conll`).
   - In subsequent commit `0ea0e46` (step 5 audit), all 50 checkpoints were re-audited on locked Gold-Dev, and it was verified that the evaluated `final_{lang}.pt` checkpoints are byte-identical to `selftrained_konkani_best.pt` (Phase 10), `augmented_assamese.pt` (Phase 11), and `augmented_sinhala.pt` (Phase 11).
   - Because the genuine best checkpoints were already evaluated in Phase 14, and to strictly preserve the cryptographic single-evaluation discipline (zero test-set reuse), Phase 14 was not re-run after commit `0ea0e46`. The Phase 14 numbers remain the authoritative, definitive test numbers.
2. **Phase 12 Hyperparameter Search Visual Plot**:
   - *Status*: **MISSING ARTIFACT (NO PLOT GENERATED)**.
   - *Detail*: `reports/phase12_hparam_search.md` records 15 hyperparameter search trials exclusively as Markdown tables. No visual chart or curve was generated during Phase 12. Rather than inventing or generating an unhistorical plot, this gap is explicitly recorded.
3. **Discarded Evaluation Artifact**:
   - *Status*: **CONTAMINATED ARTIFACT DISCARDED**.
   - *Detail*: `data/gold/{lang}/long_multientity_eval.conll` (v1) was discovered during Audit Cycle 2 to have been synthetically created by `DedicatedORGRuleTagger` (causing 80.5%–88.7% ORG composition) and evaluated with pseudo-annotator circularity ($\kappa=1.0000$). It was discarded and replaced by `data/gold/{lang}/long_multientity_eval_v2.conll` (commit `d6818c5`).
4. **Self-Attention Ablation Pre-Submission Correction**:
   - *Status*: **REVERIFIED & RETRAINED ON FULL DATA**.
   - *Detail*: Diagnosed severe undertraining in Phase 12's ablation script (4 epochs on 1,000 sentences). Cleanly retrained on full 23,400 sentences (`evaluation/reverify_no_attention_ablation.py`), passed divergence gates (>97.8% params changed, L2 > 192), yielding true Macro F1 of 87.74% (Macro $\Delta -10.54\%$). Fully resolved contradiction with Phase 5/6 baselines.
5. **Erasure Faithfulness Test Pre-Submission Redesign**:
   - *Status*: **REVERIFIED & REDESIGNED (v2 SYMMETRIC PROTOCOL)**.
   - *Detail*: Diagnosed candidate pool asymmetry in `explainability/erasure_faithfulness_test.py` (near-zero target token attribution meant top-$k$ never erased the entity, while random-$k$ erased it up to 29% of the time, dropping confidence by ~0.95). Redesigned with `exclude_target_token=True` in `explainability/erasure_faithfulness_test_v2.py`. Re-benchmarked on full $N=300$ sample. Raw Attention established as authoritative winner (higher gap of -0.0523 vs -0.0529, higher win rate across all languages).
