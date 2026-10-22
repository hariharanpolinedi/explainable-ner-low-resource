# Figures Manifest & Visual Artifacts Catalog

**Paper Section Target**: Results / Visualizations / Figures & Captions  
**Figures Location**: `paper_assets/figures/`  
**Total Visual Assets**: 16 files (7 PNG figures + 9 HTML heatmaps)  

---

## 1. Summary Table of Figure Artifacts

| New File Name in `paper_assets/figures/` | Format | Original Source File | Figure Type / Content | Camera-Ready Caption |
| :--- | :---: | :--- | :--- | :--- |
| `fig_confusion_matrix_entities_konkani.png` | PNG | `reports/phase14_confusion_matrices/konkani_entity_matrix.png` | Entity Confusion Matrix | **Figure 1a**: Konkani span-level entity confusion matrix across PER, LOC, ORG, MISC, and Background (O) on locked gold-test. |
| `fig_confusion_matrix_tokens_konkani.png` | PNG | `reports/phase14_confusion_matrices/konkani.png` | Token-Level Confusion Matrix | **Figure 1b**: Konkani token-level 9-class BIO confusion matrix on locked gold-test. |
| `fig_confusion_matrix_entities_assamese.png` | PNG | `reports/phase14_confusion_matrices/assamese_entity_matrix.png` | Entity Confusion Matrix | **Figure 2a**: Assamese span-level entity confusion matrix across PER, LOC, ORG, MISC, and Background (O) on locked gold-test. |
| `fig_confusion_matrix_tokens_assamese.png` | PNG | `reports/phase14_confusion_matrices/assamese.png` | Token-Level Confusion Matrix | **Figure 2b**: Assamese token-level 9-class BIO confusion matrix on locked gold-test. |
| `fig_confusion_matrix_entities_sinhala.png` | PNG | `reports/phase14_confusion_matrices/sinhala_entity_matrix.png` | Entity Confusion Matrix | **Figure 3a**: Sinhala span-level entity confusion matrix across PER, LOC, ORG, MISC, and Background (O) on locked gold-test. |
| `fig_confusion_matrix_tokens_sinhala.png` | PNG | `reports/phase14_confusion_matrices/sinhala.png` | Token-Level Confusion Matrix | **Figure 3b**: Sinhala token-level 9-class BIO confusion matrix on locked gold-test. |
| `fig_selftraining_curve.png` | PNG | `reports/phase10_selftraining_curve.png` | Multi-Round Training Curve | **Figure 4**: Iterative self-training dev F1 progression across rounds 0–2 with confidence threshold tightening and automated rollback guard. |
| `heatmap_konkani_org_1.html` | HTML | `reports/phase13_heatmap_gallery/konkani_org_1.html` | Attention Attribution Heatmap | **Figure 5a**: Konkani ORG attribution for entity `कला` (B-ORG) showing strong gradient-weighted focus on context tokens `ओल्ड` and `आनी`. |
| `heatmap_konkani_loc_1.html` | HTML | `reports/phase13_heatmap_gallery/konkani_loc_1.html` | Attention Attribution Heatmap | **Figure 5b**: Konkani LOC attribution for entity `गोंय` (B-LOC) highlighting syntactic postposition and governing clause verbs. |
| `heatmap_konkani_per_1.html` | HTML | `reports/phase13_heatmap_gallery/konkani_per_1.html` | Attention Attribution Heatmap | **Figure 5c**: Konkani PER attribution for entity `नाईक` (I-PER) illustrating lexical surname linkage to preceding given name. |
| `heatmap_assamese_loc_1.html` | HTML | `reports/phase13_heatmap_gallery/assamese_loc_1.html` | Attention Attribution Heatmap | **Figure 6a**: Assamese LOC attribution for entity `গুৱাহাটী` (B-LOC) showing directional reliance on toponymic headwords. |
| `heatmap_assamese_per_1.html` | HTML | `reports/phase13_heatmap_gallery/assamese_per_1.html` | Attention Attribution Heatmap | **Figure 6b**: Assamese PER attribution for entity `বৰুৱা` (I-PER) demonstrating title and honorific prefix attention routing. |
| `heatmap_assamese_org_1.html` | HTML | `reports/phase13_heatmap_gallery/assamese_org_1.html` | Attention Attribution Heatmap | **Figure 6c**: Assamese ORG attribution for entity `বিদ্যালয়` (I-ORG) showing associative alignment to institutional descriptors. |
| `heatmap_sinhala_loc_1.html` | HTML | `reports/phase13_heatmap_gallery/sinhala_loc_1.html` | Attention Attribution Heatmap | **Figure 7a**: Sinhala LOC attribution for entity `කොළඹ` (B-LOC) resolving capital city context via locative markers. |
| `heatmap_sinhala_per_1.html` | HTML | `reports/phase13_heatmap_gallery/sinhala_per_1.html` | Attention Attribution Heatmap | **Figure 7b**: Sinhala PER attribution for entity `සිල්වා` (I-PER) showing honorific `මහතා` and patronymic coupling. |
| `heatmap_sinhala_org_1.html` | HTML | `reports/phase13_heatmap_gallery/sinhala_org_1.html` | Attention Attribution Heatmap | **Figure 7c**: Sinhala ORG attribution for entity `විශ්වවිද්‍යාලය` (B-ORG) highlighting educational suffix salience. |

---

## 2. Notes on Missing / Non-Existent Figures

- **Phase 12 Hyperparameter Search (`reports/phase12_hparam_search.md`)**:
  - *Status*: **TABULAR ONLY (NO IMAGE ARTIFACT)**.
  - *Traceability Note*: Phase 12 records a 15-trial search across hidden dimensions, heads, layers, learning rates, and language dimensions formatted entirely as Markdown tables in `reports/phase12_hparam_search.md`. No raster or vector plot file was generated during Phase 12.
