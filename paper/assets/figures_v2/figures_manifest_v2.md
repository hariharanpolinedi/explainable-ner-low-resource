# Figures & Heatmap Gallery Manifest (v2 System)

## 1. Overview

This directory contains the final visual assets, confusion matrices, and interactive attention attribution heatmaps representing the locked **v2 Four-Language System** (Konkani, Assamese, Sinhala, Kokborok).

All confusion matrices reflect predictions generated on the locked gold test sets (`data/gold/*/test.conll`) by the final checkpoints:
- **Konkani**: `checkpoints/current_best_konkani.pt`
- **Assamese**: `checkpoints/current_best_assamese.pt`
- **Sinhala**: `checkpoints/improved_sinhala_v2.pt`
- **Kokborok**: `checkpoints/final_kokborok.pt`

All attention attribution heatmaps were generated during Phase 15 using the validated **Raw Multi-Head Self-Attention Attribution** module under the symmetric erasure evaluation protocol.

---

## 2. Confusion Matrix Catalog

| File Name | Language | Type | Description | Source Phase |
|:---|:---:|:---:|:---|:---:|
| `konkani.png` | Konkani (`gom`) | Token-Level | Token-level 9x9 confusion matrix (BIO tags) | Phase 14 |
| `konkani_entity_matrix.png` | Konkani (`gom`) | Entity-Level | Exact span 5x5 entity-level confusion matrix | Phase 14 |
| `assamese.png` | Assamese (`asm`) | Token-Level | Token-level 9x9 confusion matrix (BIO tags) | Phase 14 |
| `assamese_entity_matrix.png` | Assamese (`asm`) | Entity-Level | Exact span 5x5 entity-level confusion matrix | Phase 14 |
| `sinhala.png` | Sinhala (`sin`) | Token-Level | Token-level 9x9 confusion matrix (BIO tags) | Phase 14 |
| `sinhala_entity_matrix.png` | Sinhala (`sin`) | Entity-Level | Exact span 5x5 entity-level confusion matrix | Phase 14 |
| `kokborok.png` | Kokborok (`kok`) | Token & Entity | Complete token & span confusion breakdown | Phase 13 |

---

## 3. Representative Interactive Attention Heatmap Gallery

The interactive standalone HTML files visualize token-to-token attention attribution for gold entity spans across all four languages. Each card displays the input sentence, gold and predicted entities, and the color-intensity heatmap of contextual attention weights $\alpha_{i,j}$.

### 3.1 Konkani (`gom`)
- `konkani_per_1.html`, `konkani_per_2.html`, `konkani_per_3.html`: Person name attributions highlighting honorifics and patronymic markers.
- `konkani_loc_1.html`, `konkani_loc_2.html`, `konkani_loc_3.html`: Location attributions highlighting geographic postpositions.
- `konkani_org_1.html`, `konkani_org_2.html`, `konkani_org_3.html`: Organization names showing multi-word corporate suffix binding.
- `konkani_misc_1.html`, `konkani_misc_2.html`, `konkani_misc_3.html`: Miscellaneous proper nouns and calendar terms.

### 3.2 Assamese (`asm`)
- `assamese_per_1.html`, `assamese_per_2.html`, `assamese_per_3.html`: Person name attributions highlighting nominal suffixes and titles.
- `assamese_loc_1.html`, `assamese_loc_2.html`, `assamese_loc_3.html`: Location spans demonstrating case clitic resolution.
- `assamese_org_1.html`, `assamese_org_2.html`, `assamese_org_3.html`: Government and administrative organizational mentions.
- `assamese_misc_1.html`, `assamese_misc_2.html`: Miscellaneous cultural entities.

### 3.3 Sinhala (`sin`)
- `sinhala_per_1.html`, `sinhala_per_2.html`, `sinhala_per_3.html`: Person entities in long-context clauses, including recovered leading entities.
- `sinhala_loc_1.html`, `sinhala_loc_2.html`, `sinhala_loc_3.html`: Location entities highlighting Brahmic script postpositional markers.
- `sinhala_org_1.html`, `sinhala_org_2.html`, `sinhala_org_3.html`: Institutional and commercial entities.
- `sinhala_misc_1.html`, `sinhala_misc_2.html`, `sinhala_misc_3.html`: Event and calendar entity attributions.

### 3.4 Kokborok (`kok`)
- `kokborok_per_1.html`, `kokborok_per_2.html`: Person name attributions in Latin script orthography.
- `kokborok_loc_1.html`, `kokborok_loc_2.html`, `kokborok_loc_3.html`: Geographic place names in Tripura.
- `kokborok_org_1.html`, `kokborok_org_2.html`: Cultural and political organizational entities.
- `kokborok_misc_1.html`, `kokborok_misc_2.html`, `kokborok_misc_3.html`: Miscellaneous entity spans demonstrating cross-lingual sub-word feature transfer.

---

## 4. Viewing Instructions

All HTML heatmap cards are fully self-contained with embedded CSS and JavaScript. They can be opened in any standard web browser (Chrome, Edge, Firefox) without requiring an active web server or internet connection.
