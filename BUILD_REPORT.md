# LaTeX + Figure + PDF Build Report

**Project:** *Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution*  
**Scope:** Final LaTeX + Figure + PDF Build of Section 7 (Results & Discussion) and Section 8 (Conclusion & Future Scope)  
**Date:** October 1, 2026  
**Compiler Engine:** Tectonic 0.15.0 (XeTeX-based self-contained modern TeX engine)  
**Document Class:** Springer Nature (`sn-jnl.cls`, `sn-mathphys-num`)

---

## 1. Executive Summary

| Metric | Build Result |
|---|---|
| **Compilation Status** | **SUCCESSFUL** (`exit code 0`) |
| **Primary Output PDF** | [`Section_7_8_Results_Conclusion_Springer_Final.pdf`](file:///c:/Users/harih/Desktop/NLP-MARK-3/Section_7_8_Results_Conclusion_Springer_Final.pdf) |
| **PDF Page Count** | **14 Pages** (complete, balanced layout without orphan trailing pages) |
| **PDF File Size** | 1.62 MiB (high-resolution embedded vector/raster graphics) |
| **Section Hierarchy** | **7 Results and Discussion** (7.1 – 7.10) & **8 Conclusion and Future Scope** (8.1 – 8.2) |
| **Total Publication Figures Generated** | **10 figures** (dual format: 300 DPI PNG + vector PDF) |
| **Total Publication Figures Inserted** | **10 figures** (Figures 3 through 12, synchronized with Section 6) |
| **Total Tables Formatted** | **5 tables** (Tables 5 through 9, synchronized with Section 6) |
| **Undefined References / Labels** | **0** (all citations, figure labels, and table labels cleanly resolved) |
| **Fatal Errors** | **0** |

---

## 2. Final Correction Pass Summary (Pass 6)

In response to reviewer feedback on visual presentation and scientific precision:
1. **Figure 12 Kokborok Entity Matrix Harmonization**:
   - *Investigation*: Discovered that the previous `reports/kokborok_confusion_matrix.png` combined a 9x9 BIO token matrix and an entity matrix side-by-side, creating format inconsistency with Konkani, Assamese, and Sinhala.
   - *Resolution*: Extracted the authoritative locked test predictions from `reports/experiment_logs/kokborok_test_eval_lock.json` and generated `reports/phase14_confusion_matrices/kokborok_entity_matrix.png` (exact 5x5 entity format: `PER`, `LOC`, `ORG`, `MISC`, `O`). Updated `make_figure_k()` in `scripts/generate_publication_figures.py` to compile all four language matrices into a uniform, large, readable 2x2 grid.
2. **Figure 5 & Figure 7 Axis Calibration**:
   - Replaced truncated y-axes (previously 90–101% and 80–105%) with standard, non-exaggerating 0–100% full scales (`ylim=(0, 115)`, `yticks=np.arange(0, 101, 20)`). This prevents visual exaggeration of minor percentage-point differences while keeping bar labels completely readable.
3. **Training Sentence Count Rectification (Section 7.9)**:
   - Line 348 was corrected from "fewer than 850 training sentences per language" to `"at most 850 training sentences per language (850 for Konkani, Assamese, and Sinhala; 849 for Kokborok)"`, perfectly aligning with Section 7.7.
4. **Figure 12 Placement & Layout Optimization**:
   - Adjusted graphic scale (`width=0.82\columnwidth`) and vertical spacing so that Figure 12 sits cleanly immediately below Section 7.8 Error Analysis text without artificial gaps.
5. **Section 8.1 (Conclusion) Substantive Expansion**:
   - Expanded the previously brief Conclusion into six structured, evidence-grounded paragraphs:
     - *From-scratch architecture contribution*: Zero-pretrained constraint, learned word/CNN-grapheme embeddings, language conditioning, BiLSTM, multi-head attention, shared CRF.
     - *Four-language performance findings*: F1 scores of 96.27% (Konkani), 97.87% (Assamese), 97.76% (Sinhala), and 82.35% (Kokborok); token accuracies >99.4%; baseline comparisons (+0.40, +0.01, +0.72 pp).
     - *Explainability findings*: Target-token attribution audit (>92% receiving $\le 0.001$ attribution mass; mean ranks 16.94–20.63).
     - *Faithfulness findings*: Divergent cross-lingual behavior under token erasure (+0.0127 to +0.0412 in Konkani vs negative gaps in Assamese and Kokborok).
     - *Kokborok resource constraint significance*: 849 sentences containing only 30 entity mentions (4.5/1000 tokens vs 31.4–47.1 in others); exclusion of noisy pseudo-annotations.
     - *Practical and research significance*: Demonstrates a reproducible methodology for transparent sequence labeling in computational environments lacking foundation models.
   - Utilized slight vertical adjustments so Section 8 and Future Scope fit completely across Pages 13 and 14 without spilling an orphan trailing page onto Page 15.

---

## 3. Publication Figures Generated & Inserted

All 10 figures were generated strictly from verified experimental artifacts via [`scripts/generate_publication_figures.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/scripts/generate_publication_figures.py):

| Fig. # | Topic / Title | Source File | Format | Inserted Location | PDF Page |
|---|---|---|---|---|---|
| **Fig. 3** | Entity-Level F1 by Language | `figures/fig_a_entity_f1.png` | 300 DPI / Vector PDF | Section 7.2 | **Page 3** (top) |
| **Fig. 4** | Precision / Recall / F1 Breakdown | `figures/fig_b_precision_recall_f1.png` | 300 DPI / Vector PDF | Section 7.2 | **Page 3** (bottom) |
| **Fig. 5** | Final Model vs. Joint Baseline | `figures/fig_c_final_vs_joint_baseline.png` | 300 DPI / Vector PDF | Section 7.3 | **Page 4** (center) |
| **Fig. 6** | Actual Attribution Heatmap (Gold-Dev) | `figures/fig_d_attribution_heatmap.png` | 300 DPI / Vector PDF | Section 7.5 | **Page 6** (top) |
| **Fig. 7** | Target-Token Near-Zero Attribution | `figures/fig_e_target_token_near_zero.png` | 300 DPI / Vector PDF | Section 7.5 | **Page 7** (top) |
| **Fig. 8** | Target-Token Attribution Mean Rank | `figures/fig_f_target_token_rank.png` | 300 DPI / Vector PDF | Section 7.5 | **Page 7** (center) |
| **Fig. 9** | Faithfulness Gap Line Graph ($k \in \{1,2,3\}$) | `figures/fig_g_faithfulness_line.png` | 300 DPI / Vector PDF | Section 7.6 | **Page 8** (bottom) |
| **Fig. 10** | Faithfulness Gap Heatmap | `figures/fig_h_faithfulness_heatmap.png` | 300 DPI / Vector PDF | Section 7.6 | **Page 9** (center) |
| **Fig. 11** | Gold Training Resource & Density Audit | `figures/fig_i_data_density.png` | 300 DPI / Vector PDF | Section 7.7 | **Page 10** (center) |
| **Fig. 12** | Test Confusion Matrices (4 Languages) | `figures/fig_k_confusion_matrices.png` | 300 DPI / Vector PDF | Section 7.8 | **Page 12** (top) |

---

## 3. Data Integrity & Scientific Verification

1. **Locked-Test Final Metrics Preserved**:
   - **Konkani**: Precision = 99.15%, Recall = 93.55%, F1 = **96.27%**, Token Accuracy = 99.47% (116 TP, 1 FP, 8 FN / 124 mentions)
   - **Assamese**: Precision = 95.83%, Recall = 100.00%, F1 = **97.87%**, Token Accuracy = 99.64% (138 TP, 6 FP, 0 FN / 138 mentions)
   - **Sinhala**: Precision = 96.32%, Recall = 99.24%, F1 = **97.76%**, Token Accuracy = 99.70% (131 TP, 5 FP, 1 FN / 132 mentions)
   - **Kokborok**: Precision = 77.78%, Recall = 87.50%, F1 = **82.35%**, Token Accuracy = 99.55% (14 TP, 4 FP, 2 FN / 16 mentions)
   - *Kokborok 82.35% is strictly enforced; earlier 68.97% intermediate exploratory scores were not substituted.*

2. **Multilingual Joint Baseline**:
   - Konkani: 95.87% ($\Delta = +0.40$ pp)
   - Assamese: 97.86% ($\Delta = +0.01$ pp)
   - Sinhala: 97.04% ($\Delta = +0.72$ pp)
   - *No fictitious Kokborok joint baseline was created; Kokborok is excluded from Figure 5 and Table 6's baseline column.*

3. **Faithfulness Evaluation (`phase15_faithfulness_four_language_raw.json`)**:
   - Direct verification from the raw JSON artifact confirmed exact parity across Table 8, Figure 9, and Figure 10:
     - **Konkani Raw Attention**: $k=1$: +0.0363, $k=2$: +0.0412, $k=3$: +0.0316
     - **Konkani Grad $\times$ Attention**: $k=1$: +0.0190, $k=2$: +0.0250, $k=3$: +0.0127
     - **Assamese Raw Attention**: $k=1$: -0.0135, $k=2$: -0.0150, $k=3$: -0.0130
     - **Assamese Grad $\times$ Attention**: $k=1$: -0.0232, $k=2$: -0.0220, $k=3$: -0.0143
     - **Sinhala Raw Attention**: $k=1$: +0.0107, $k=2$: -0.0011, $k=3$: +0.0029
     - **Sinhala Grad $\times$ Attention**: $k=1$: +0.0058, $k=2$: -0.0080, $k=3$: -0.0051
     - **Kokborok Raw Attention**: $k=1$: -0.0132, $k=2$: -0.0198, $k=3$: -0.0381
     - **Kokborok Grad $\times$ Attention**: $k=1$: -0.0141, $k=2$: -0.0365, $k=3$: -0.0539
   - Sample sizes verified: Konkani = 100, Assamese = 100, Sinhala = 100, Kokborok = 17.

4. **Strict Gold-Test Set Protection**:
   - All explainability visualizations (Figure 6 heatmap, Figure 7 sparsity, Figure 8 rank) and faithfulness benchmarks (Figures 9 and 10) are strictly derived from **Gold-Dev**, keeping the Gold-Test set uncontaminated.

---

## 4. Figures Intentionally Omitted

In accordance with strict scientific integrity:
- **Models 1–3 Standalone Test Curves**: Standalone numerical test scores on the locked test sets were logged exclusively for the final composite model (Model 4). Speculative numerical curves for Models 1–3 were intentionally omitted.
- **Kokborok Joint Baseline**: A joint baseline checkpoint for Kokborok was not part of the initial joint training pool; no simulated or interpolated baseline was fabricated.
- **Unverified Attribution Methods**: Attributions such as Integrated Gradients or DeepLIFT were excluded because authoritative Cycle 2 / Phase 15 artifacts support Raw Attention and Gradient $\times$ Attention.
- **Synthetic/Schematic Heatmaps**: The illustrative example (*Dr. Himanta Biswa Sarma*) was removed and replaced with the verified Gold-Dev model inference output (*Christmas*, Kokborok Gold `B-MISC`).

---

## 5. Visual QA & Formatting Audit

- **Transparency & PDF Renderer SMask Resolution**: All 10 publication figures were re-generated and converted to solid 24-bit RGB mode (`facecolor='white'`, `transparent=False`), eliminating soft masks (`smask=0`). In certain Windows PDF viewers (e.g. Acrobat Reader, Edge, and preview engines), RGBA transparency masks in PDF XObjects caused Figures 6, 9, 10, and 11 to render invisible or blank. Pure 24-bit RGB embedding guarantees immediate, solid visibility across all PDF viewing engines and zoom levels.
- **Figures 6, 9, 10, 11 Verification**:
  - **Figure 6 (Page 6)**: Rendered immediately after Section 7.5 introductory paragraph; uses verified Gold-Dev prediction for Kokborok entity *Christmas* (`B-MISC`).
  - **Figure 9 (Page 8)**: Line graph across $k \in \{1, 2, 3\}$ for Raw and Gradient $\times$ Attention; placed immediately after Section 7.6 paragraph 3.
  - **Figure 10 (Page 9)**: Summary heatmap placed immediately after Table 8; values match Table 8 and Figure 9 exactly to 4 decimal places.
  - **Figure 11 (Page 10)**: Comparative audit of verified Gold training resources (entities and supervision density per 1,000 tokens); placed immediately after Section 7.7 paragraph 2.
- **Causal Statement Restraint**: Line 116 strictly reports the measured final-vs-baseline performance deltas (+0.40 pp in Konkani, +0.01 pp in Assamese, +0.72 pp in Sinhala) without implying that unfreezing itself was experimentally isolated as the cause.
- **Section Hierarchy**: Section counter initialized with `\setcounter{section}{6}`, correctly generating `7 Results and Discussion` (subsections 7.1 to 7.10) and `8 Conclusion and Future Scope` (subsections 8.1 and 8.2).
- **Float Queue Normalization**: Replaced `figure*` with single-column `figure` floats (`\begin{figure}[htbp]`), ensuring that all figures remain in strict sequential FIFO float order without being deferred or separated.
- **Table Dimensions**: Tables 5–9 were audited for column widths. `\resizebox` was eliminated to prevent TeX grouping conflicts with `sn-jnl.cls`, replacing it with calibrated font sizes (`\small`, `\footnotesize`, `\scriptsize`), custom `\tabcolsep`, and wrapping `p{...}` paragraph columns for Table 9.
- **Remaining Warnings**: 0 compilation errors, 0 undefined references.

---

## 6. Scientific-Precision Corrections Audit (Final Pass)

In accordance with strict scientific precision, all mechanistic and causal interpretations not directly isolated by an experiment were softened to cautious observational language:

1. **Claim #1 (Section 7.6 Faithfulness Causal Language)**:
   - *Previous:* "In recurrent sequence architectures, zeroing arbitrary context tokens can disrupt the bidirectional state propagation of the BiLSTM more severely than zeroing specific tokens identified by post-recurrent attention. These empirical results demonstrate that attention attribution scores cannot be assumed to function as faithful causal explanations of model decisions."
   - *Updated:* "One possible explanation is that, because the attention layer operates on recurrently contextualized representations, perturbing arbitrary context tokens may alter the BiLSTM-derived representations more broadly than perturbing tokens selected by the post-recurrent attribution mechanism. These results therefore indicate that attention attribution should not be assumed to provide faithful causal explanations without empirical validation."
2. **Claim #2 (Section 8.1 & Conclusion BiLSTM Interpretation)**:
   - *Previous:* "...reflecting the contextualizing effect of the preceding BiLSTM layers."
   - *Updated:* "...consistent with the contextualized representations produced before the attention layer."
   - *Section 8.1 Conclusion sentence:* "These results indicate that attention distributions should not automatically be interpreted as direct causal explanations without empirical validation."
3. **Claim #3 (Section 7.10 Supervision Sensitivity)**:
   - *Previous:* "...highlights that sequence taggers remain sensitive to the absolute volume of annotated entity instances, even when token-level accuracy remains above 99%."
   - *Updated:* "The lower Kokborok F1 coincides with substantially lower annotated entity density, indicating that supervision availability is an important factor for further investigation. The present experiments do not isolate data availability from other possible factors."
4. **Claim #4 (Table 9 Assamese Error Analysis)**:
   - *Previous:* "Attached case markers alter word boundaries, leading to boundary inclusion errors."
   - *Updated:* "Attached case markers may contribute to boundary-inclusion errors."
5. **Claim #5 (Table 9 Sinhala Error Analysis & Konkani)**:
   - *Previous:* "Inconsistent zero-width joiners alter character CNN representations."
   - *Updated:* "Inconsistent zero-width joiners may affect character-level representations of conjunct forms."
   - *Konkani in Table 9:* "Omission of title context may contribute to false negative classifications for person names."
6. **Claim #6 (Section 7.9 Literature / Attention Interpretation)**:
   - *Previous:* "...attention weights operate primarily as internal routing mechanisms rather than faithful causal explanations."
   - *Updated:* "The present attribution and erasure results support interpreting attention weights cautiously as internal information-routing signals rather than automatically treating them as faithful causal explanations."
7. **Line 194 (Attribution Audit Phrasing)**:
   - *Previous:* "The attribution audit confirms that high entity-level NER performance is not driven by the concentration of attention on the target token itself."
   - *Updated:* "The attribution audit demonstrates that high entity-level NER performance is not accompanied by the concentration of attention on the target token itself."
8. **Confirmation of Zero Numerical or Experimental Changes**:
   - Zero verified numbers were changed. All final Gold-Test metrics (Konkani 96.27%, Assamese 97.87%, Sinhala 97.76%, Kokborok 82.35%), baseline deltas (+0.40 pp, +0.01 pp, +0.72 pp), Phase 15 faithfulness scores, and data audit counts remain 100% identical.
   - Zero experiments, ablations, or synthetic claims were added.
   - All 10 figures (Figures 3–12) remain intact, sharp, and physically visible in the compiled PDF.
