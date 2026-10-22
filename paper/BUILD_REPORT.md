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
| **PDF Page Count** | **14 Pages** |
| **PDF File Size** | 1.74 MiB (high-resolution embedded vector/raster graphics) |
| **Section Hierarchy** | **7 Results and Discussion** (7.1 – 7.10) & **8 Conclusion and Future Scope** (8.1 – 8.2) |
| **Total Publication Figures Generated** | **10 figures** (dual format: 300 DPI PNG + vector PDF) |
| **Total Publication Figures Inserted** | **10 figures** (Figures 3 through 12, synchronized with Section 6) |
| **Total Tables Formatted** | **5 tables** (Tables 5 through 9, synchronized with Section 6) |
| **Undefined References / Labels** | **0** (all citations, figure labels, and table labels cleanly resolved) |
| **Fatal Errors** | **0** |

---

## 2. Publication Figures Generated & Inserted

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

- **Section Hierarchy**: Section counter initialized with `\setcounter{section}{6}`, correctly generating `7 Results and Discussion` (subsections 7.1 to 7.10) and `8 Conclusion and Future Scope` (subsections 8.1 and 8.2).
- **Float Queue Normalization**: Replaced `figure*` with single-column `figure` floats (`\begin{figure}[htbp]`), ensuring that all figures remain in strict sequential FIFO float order without being deferred or separated.
- **Figure 6 Positioning**: Figure 6 (Attribution Heatmap) now appears at the top of Section 7.5 right after its introductory paragraph, before Table 7.
- **Figure 9 & 10 Separation**: Figures 9 (line graph) and 10 (heatmap) each possess distinct introductory paragraphs and follow-up discussion.
- **Table Dimensions**: Tables 5–9 were audited for column widths. `\resizebox` was eliminated to prevent TeX grouping conflicts with `sn-jnl.cls`, replacing it with calibrated font sizes (`\small`, `\footnotesize`, `\scriptsize`), custom `\tabcolsep`, and wrapping `p{...}` paragraph columns for Table 9.
- **Remaining Warnings**: 0 compilation errors, 0 undefined references.
