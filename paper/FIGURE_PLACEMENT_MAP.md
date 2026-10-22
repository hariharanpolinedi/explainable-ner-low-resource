# Figure Placement Map

This document records the exact insertion position, LaTeX label, source file, narrative context, and compiled PDF page for every publication figure in **Section 7 (Results and Discussion)** and **Section 8 (Conclusion and Future Scope)** of the manuscript:

> **Manuscript Title:** *Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution*  
> **Compiled Document:** `Section_7_8_Results_Conclusion_Springer_Final.pdf` (14 pages)  
> **LaTeX Source:** `Section_7_8_Results_Conclusion_Springer_Final.tex` (synchronized in `paper/`)

---

## Complete Figure Mapping (Figures 3 – 12)

*Note: Figures 1 and 2 and Tables 1–4 belong to Section 6 (Methodology). Section 7 resumes with Section counter 7, Table counter 5, and Figure counter 3.*

### Figure 3 — Entity-Level F1 by Language
- **Title:** Entity-Level F1 Performance on Official Gold-Test Sets
- **LaTeX Label:** `fig:entity_f1_bar`
- **Image File:** `figures/fig_a_entity_f1.png` (vector PDF: `figures/fig_a_entity_f1.pdf`)
- **Inserted After:** Section 7.2 (*Overall NER Performance*), paragraph 2 (following Table 5)
- **PDF Page:** **Page 3** (top)
- **Discussion Context:** Section 7.2 paragraph 3 analyzes exact-span matching performance on Konkani (96.27%), Assamese (97.87%), Sinhala (97.76%), and Kokborok (82.35%), and contrasts with token accuracy (>99.4%).
- **Verified Data Source:** Official locked Gold-Test evaluation logs.

---

### Figure 4 — Precision, Recall, and Entity F1 Grouped Comparison
- **Title:** Grouped Comparison of Precision, Recall, and Entity-Level F1
- **LaTeX Label:** `fig:precision_recall_f1`
- **Image File:** `figures/fig_b_precision_recall_f1.png` (vector PDF: `figures/fig_b_precision_recall_f1.pdf`)
- **Inserted After:** Section 7.2 (*Overall NER Performance*), paragraph 3
- **PDF Page:** **Page 3** (bottom)
- **Discussion Context:** Section 7.2 paragraph 4 evaluates language-specific error profiles: Assamese (100% recall, 95.83% precision), Konkani (99.15% precision, 93.55% recall), Sinhala (96.32% precision, 99.24% recall), and Kokborok (77.78% precision, 87.50% recall).
- **Verified Data Source:** Official locked Gold-Test evaluation logs.

---

### Figure 5 — Final Model vs. Multilingual Joint Baseline
- **Title:** Multilingual Joint Baseline vs. Final Model Comparison
- **LaTeX Label:** `fig:final_vs_joint_baseline`
- **Image File:** `figures/fig_c_final_vs_joint_baseline.png` (vector PDF: `figures/fig_c_final_vs_joint_baseline.pdf`)
- **Inserted After:** Section 7.3 (*Language-Wise Performance*), paragraph 2 (following Table 6)
- **PDF Page:** **Page 4** (center)
- **Discussion Context:** Section 7.3 paragraph 3 evaluates the measured deltas: +0.40 pp in Konkani (95.87% to 96.27%), +0.01 pp in Assamese (97.86% to 97.87%), and +0.72 pp in Sinhala (97.04% to 97.76%). Kokborok is strictly excluded because no joint baseline checkpoint was logged.
- **Verified Data Source:** Multilingual joint baseline evaluation logs.

---

### Figure 6 — Attribution Heatmap on Actual Gold-Dev Prediction
- **Title:** Verified Gold-Dev Attribution Heatmap (Kokborok *Christmas*, Gold `B-MISC`)
- **LaTeX Label:** `fig:attribution_heatmap`
- **Image File:** `figures/fig_d_attribution_heatmap.png` (vector PDF: `figures/fig_d_attribution_heatmap.pdf`)
- **Inserted After:** Section 7.5 (*Explainability and Attention Attribution Analysis*), paragraph 1 (immediately introducing the heatmap)
- **PDF Page:** **Page 6** (top)
- **Discussion Context:** Section 7.5 paragraph 2 analyzes normalized attribution on query token *Christmas* (position 3). Demonstrates diffuse attribution mass across surrounding context tokens (*Lunglei*: 0.1463 Raw / 0.2030 Grad $\times$ Attn, *song*: 0.2035 / 0.1818, *tongo*: 0.2058 / 0.1168) rather than direct isolation of the entity.
- **Verified Data Source:** Real model inference telemetry on Gold-Dev.

---

### Figure 7 — Target-Token Attribution Sparsity Distribution
- **Title:** Target-Token Near-Zero Attribution Mass ($\le 0.001$) Proportion
- **LaTeX Label:** `fig:target_token_near_zero`
- **Image File:** `figures/fig_e_target_token_near_zero.png` (vector PDF: `figures/fig_e_target_token_near_zero.pdf`)
- **Inserted After:** Section 7.5 (*Explainability and Attention Attribution Analysis*), paragraph 4 (following Table 7)
- **PDF Page:** **Page 7** (top)
- **Discussion Context:** Section 7.5 paragraph 5 shows that over 92% of target entity tokens receive attribution mass $\le 0.001$ under both methods across all three evaluated languages (reaching 98.56% in Assamese and 99.24% in Sinhala).
- **Verified Data Source:** Internal target-token attribution audit across Gold-Dev entities ($N=125$ Konkani, $N=139$ Assamese, $N=131$ Sinhala).

---

### Figure 8 — Target-Token Attribution Mean Rank Distribution
- **Title:** Mean Rank of Target Entity Tokens in Contextual Attribution Ordering
- **LaTeX Label:** `fig:target_token_rank`
- **Image File:** `figures/fig_f_target_token_rank.png` (vector PDF: `figures/fig_f_target_token_rank.pdf`)
- **Inserted After:** Section 7.5 (*Explainability and Attention Attribution Analysis*), paragraph 5
- **PDF Page:** **Page 7** (center)
- **Discussion Context:** Section 7.5 paragraph 6 explains why target tokens rank between 16.94 and 20.63 in sentence importance, connecting this to the contextualizing buffer of the preceding two-layer BiLSTM encoder.
- **Verified Data Source:** Target-token attribution audit logs.

---

### Figure 9 — Faithfulness Erasure Gap Line Graph
- **Title:** Faithfulness Gap Trajectories across Context Erasure Sizes $k \in \{1, 2, 3\}$
- **LaTeX Label:** `fig:faithfulness_line`
- **Image File:** `figures/fig_g_faithfulness_line.png` (vector PDF: `figures/fig_g_faithfulness_line.pdf`)
- **Inserted After:** Section 7.6 (*Faithfulness Evaluation*), paragraph 3 (following Equations 1 & 2)
- **PDF Page:** **Page 8** (bottom)
- **Discussion Context:** Section 7.6 paragraph 5 evaluates faithfulness dynamics: consistently positive gaps in Konkani (+0.0127 to +0.0412), near zero in Sinhala, and negative gaps in Assamese (-0.0130 to -0.0232) and Kokborok (-0.0132 to -0.0539).
- **Verified Data Source:** `reports/experiment_logs/phase15_faithfulness_four_language_raw.json`.

---

### Figure 10 — Compact Faithfulness Gap Heatmap
- **Title:** Compact Heatmap Comparison of Faithfulness Gaps
- **LaTeX Label:** `fig:faithfulness_heatmap`
- **Image File:** `figures/fig_h_faithfulness_heatmap.png` (vector PDF: `figures/fig_h_faithfulness_heatmap.pdf`)
- **Inserted After:** Section 7.6 (*Faithfulness Evaluation*), paragraph 4 (following Table 8)
- **PDF Page:** **Page 9** (center)
- **Discussion Context:** Section 7.6 paragraph 5 connects the visual color gradients with Table 8, explaining that negative deletion gaps occur when arbitrary context token erasure damages recurrent BiLSTM state propagation more severely than removing tokens identified by post-recurrent attention.
- **Verified Data Source:** `reports/experiment_logs/phase15_faithfulness_four_language_raw.json`.

---

### Figure 11 — Gold Training Resource and Entity Density Audit
- **Title:** Comparative Audit of Gold Training Data Resources and Supervision Density
- **LaTeX Label:** `fig:data_density`
- **Image File:** `figures/fig_i_data_density.png` (vector PDF: `figures/fig_i_data_density.pdf`)
- **Inserted After:** Section 7.7 (*Kokborok Resource-Limited Analysis*), paragraph 2
- **PDF Page:** **Page 10** (center)
- **Discussion Context:** Section 7.7 paragraph 3 contrasts Kokborok's 30 training entities (4.5 per 1,000 tokens) against Konkani (590 entities, 43.0/k tokens), Assamese (690 entities, 47.1/k tokens), and Sinhala (519 entities, 31.4/k tokens), demonstrating that matched sentence counts do not imply equal supervision.
- **Verified Data Source:** `reports/kokborok_exp9_data_statistics.json`, `reports/kokborok_exp10_corpus_statistics.json`.

---

### Figure 12 — Entity Confusion Matrices across Four Languages
- **Title:** Locked Gold-Test Entity Prediction Confusion Matrices
- **LaTeX Label:** `fig:confusion_matrices`
- **Image File:** `figures/fig_k_confusion_matrices.png` (vector PDF: `figures/fig_k_confusion_matrices.pdf`)
- **Inserted After:** Section 7.8 (*Error Analysis*), paragraph 2 (following Table 9)
- **PDF Page:** **Page 12** (top)
- **Discussion Context:** Section 7.8 paragraph 3 analyzes the strong diagonal nature of Konkani, Assamese, and Sinhala, alongside the isolated Kokborok `Tripura board` (Gold `ORG` truncated to `Tripura` `LOC`) boundary failure.
- **Verified Data Source:** `reports/kokborok_v4_test_eval_results.json`, `reports/kokborok_exp9_error_matrix.csv`.

---

## Tables Summary

| Table | Title | Label | Inserted Location | PDF Page |
|---|---|---|---|---|
| **Table 5** | Overall Performance of the Final Proposed NER Model on the Official Gold-Test Set | `tab:overall_perf` | Section 7.2 | **Page 2** |
| **Table 6** | Language-Wise Performance on the Official Gold-Test Set and Joint Baseline Comparison | `tab:lang_breakdown` | Section 7.3 | **Page 4** |
| **Table 7** | Target-Token Attribution Audit across Gold-Development Entity Predictions | `tab:target_token_audit` | Section 7.5 | **Page 6** |
| **Table 8** | Faithfulness Evaluation under the Primary Token-Erasure Protocol | `tab:faithfulness_results` | Section 7.6 | **Page 9** |
| **Table 9** | Documented Error Patterns across Test Evaluations | `tab:error_analysis` | Section 7.8 | **Page 11** |
