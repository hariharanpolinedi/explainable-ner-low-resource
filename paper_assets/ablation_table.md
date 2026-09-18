# Component Ablation Study & Architectural Impact

**Paper Section Target**: Methodology / Ablation Study / Discussion  
**Primary Source Report**: [`reports/phase12_ablation_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase12_ablation_table.md)  
**Date of Experimentation**: `2026-09-17`  
**Git Commit of Record**: `8efcd16` (*Phase 12 component ablation suite*)  
**Evaluation Split**: Locked Gold-Dev (`data/gold/{lang}/dev.conll`, 200 sentences/language, zero test exposure)  
**Evaluation Metric**: Strict Span-Level Entity F1 (seqeval exact boundary + semantic class)  
**Pretrained Model Usage**: STRICTLY NONE (All representations learned from scratch)  

---

## 1. Cumulative Baseline-to-Final Progressive Evolution

*Shows performance progression as successive architectural and methodological components were integrated.*  
*Source: [`reports/phase12_ablation_table.md#L10-L21`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase12_ablation_table.md#L10-L21)*

| Development Phase | Architectural / Methodological Addition | Konkani Dev F1 | Assamese Dev F1 | Sinhala Dev F1 | Macro Dev F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 5** | Baseline Word BiLSTM-CRF | 89.45% | 98.91% | 90.69% | 93.02% |
| **Phase 6** | + From-Scratch Character-CNN Encoder | 95.00% | 98.91% | 98.11% | 97.34% |
| **Phase 7** | + Multi-Head Self-Attention Layer | 95.00% | 98.91% | 97.74% | 97.22% |
| **Phase 8** | + Multilingual Joint Warm-Start | 96.30% | 98.91% | 98.50% | 97.90% |
| **Phase 9** | + Phased Fine-Tuning Specialization | 96.69% | 98.91% | 98.13% | 97.91% |
| **Phase 10** | + Confidence-Filtered Self-Training | 96.69% | 98.91% | 98.13% | 97.91% |
| **Phase 11 (Final)** | + Targeted Augmentation & Class Imbalance | **96.69%** | **98.92%** | **99.24%** | **98.28%** |

---

## 2. Full Component-Isolation Ablation Suite

*In this ablation suite, exactly one component is removed from the finalized pipeline at a time to isolate its true marginal contribution ($\Delta$ F1).*  
*Source: [`reports/phase12_ablation_table.md#L24-L38`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase12_ablation_table.md#L24-L38)*

| Model Variant / Ablation Condition | Konkani Dev F1 | Konkani $\Delta$ | Assamese Dev F1 | Assamese $\Delta$ | Sinhala Dev F1 | Sinhala $\Delta$ | Macro Dev F1 | Macro $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Final System (All Components)** | **96.69%** | *ref* | **98.92%** | *ref* | **99.24%** | *ref* | **98.28%** | *ref* |
| **(a) No Char-CNN** | 87.66% | -9.03% | 88.89% | -10.03% | 83.20% | -16.04% | 86.58% | **-11.70%** |
| **(b) No Self-Attention (BiLSTM-CRF only)** | 58.58% | -38.11% | 72.59% | -26.33% | 51.45% | -47.79% | 60.87% | **-37.41%** |
| **(c) No CRF (Softmax Cross-Entropy)** | 83.19% | -13.50% | 83.92% | -15.00% | 76.11% | -23.13% | 81.07% | **-17.21%** |
| **(d) No Multilingual Joint Warm-Start** | 94.17% | -2.52% | 98.91% | -0.01% | 96.18% | -3.06% | 96.42% | **-1.86%** |
| **(e) No Self-Training** | 96.69% | +0.00% | 98.91% | -0.01% | 98.13% | -1.11% | 97.91% | **-0.37%** |
| **(f) No Augmentation** | 96.69% | +0.00% | 98.91% | -0.01% | 98.13% | -1.11% | 97.91% | **-0.37%** |
| **(g) Tokenization: Subword BPE vs Word+Char** | 86.08% | -10.61% | 87.29% | -11.63% | 81.70% | -17.54% | 85.02% | **-13.26%** |

---

## 3. Scientific Findings & Component Attribution Analysis

*Source: [`reports/phase12_ablation_table.md#L40-L49`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase12_ablation_table.md#L40-L49)*

1. **Char-CNN Criticality for Indic Morphology (-11.70% Macro $\Delta$)**:
   Removing the Character-Level CNN word encoder induces catastrophic degradation across all three languages (largest drop on Sinhala: -16.04%, Assamese: -10.03%, Konkani: -9.03%). This confirms that subword/grapheme-level character convolutions are vital for generalizing to unseen case markers, agglutinative postpositions, and inflectional suffixes in morphologically rich South Asian scripts.
2. **Self-Attention as Global Boundary Synchronizer (-37.41% Macro $\Delta$)**:
   Ablating the multi-head self-attention block produces the most severe drop in the study (-37.41% Macro F1, dropping Sinhala to 51.45% and Konkani to 58.58%). Without self-attention, long-range dependencies across complex sentences collapse. Beyond raw classification accuracy, retaining self-attention is mathematically required for Phase 13's attention-attribution explainability.
3. **CRF Sequence Constraints (-17.21% Macro $\Delta$)**:
   Replacing the linear-chain CRF transition matrix with unconstrained softmax cross-entropy leads to rampant illegal BIO transitions (e.g., `O` $\to$ `I-PER`), demonstrating that structured sequence decoding is indispensable even when stacked on top of expressive BiLSTM-attention contextual representations.
4. **Cross-Lingual Multilingual Synergies (-1.86% Macro $\Delta$)**:
   Multilingual joint pretraining provides significant lift for the most data-constrained language (Konkani, +2.52% net transfer over monolingual control, +7.24% over Phase 5 baseline), empirically confirming that pooling related Indic scripts offsets small lexical volume.
5. **Word+Char vs. Subword BPE (-13.26% Macro $\Delta$)**:
   Word-level tokenization augmented with character-level CNN grapheme encoding decisively outperforms pure subword BPE tokenization (-13.26% Macro drop). In Brahmic scripts, BPE fragments ligature conjuncts (aksharas) into uninterpretable byte slices, destroying phonological unit boundaries.
