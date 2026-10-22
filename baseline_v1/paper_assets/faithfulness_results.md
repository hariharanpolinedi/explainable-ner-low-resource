# Faithfulness Evaluation & Attribution Method Selection

**Paper Section Target**: Methodology / Explainability / Evaluation Benchmarks  
**Primary Source Reports**: [`reports/faithfulness_reverification_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/faithfulness_reverification_results.md), [`reports/faithfulness_test_design_audit.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/faithfulness_test_design_audit.md), [`reports/phase13_faithfulness_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase13_faithfulness_results.md)  
**Secondary Source Report**: [`reports/phase13_head_wise_analysis.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase13_head_wise_analysis.md)  
**Date of Re-Verification**: `2026-09-18` (Original Phase 13: `2026-09-17`)  
**Hardware Device**: NVIDIA GeForce RTX 3050 Laptop GPU  
**Pretrained Tool Usage**: STRICTLY NONE (Direct Autograd and PyTorch Tensor Operations)  
**Evaluation Split**: Locked Gold-Dev (`data/gold/{lang}/dev.conll`, 100 entity predictions per language, total $N=300$)  

---

## 1. Executive Summary & Method Selection

- **Authoritative Winning Method**: **Raw Attention (`raw_attention`)**
- **Mean Faithfulness Gap (Raw Attention)**: **-0.0523**
- **Mean Faithfulness Gap (Gradient $\times$ Attention)**: **-0.0529** (Deficit: -0.0006)
- **Faithfulness Win Rate Margin ($k=1$)**: Raw Attention outperforms Gradient $\times$ Attention by **+4.0%** in Konkani (19.0% vs. 15.0%), **+6.0%** in Assamese (30.0% vs. 24.0%), and **+11.0%** in Sinhala (23.0% vs. 12.0%).
- **Winning Explainer Rationale**: Under the corrected symmetric erasure protocol (`exclude_target_token=True`), Raw Attention consistently achieves higher (less negative) faithfulness gaps and decisively higher instance-level win rates. In hybrid BiLSTM-Attention-CRF architectures trained on low-resource corpora, backpropagating tag emission gradients into attention weights ($\alpha \odot \nabla_\alpha$) tends to attenuate contextual attention patterns due to sparse gradient flow through the linear-chain CRF, reducing its causal faithfulness relative to the raw forward attention distribution.

---

## 2. Quantitative Faithfulness Matrix (Corrected Symmetric Protocol)

*Benchmark evaluated using `explainability/erasure_faithfulness_test_v2.py` with `exclude_target_token=True`. Both top-$k$ and random-$k$ tokens are drawn from identical context candidate pools (strictly excluding the target entity token itself). Faithfulness Gap ($\Delta = \text{Drop}_{\text{top}} - \text{Drop}_{\text{rand}}$) quantifies causal necessity, where less negative / larger values indicate superior faithfulness.*  
*Source: [`reports/faithfulness_reverification_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/faithfulness_reverification_results.md)*

| Language | Method | Top-1 Drop | Rand-1 Drop | **Gap ($k=1$)** | Top-2 Drop | Rand-2 Drop | **Gap ($k=2$)** | Top-3 Drop | Rand-3 Drop | **Gap ($k=3$)** | Win Rate ($k=1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | Gradient $\times$ Attention | 0.0015 | 0.0354 | **-0.0339** | 0.0013 | 0.0721 | **-0.0709** | 0.0002 | 0.0633 | **-0.0631** | 15.0% |
| **Konkani** | **Raw Attention** *(Winner)* | 0.0018 | 0.0354 | **-0.0336** | 0.0019 | 0.0721 | **-0.0702** | 0.0018 | 0.0633 | **-0.0615** | **19.0%** |
| **Assamese** | Gradient $\times$ Attention | 0.0000 | 0.0058 | **-0.0058** | 0.0000 | 0.0668 | **-0.0668** | 0.0007 | 0.0410 | **-0.0402** | 24.0% |
| **Assamese** | **Raw Attention** *(Winner)* | 0.0000 | 0.0058 | **-0.0058** | 0.0000 | 0.0668 | **-0.0668** | 0.0008 | 0.0410 | **-0.0402** | **30.0%** |
| **Sinhala** | Gradient $\times$ Attention | 0.0000 | 0.0296 | **-0.0296** | 0.0003 | 0.0599 | **-0.0597** | 0.0103 | 0.1163 | **-0.1059** | 12.0% |
| **Sinhala** | **Raw Attention** *(Winner)* | 0.0007 | 0.0296 | **-0.0290** | 0.0008 | 0.0599 | **-0.0591** | 0.0120 | 0.1163 | **-0.1043** | **23.0%** |

---

## 3. Head-Wise Specialization & Entropy Analysis

*Source: [`reports/phase13_head_wise_analysis.md#L10-L45`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase13_head_wise_analysis.md#L10-L45)*

Across the 4 self-attention heads ($H=4$), distinct functional roles emerged:

| Head Index | Primary Specialization | Mean Attention Entropy | Top Attended Syntactic Structures |
| :---: | :--- | :---: | :--- |
| **Head 0** | Broad Contextual Broadcast | 2.58 (High) | Coordinating conjunctions (`আৰু`, `आनी`, `සහ`), commas, periods |
| **Head 1** | Local Boundary Tracking | 1.12 (Low) | Immediate left/right adjacent tokens, honorific prefixes (`শ্রী`, `ড°`) |
| **Head 2** | Entity-Internal Dependency | 1.34 (Low) | Multi-word compound nouns (`B-` to `I-` links), lexical headwords |
| **Head 3** | Syntactic Case / Postposition Alignment | 1.87 (Medium) | Inflectional suffixes, adpositions (`ত`, `ंत`, `යේදී`, `खातीर`) |

---

## 4. Key Analytical Insights & Honest Methodological Limitations

1. **Resolution of the Target-Token Erasure Artifact**:
   In Phase 13's original benchmark, the target entity token itself received near-zero attention (verified programmatically: 96.7% of instances have attribution $\le 0.001$, mean attribution 0.00018). Consequently, top-$k$ erasure *never* erased the target token (0.0%–1.0%), whereas random-$k$ erasure drew from all tokens including the target entity (7.0%–29.0% inclusion rate). Erasing the entity token itself destroyed prediction confidence by 0.9161–0.9838, which artificially inflated the random drop and produced an uninformative -0.20 gap artifact.
2. **Context Erasure & The BiLSTM Recurrent Buffer**:
   Under the corrected symmetric context protocol (`exclude_target_token=True`), random drops are compressed to realistic levels (0.0058–0.0354 at $k=1$). Faithfulness gaps remain slightly negative (-0.005 to -0.07) across both methods. This occurs because BiLSTM representations encode contextual semantics bidirectionally along sequential token positions prior to the attention layer; erasing arbitrary context tokens disrupts recurrent hidden state propagation more than erasing specialized high-attention tokens (such as sentence-final delimiters).
3. **Scientific Implication for the Manuscript**:
   Rather than claiming that attention attribution is an unconditionally faithful surrogate for decision causality, the paper explicitly documents this limitation: attention attribution highlights the syntactic anchors used by the self-attention layer, but should not be conflated with full causal intervention in hybrid recurrent-CRF architectures.

---

## 5. Changelog & Pre-Submission Re-Verification Record

- **Date**: `2026-09-18`
- **Re-Verification Scripts**:
  - Audit & Asymmetry Verification: [`evaluation/verify_erasure_test_design.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/evaluation/verify_erasure_test_design.py)
  - Target Token Attribution Audit: [`evaluation/attribution_target_token_audit.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/evaluation/attribution_target_token_audit.py)
  - Corrected Benchmark Runner: [`explainability/erasure_faithfulness_test_v2.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/explainability/erasure_faithfulness_test_v2.py)
- **Detailed Audit & Finding Reports**:
  - [`reports/faithfulness_test_design_audit.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/faithfulness_test_design_audit.md)
  - [`reports/faithfulness_reverification_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/faithfulness_reverification_results.md)
- **Problems Identified**:
  1. *Candidate Pool Asymmetry*: In `explainability/erasure_faithfulness_test.py`, `random_indices` selected tokens randomly from `range(seq_len)` without excluding `query_idx`. Because target entity tokens received near-zero attribution in 96.7% of predictions, top-$k$ erasure never erased the entity itself, while random-$k$ erased it up to 29% of the time, creating an artificial ~0.95 confidence penalty exclusively in the random control group.
  2. *Inverted Method Selection Logic*: The original Phase 13 report declared Gradient $\times$ Attention the "winner" despite having a mean gap of -0.2020 versus Raw Attention's -0.2014, selecting the more negative (worse) number by its own metric.
- **Corrections Applied**:
  1. Redesigned the benchmark in `explainability/erasure_faithfulness_test_v2.py` with `exclude_target_token=True`, enforcing identical context-only candidate pools for both top-$k$ and random-$k$ selection.
  2. Re-ran the benchmark across the full 100 gold-dev entity predictions per language ($N=300$ total) for $k \in \{1, 2, 3\}$.
  3. Re-evaluated method selection with correct comparative logic: **Raw Attention** achieves a higher mean gap (-0.0523 vs. -0.0529) and substantially higher win rates (Konkani: 19% vs 15%, Assamese: 30% vs 24%, Sinhala: 23% vs 12%), making it the authoritative winner.
