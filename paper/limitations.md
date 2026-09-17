# Scientific Limitations & Failure Mode Discussion

While the proposed system achieves the primary research target (>90% entity-level F1 across Konkani, Assamese, and Sinhala) strictly without external pretrained models or tokenizers, a rigorous research writeup requires an honest exposition of the architectural, data-centric, and methodological limitations uncovered across Phases 0–15.

---

## 1. Raw-Corpus Volume Asymmetry and Vocabulary Coverage

A primary vulnerability of learning language representations purely from scratch is dependency on raw corpus volume. In Phase 1 and subsequent data audits, the volume of clean, deduplicated raw text varied noticeably across the three target languages:
- **Sinhala**: 66,954 sentences (1,048,154 tokens, 76,826 vocabulary types)
- **Assamese**: 55,002 sentences (977,912 tokens, 69,564 vocabulary types)
- **Konkani**: 51,488 sentences (845,959 tokens, 58,162 vocabulary types)

Although Konkani reached a 96.27% test F1, its smaller lexical base caused higher sensitivity to Out-of-Vocabulary (OOV) entities (In-Vocab F1: 97.09% vs. OOV F1: 91.43%). In our Phase 14 error analysis, **88.89% of all Konkani test errors were False Negatives (missed entities)**. These errors predominantly comprised inflected multi-word organizational names and rural place names (e.g., `कोंकणी अकादमी`, `नावेली`) that occurred zero or one time in the raw corpus. While multilingual pooling in Phase 8 provided significant cross-lingual regularization (+2.52% net gain for Konkani), multilingual transfer cannot entirely substitute for native lexical frequency in low-resource orthographies.

---

## 2. Weak Supervision Noise and Silver-Label Systematic Bias

To bridge the sample-efficiency gap without external pretrained teachers, Phase 4 synthesized 7,200 silver sentences per language (12x gold-train volume) via gazetteers, toponymic rules, and Wikipedia distant supervision. Although empirical precision on gold-dev was calibrated to high thresholds:
- **Konkani**: 100.0% estimated silver precision
- **Assamese**: 96.0% estimated silver precision (4.0% noise rate)
- **Sinhala**: 96.0% estimated silver precision (4.0% noise rate)

weakly supervised labels introduce structural inductive biases:
1. **Homograph False Positives**: In Assamese and Sinhala, rule-based matching occasionally mislabeled generic common nouns that share surface orthography with historical figures or places.
2. **Boundary Truncation**: Distant supervision on Wikipedia anchor text frequently tagged base head nouns while omitting trailing case affixes or adpositions.
3. **Annotation Incompleteness in Gold Test**: In Assamese and Sinhala, 100% and 83.33% of test discrepancies respectively were **Spurious Entities (False Positives)**. Qualitative spot-checks revealed that several of these "errors" (e.g., `ভাৰতীয় কংগ্ৰেছ` [Indian Congress], `প্ৰফুল্ল কুমাৰ মহন্ত` [Prafulla Kumar Mahanta]) were in fact valid named entities that the gold annotators had inadvertently left untagged (`O`). The model generalized beyond the omissions of the gold training seed, but was penalized by strict exact-match evaluation against the locked benchmark.

---

## 3. Analysis of Target Fulfillment and Residual Failure Modes

The primary quantitative objective of this investigation was to establish whether a from-scratch neural architecture could achieve **>90% strict entity-level F1** across three distinct South Asian scripts. 

All three languages met this bar on the locked test benchmark:
- **Konkani**: 96.27% Entity F1 (**MET**, +6.27 pp margin)
- **Assamese**: 97.87% Entity F1 (**MET**, +7.87 pp margin)
- **Sinhala**: 97.76% Entity F1 (**MET**, +7.76 pp margin)

However, analysis of residual errors reveals clear linguistic limitations:
1. **Absence of Type Confusion**: Across all three test sets, the model produced **zero Type Confusion errors** (classifying a location as a person or vice-versa). The semantic boundaries between entity categories were sharply learned.
2. **Absence of Boundary Errors**: The model produced **zero Partial Boundary Overlap errors**, validating the Linear-Chain CRF's sequence transition matrix in enforcing strict BIO coherence.
3. **Long-Tail Recall Gap**: The remaining deficiency is almost purely recall-oriented on tail entities. For extremely rare compound nouns or hapax legomena in low-resource scripts, the combination of Char-CNN and BiLSTM still defaults to the high-prior background class (`O`).

---

## 4. Methodological Limitations of Attention-Based Explainability

Phase 13 evaluated the interpretability of internal model representations using a quantitative token-erasure benchmark. Our findings confirm and contextualize critical debates in NLP interpretability research (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019; Serrano & Smith, 2019):

### 4.1 Raw Attention as an Unfaithful Explanation
Raw self-attention weights represent forward contextual routing rather than retrospective decision causality. In our head-wise analysis:
- Multiple attention heads exhibited high entropy (mean entropy 2.18–2.58), acting as broadcast channels that distributed attention across punctuation tokens (commas, full stops) and high-frequency coordinating conjunctions (e.g., `আৰু`, `आनी`).
- In the quantitative erasure test, raw attention produced a negative faithfulness gap (-0.2014) because erasing high-attention delimiters does not causally alter the entity emission score.
- **Scientific Conclusion**: Raw attention weights must not be presented to end users as faithful explanations of model decisions.

### 4.2 Gradient × Attention Grounding and Recurrent Mixing
Scaling attention weights by backward decision gradients ($\nabla_{\alpha} \text{score}$) successfully suppresses uninformative syntactic delimiters and isolates tokens that actively influence the predicted tag. However, the hybrid BiLSTM-Attention architecture presents an inherent explanatory limitation:
- The bidirectional LSTM mixes contextual representations *before* vectors enter the multi-head self-attention layer. 
- In multi-token entity spans (e.g., `B-PER` followed by `I-PER`), the dependency between adjacent tokens is captured primarily by recurrent hidden state transitions rather than cross-attention weights.
- When an erasure protocol masks context tokens, the BiLSTM's bidirectional recurrence often compensates for the missing token, resulting in modest confidence drops ($\Delta \text{Conf} \approx 0.001$).
- Conversely, when random control erasure inadvertently removes the query entity itself or its adjacent multi-token span partner, the tag probability plummets. 

While Gradient × Attention provides mathematically grounded saliency over attention routing, full causal explainability in hybrid recurrent-attention architectures remains fundamentally constrained by the opaque information mixing of recurrent gates.
