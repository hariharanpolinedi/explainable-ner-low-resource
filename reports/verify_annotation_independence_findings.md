# Verification Findings: Annotation Independence and Circularity in `long_multientity_eval.conll`

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Scope**: Code and process reconstruction of annotation procedures reported in `reports/patch_v2_gold_expansion_agreement.md`.

---

## 1. Executive Summary

- **Finding**: **CIRCULAR AND NOT INDEPENDENT**.
- **Human Involvement**: **0%**. There were no human annotators. "Annotator A" and "Annotator B" were programmatic Python routines.
- **Konkani Kappa = 1.0000**: This was not genuine consensus between two independent experts. It was a mathematical identity caused by zero triggers of Annotator B's synthetic perturbation rules on the 20-sentence overlap sample.
- **Post-Agreement Label Injection**: The 161 ORGs in Konkani, 65 ORGs in Assamese, and 173 ORGs in Sinhala were injected into the file **after** the agreement metric was computed, meaning the rule-based ORG labels were never evaluated by Annotator B at all.

---

## 2. Process Reconstruction: What Were "Annotator A" and "Annotator B"?

In [annotation/expand_long_multientity_set.py](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/expand_long_multientity_set.py) lines 116-117:
```python
tags_a = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
tags_b = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_b")
```

Looking at [annotation/annotate_gold.py](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/annotate_gold.py) lines 82-179:
1. `variant="annotator_a"` performs exact substring dictionary lookup from gazetteers (`data/gazetteers/{lang}/*.txt`), assigns BIO tags in priority `ORG > MISC > PER > LOC`, and strips honorifics/punctuation.
2. `variant="annotator_b"` runs the exact same code, but adds four hardcoded synthetic boundary perturbations:
   - *Ambiguity 1*: If `PER` has preceding honorific, include honorific in `PER` span.
   - *Ambiguity 2*: If `LOC` ends with generic geographic noun (e.g. `न्हंय`, `নদী`, `ගඟ`), drop trailing token.
   - *Ambiguity 3*: If `ORG` begins with location prefix (e.g. `गोवा`, `गोंय`, `গুৱাহাটী`, `অসম`, `කොළඹ`), split first token as `B-LOC`.
   - *Ambiguity 4*: If `MISC` ends with award term, tag first token as `B-ORG`.

---

## 3. Why Konkani Produced $\kappa = 1.0000$ and Span F1 $= 1.0000$

- In `expand_long_multientity_set.py`, Cohen's Kappa was evaluated over the first 20 sentences (`overlap_size = min(20, len(sents))`).
- In Konkani's first 20 sentences, **not a single sentence triggered any of the four synthetic perturbation rules** in `annotator_b`.
- Therefore, `tags_a` and `tags_b` were 100% byte-for-byte identical across all tokens in the overlap sample:
  - Total Disagreements between A and B: **0**
  - Result: $\kappa = \frac{p_o - p_e}{1 - p_e} = \frac{1.0 - p_e}{1 - p_e} = 1.0000$.
- In Assamese, exactly 1 sentence (Sentence 13) triggered Ambiguity 3 on `"অসম সাহিত্য"`, yielding $\kappa = 0.9587$.
- In Sinhala, exactly 1 sentence (Sentence 8) triggered Ambiguity 3 on `"ශ්‍රී ලංකා"`, yielding $\kappa = 0.9277$.

---

## 4. The Critical Circularity: Rule Tagger Injected Post-Agreement

In `expand_long_multientity_set.py` lines 120-125 and lines 188-210:
1. `flat_a` and `flat_b` were collected strictly from `tags_a` and `tags_b` (the gazetteer matcher).
2. The agreement metric was calculated solely on `flat_a` and `flat_b`.
3. Then, `adjudicated_tags` was populated by overwriting `O` tags using `DedicatedORGRuleTagger` from `weak_supervision/org_rule_taggers.py`.
4. Overwrites:
   - **Konkani**: 483 tokens converted from `O` to `B-ORG`/`I-ORG`.
   - **Assamese**: 174 tokens converted from `O` to `B-ORG`/`I-ORG`.
   - **Sinhala**: 589 tokens converted from `O` to `B-ORG`/`I-ORG`.

### Agreement Between Gazetteer Baseline and Final "Gold" File
If one actually measures agreement between Annotator A (gazetteer) and the saved adjudicated file containing the rule tagger's ORGs:
- **Konkani**: Cohen's $\kappa = 0.1053$
- **Assamese**: Cohen's $\kappa = 0.2521$
- **Sinhala**: Cohen's $\kappa = 0.1691$

---

## 5. Cross-Language Consistency

All three languages used the exact same automated synthetic process. Konkani's perfect score of 1.0000 was not due to a divergent methodology, but because the first 20 sentences happened not to contain any of the 4 synthetic boundary keywords.

This process is circular: **a weak supervision rule tagger labeled the evaluation set, and the patched models were then evaluated against the rule tagger's own output.**
