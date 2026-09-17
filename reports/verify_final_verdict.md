# Verification Final Verdict: Selection-Bias and Annotation-Independence Audit

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Audit Target**: Integrity of `data/gold/{lang}/long_multientity_eval.conll` and validity of Patch v2 checkpoint recommendations.

---

## 1. Executive Summary: Definitive Answers to the Two Suspicious Numbers

### Question 1: Why is the entity composition 80.5% ORG in Konkani, 88.7% in Sinhala, and 57.0% in Assamese?
- **Definitive Answer**: **CONFIRMED SELECTION BIAS AND SYNTHETIC RULE INJECTION**.
- **Mechanisms Proven in Code**:
  1. **ORG Keyword Screening**: In [annotation/expand_long_multientity_set.py](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/expand_long_multientity_set.py) (lines 109-113), candidates were screened requiring either a gazetteer hit or an ORG keyword/modifier.
  2. **Rule-Tagger Injection**: Lines 120-125 directly ran `weak_supervision/org_rule_taggers.py:DedicatedORGRuleTagger` and wrote its predicted ORG tags directly into `adjudicated_tags`.
  3. **Termination Imbalance**: Sourcing required 25 middle-list entities. Because middle entities are sparse in Konkani and Sinhala, the loop ran for **72 sentences in Konkani** and **78 sentences in Sinhala** (accumulating ~2.2 rule-injected ORGs per sentence, reaching 161 and 173 ORGs), whereas Assamese found middle entities faster and terminated at 30 sentences (accumulating 65 ORGs).
  4. **Proportion of Injected ORGs**:
     - Konkani: 159 of 161 ORGs (**98.8%**) came from `DedicatedORGRuleTagger`.
     - Assamese: 64 of 65 ORGs (**98.5%**) came from `DedicatedORGRuleTagger`.
     - Sinhala: 172 of 173 ORGs (**99.4%**) came from `DedicatedORGRuleTagger`.

### Question 2: Why was Konkani's inter-annotator agreement reported as perfect Cohen's $\kappa = 1.0000$?
- **Definitive Answer**: **CONFIRMED PROCESS CIRCULARITY AND ARTIFACT OF SYNTHETIC CODE**.
- **Mechanisms Proven in Code**:
  1. **No Human Annotators**: Neither Annotator A nor Annotator B was human. Both were programmatic calls to `annotate_gold.py:annotate_sentence()`.
  2. **Exact Identity**: Annotator B is simply Annotator A with 4 specific synthetic boundary perturbation rules. In Konkani's first 20 overlap sentences, **none** of the 4 perturbation triggers were activated. As a result, `tags_a` and `tags_b` were 100% identical token-for-token across all 20 sentences, yielding an exact $\kappa = 1.0000$ and Span F1 $= 1.0000$.
  3. **Agreement Metric Excluded the Injected ORGs**: The $\kappa$ metric was calculated strictly between Annotator A and Annotator B **before** `DedicatedORGRuleTagger` injected the 161 ORGs. If one actually computes agreement between the gazetteer baseline (Annotator A) and the final file containing the rule tagger's ORGs, the true agreement is **$\kappa = 0.1053$** for Konkani.
  4. **Cross-Language Confirmation**: Assamese ($\kappa = 0.9587$) and Sinhala ($\kappa = 0.9277$) used the exact same synthetic script, but each happened to have exactly 1 sentence in their first 20 that triggered Ambiguity Rule 3.

---

## 2. Recomputed Entity-Type Distribution Table

Recomputed directly from file contents across all four datasets per language:

| Language | Dataset | Sentences | Total Ents | PER (%) | LOC (%) | ORG (%) | MISC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `long_multientity_eval` | 72 | 200 | 0.0% | 19.5% | **80.5%** | 0.0% |
| | `dev` (baseline) | 200 | 125 | 16.0% | 24.8% | **29.6%** | 29.6% |
| | `test` (locked) | 200 | 124 | 16.1% | 24.2% | **29.8%** | 29.8% |
| | `train` | 600 | 372 | 16.4% | 24.5% | **29.6%** | 29.6% |
| **Assamese** | `long_multientity_eval` | 30 | 114 | 0.9% | 42.1% | **57.0%** | 0.0% |
| | `dev` (baseline) | 200 | 139 | 13.0% | 56.8% | **10.8%** | 19.4% |
| | `test` (locked) | 200 | 138 | 13.8% | 57.2% | **10.1%** | 18.8% |
| | `train` | 600 | 416 | 13.2% | 57.0% | **10.6%** | 19.2% |
| **Sinhala** | `long_multientity_eval` | 78 | 195 | 0.0% | 11.3% | **88.7%** | 0.0% |
| | `dev` (baseline) | 200 | 131 | 28.2% | 29.8% | **21.4%** | 20.6% |
| | `test` (locked) | 200 | 132 | 28.8% | 29.5% | **21.2%** | 20.5% |
| | `train` | 600 | 393 | 28.8% | 29.5% | **21.1%** | 20.6% |

---

## 3. Remediation Recommendation: How to Fix `long_multientity_eval.conll`

To convert `long_multientity_eval.conll` into a genuine, scientifically defensible evaluation set:
1. **Unbiased Candidate Selection**:
   - Filter sentences from the untouched raw corpus (`data/processed/{lang}/sentences.txt`) strictly by length ($30 \le \text{tokens} \le 80$) and natural delimiter density ($\ge 2$ commas or native conjunctions).
   - **DO NOT filter or rank by ORG keywords or gazetteer hits**.
2. **Strict Removal of Rule Taggers**:
   - `DedicatedORGRuleTagger` must be completely banned from modifying evaluation sets. A model evaluated against a weak-supervision rule tagger only measures adherence to the heuristic, not real-world generalization.
3. **Genuine Double Annotation**:
   - Annotation must be performed by independent passes without seeing the rule-tagger output.
   - Agreement metrics ($\kappa$) must be calculated on the full candidate set, and adjudication must be recorded with an explicit dispute log.
4. **Corpus Feasibility**:
   - The raw corpus contains **30,980** Konkani, **55,002** Assamese, and **66,954** Sinhala sentences.
   - There are ample natural sentences ($\ge 2,000$ per language) meeting length and conjunction criteria to support constructing a balanced 50-sentence slice per language with natural distributions of PER, LOC, ORG, and MISC.

---

## 4. Final Verdict on Patch v2 Recommendation

### FINAL VERDICT: **NEEDS REVISION**

### Justification:
1. **Konkani**: The rejection of `patched_v2_konkani.pt` in favor of baseline `selftrained_konkani_best.pt` **STANDS**. The patched model regressed on both the slice and genuine Gold-Dev.
2. **Sinhala**: The rejection of `patched_v2_sinhala.pt` in favor of baseline `augmented_sinhala.pt` **STANDS**. The patched model regressed on Gold-Dev and provided no benefit.
3. **Assamese**: The recommendation to advance `patched_v2_assamese.pt` **CANNOT STAND AS CURRENTLY JUSTIFIED**. Its apparent $+8.49\%$ ORG F1 gain was measured on an evaluation set that was $57\%$ ORG, of which $98.5\%$ were tagged by the weak supervision rule tagger. On genuine Gold-Dev, it showed zero gain on ORG and long sentences and dropped $-0.70\%$ overall.
4. **Recommended Revision**:
   - **Option A (Safest)**: Advance to Phase 12 using the **pre-patch baselines for all three languages** (`selftrained_konkani_best.pt`, `augmented_assamese.pt`, `augmented_sinhala.pt`), discarding the contaminated Patch v2 checkpoints entirely.
   - **Option B (Re-run Eval)**: Re-create an unbiased `long_multientity_eval.conll` using length-only sampling, re-evaluate `patched_v2_assamese.pt` vs `augmented_assamese.pt`, and let the human review the genuine numbers.
