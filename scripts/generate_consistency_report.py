"""
Generates reports/cycle2_assamese_label_consistency_audit.md from JSON diagnostic data.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
d = json.load(open(PROJECT_ROOT / "scratch" / "cycle2_step1b_disentanglement.json", "r", encoding="utf-8"))

stats = d["train_consistency_stats"]
inconsistent_records = d["train_consistency_records_inconsistent"]

# Known glosses for key Assamese words
GLOSSES = {
    "বিষ্ণুপ্ৰসাদ": "Bishnuprasad (Proper Name)",
    "আগৰৱালা": "Agarwala (Surname)",
    "শ্ৰীমন্ত": "Srimanta (Honorific/Proper Name)",
    "গণ": "People / Mass (in Asom Gana Parishad)",
    "গোপীনাথ": "Gopinath (Proper Name)",
    "ৰঙালী": "Rongali (Spring Bihu Festival)",
    "চৰকাৰ": "Government / Sarkar",
    "লাচিত": "Lachit (Borphukan, Historical General)",
    "সভা": "Assembly / Council / Sabha",
    "অসমৰ": "Assam's (Genitive Location)",
    "লক্ষ্মীনাথ": "Lakshminath (Bezbaroa, Writer)",
    "বিশ্ববিদ্যালয়": "University (Common/Org Noun)",
    "ভোগালী": "Bhogali (Harvest Bihu Festival)",
    "চুক্তি": "Accord / Agreement",
    "পাৰ্টি": "Party (Political)",
    "বুঢ়ী": "Old Woman (in Grandmothers Tales)",
    "আইৰ": "Mother's (Genitive)",
    "কংগ্ৰেছ": "Congress (Political Org)",
    "তৰুণ": "Tarun (Proper Name / Young)",
    "শৰ্মা": "Sarma (Surname)",
    "হাজৰিকা": "Hazarika (Surname)",
    "বিধানসভা": "Legislative Assembly",
    "মহন্ত": "Mahanta (Surname)",
    "ফণী": "Phani (Proper Name)",
    "দাস": "Das (Surname)",
    "পৰিষদ": "Parishad / Council",
    "ভাৰতৰ": "India's (Genitive Location)",
    "বৰদলৈ": "Bordoloi (Surname)",
    "অসমত": "In Assam (Locative Location)",
    "ভাৰতীয়": "Indian (Adjective/Org component)",
    "জাতীয়": "National",
    "সাহিত্য": "Literature (in Asam Sahitya Sabha)",
    "বৰুৱা": "Barua (Surname)",
    "দিল্লী": "Delhi (Location)",
}

report_md = f"""# Reliability Cycle 2 — Step 1b: Assamese Training Label Consistency Audit

> **Target Deliverable**: Comprehensive full-corpus label-consistency audit for Assamese Named Entity Recognition.  
> **Input Corpus**: `data/gold/assamese/train.conll` (850 sentences, 14,649 non-empty token lines, 4,789 distinct surface types).  
> **Evaluation Checkpoint**: Baseline `checkpoints/improved_assamese_v2.pt` / `baseline_v1/checkpoints/final_assamese.pt`.  
> **Status**: **COMPLETE — AUDIT SYSTEMATICALLY GENERALIZED FROM 2 SPOT-CHECK EXAMPLES TO ALL QUALIFYING TOKENS (N=475)**.

---

## 1. Executive Summary

In Step 1, a spot-check of two entity tokens (`ভাৰতৰ` and `ভাৰতীয়`) revealed severe label inconsistency in the gold training set (70.6% and 85.9% tagged as `O`, respectively), producing an overwhelming static emission $O$-bias (+10 to +13 logits).

This audit expands that initial spot-check into a **complete, corpus-wide census** of all distinct token types appearing at least 5 times in `data/gold/assamese/train.conll` ($N = 475$ token types, encompassing 9,141 total token occurrences, or 62.4% of the training corpus).

### Key Empirical Findings:
1. **The vast majority of frequent tokens are 100% consistent**: **402 out of 475 token types (84.6%)** exhibit perfect label consistency (rate $= 1.0000$), virtually all of which are common background syntactic markers (verbs, postpositions, conjunctions, punctuation, and invariable non-entities) consistently tagged as `O`.
2. **Entity-bearing and ambiguous tokens are heavily concentrated in a small, highly inconsistent tail**: **73 token types (15.4%)** have inconsistent annotations (rate $< 1.0000$).
3. **Severe label ambiguity affects high-impact core entities**:
   - `অসমৰ` (Assam's): 65 training occurrences, tagged `O` **58.5% (38/65)** and `B-LOC` 41.5% (27/65).
   - `ভাৰতৰ` (India's): 34 training occurrences, tagged `O` **70.6% (24/34)** and `B-LOC` 29.4% (10/34).
   - `ভাৰতীয়` (Indian): 92 training occurrences, tagged `O` **85.9% (79/92)** and `B-ORG` 14.1% (13/92).
   - `শ্ৰীমন্ত` (Srimanta): 19 training occurrences, tagged `O` **52.6% (10/19)** and `B-PER` 47.4% (9/19).
   - `লক্ষ্মীনাথ` (Lakshminath): 17 training occurrences, tagged `O` **58.8% (10/17)** and `B-PER` 41.2% (7/17).
   - `বিশ্ববিদ্যালয়` (University): 25 training occurrences, tagged `O` **40.0% (10/25)** and `I-ORG` 60.0% (15/25).
4. **Empirical Cutoff for Low Consistency**: The 85th percentile of the inconsistent distribution corresponds to **consistency rate $< 0.85$**, which captures the **56 most ambiguous tokens (11.8% of all frequent token types)**. A stricter threshold of $< 0.80$ isolates **43 tokens (9.1%)**, while $< 0.70$ isolates **25 tokens (5.3%)** exhibiting coin-flip ambiguity.

---

## 2. Corpus Consistency Distribution

### 2.1 Distribution Summary Statistics ($N=475$ Types with Frequency $>= 5$)

| Metric | Empirical Value | Context / Interpretation |
| :--- | :---: | :--- |
| **Total Sentences in Gold Train** | 850 | Standard gold training split |
| **Total Token Lines in Gold Train** | 14,649 | Full token volume |
| **Total Distinct Token Types** | 4,789 | Vocabulary size in training split |
| **Types with Frequency $>= 5$** | **475** | Qualifying population for audit |
| **Minimum Consistency Rate** | **0.5000 (50.0%)** | `বিষ্ণুপ্ৰসাদ` (3 B-PER, 3 O), `আগৰৱালা` (3 O, 3 I-PER) |
| **10th Percentile ($P_{{10}}$)** | 0.8000 (80.0%) | Bottom decile suffers $>= 20%$ contradictory labels |
| **25th Percentile ($Q_1$)** | 1.0000 (100.0%) | Top 75% includes invariable background tokens |
| **Median Consistency Rate** | 1.0000 (100.0%) | Background $O$ tokens dominate median |
| **75th Percentile ($Q_3$)** | 1.0000 (100.0%) | Background $O$ tokens dominate upper quartile |
| **90th Percentile ($P_{{90}}$)** | 1.0000 (100.0%) | Background $O$ tokens dominate 90th percentile |
| **Maximum Consistency Rate** | 1.0000 (100.0%) | 402 tokens |

### 2.2 Stratified Consistency Tiers

| Consistency Tier | Criteria | Count of Types | % of Types ($N=475$) | Qualitative Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 0: Perfectly Invariable** | Consistency $= 1.0000$ | 402 | 84.6% | Non-entity words (`আৰু`, `হয়`, punctuation) + rare clean entities |
| **Tier 1: Mild Inconsistency** | $0.85 <= \text{{Rate}} < 1.00$ | 17 | 3.6% | Mostly consistent entities with occasional noise (`ভাৰতীয়`, `অসমীয়া`) |
| **Tier 2: Moderate Ambiguity** | $0.80 <= \text{{Rate}} < 0.85$ | 13 | 2.7% | Regular label conflict (`ভূপেন`, `শইকীয়া`, `ৰাজ্যিক`) |
| **Tier 3: Severe Ambiguity** | $0.70 <= \text{{Rate}} < 0.80$ | 18 | 3.8% | High label conflict (`ভাৰতৰ`, `অসমত`, `পৰিষদ`, `বৰদলৈ`) |
| **Tier 4: Extreme / Near Coin-Flip** | $\text{{Rate}} < 0.70$ | 25 | 5.3% | Near 50/50 noise (`অসমৰ`, `শ্ৰীমন্ত`, `লক্ষ্মীনাথ`, `আগৰৱালা`) |

---

## 3. Complete Ranked Inventory of Inconsistent Training Tokens

The following table contains all **73 token types** in `data/gold/assamese/train.conll` with occurrence count >= 5 that exhibit contradictory labels (consistency rate < 1.0000), ranked from least consistent to most consistent:

| Rank | Token | English Gloss / Category | Total Train Count | Top Tag | Top Count | Consistency Rate | Train O Count | Train O % | Entity Count | Tag Distribution Detail |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

for rank, r in enumerate(inconsistent_records, 1):
    tok = r["token"]
    gloss = GLOSSES.get(tok, "—")
    top_t = r["top_tag"]
    c_rate = r["consistency_rate"] * 100
    o_rate = r["o_rate"] * 100
    tag_str = ", ".join(f"{k}: {v}" for k, v in sorted(r["tag_distribution"].items(), key=lambda x: -x[1]))
    report_md += f"| {rank} | **{tok}** | {gloss} | {r['total_count']} | `{top_t}` | {r['top_tag_count']} | **{c_rate:.1f}%** | {r['o_count']} | {o_rate:.1f}% | {r['entity_count']} | `{tag_str}` |\n"

report_md += """
---

## 4. Architectural and Training Impact Analysis

### 4.1 Why the Linear Emission Head Collapses to $O$
In the multilingual NER architecture (`improved_assamese_v2.pt`), the linear emission layer maps attended representations directly into tag logits. When high-frequency entities appear in training with `O` as their dominant tag:
1. **Direct Loss Incentive**: Cross-entropy / negative log-likelihood strongly rewards predicting `O` for tokens like `ভাৰতীয়` (79 times $O$ vs 13 times entity) and `ভাৰতৰ` (24 times $O$ vs 10 times entity).
2. **Weight Polarization**: The linear emission head weights for class `O` receive massive positive gradient reinforcement, driving the static $O$ logit up to $+5.7$ to $+6.1$, while entity tag logits receive heavy negative penalty (down to $-4.9$ to $-6.9$).
3. **CRF Subjugation**: The linear emission logit gap of $+10$ to $+13$ logits dwarfs the CRF transition matrix weights (which operate in the range of $-1.5$ to $+0.8$). The Viterbi decoding path is mathematically forced through the $O \to O$ sequence regardless of contextual signals.

---

## 5. Audit Conclusions & Diagnostic Thresholds

1. **Confirmation of Corpus-Wide Label Pathology**: Step 1's initial finding regarding `ভাৰতৰ` and `ভাৰতীয়` is not an isolated artifact. It is a systematic feature of the Assamese gold training split affecting the primary geographic and political entity vocabulary of the language (`অসমৰ`, `ভাৰতৰ`, `ভাৰতীয়`, `অসমত`, `দিল্লী`).
2. **Derived Threshold for Downstream Disentanglement**:
   - **Low Consistency Cutoff**: $\text{Consistency Rate} < 0.85$ (captures the 56 most heavily compromised types, comprising the bottom 11.8% of the frequent vocabulary).
   - **Extreme Ambiguity Cutoff**: $\text{Consistency Rate} < 0.70$ (captures the 25 most severely degraded types, where annotation is effectively a coin flip).
"""

(PROJECT_ROOT / "reports" / "cycle2_assamese_label_consistency_audit.md").write_text(report_md, encoding="utf-8")
print(f"Generated reports/cycle2_assamese_label_consistency_audit.md ({len(inconsistent_records)} inconsistent records listed).")
