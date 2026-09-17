# Verification Findings: Selection Bias in `long_multientity_eval.conll`

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Scope**: Code audit of `annotation/expand_long_multientity_set.py` and comparison against raw corpus candidate pool.

---

## 1. Executive Summary

- **Finding**: **CONFIRMED BIASED**.
- The candidate sentence selection in `annotation/expand_long_multientity_set.py` systematically and heavily biased selection toward ORG-containing sentences.
- Furthermore, an external rule tagger (`weak_supervision/org_rule_taggers.py:DedicatedORGRuleTagger`) was used directly inside the script to inject synthetic ORG annotations onto tokens where gazetteers had found nothing. This single injection accounted for:
  - **98.8% of all ORGs in Konkani** (159 of 161 ORGs)
  - **98.5% of all ORGs in Assamese** (64 of 65 ORGs)
  - **99.4% of all ORGs in Sinhala** (172 of 173 ORGs)

---

## 2. Selection Logic Trace: Exact Code Evidence

From [annotation/expand_long_multientity_set.py](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/expand_long_multientity_set.py):

### Filter 1: Token Length Window (Lines 106-107)
```python
toks = line_str.split()
if len(toks) < min_tokens or len(toks) > 85:
    continue
```
*Effect*: Constrains search to long sequences ($28 \le \text{len} \le 85$).

### Filter 2: Rapid Keyword Screening (Lines 88-92, 109-113)
```python
# Screening sets for rapid corpus traversal
gaz_first_words = set(span[0] for etype in entity_dict for span in entity_dict[etype] if span)
org_head_keywords = set(org_tagger.head_keywords)
org_modifiers = set(org_tagger.prefix_modifiers)
...
# Quick keyword screen: must contain either gazetteer token, org keyword, or delimiter
has_gaz = any(t in gaz_first_words for t in toks)
has_org = any(t in org_head_keywords or t in org_modifiers for t in toks)
if not (has_gaz or has_org):
    continue
```
*Effect*: Drops all sentences lacking an exact token match from either the entity dictionaries or the ORG rule tagger's institutional keyword list. In the raw corpus, only 14.45% of long Konkani sentences, 20.45% of Assamese sentences, and 29.40% of Sinhala sentences contain an ORG cue keyword.

### Filter 3: Rule-Tagger Injection into "Gold" Labels (Lines 120-125)
```python
# Institutional ORG expansion per guidelines
org_tags = org_tagger.tag_sentence(toks)
adjudicated_tags = list(tags_a)
for i in range(len(toks)):
    if adjudicated_tags[i] == "O" and org_tags[i] != "O":
        adjudicated_tags[i] = org_tags[i]
```
*Effect*: Directly executes the weak-supervision heuristic `DedicatedORGRuleTagger` and mutates the tags, converting `O` tokens into `B-ORG` and `I-ORG`.

### Filter 4: Acceptance Criteria and Loop Termination (Lines 133-156)
```python
ents = extract_entities(adjudicated_tags)
if len(ents) < 2:
    continue

mids = find_middle_entities(toks, ents, lang)
orgs = [e for e in ents if e[0] == "ORG"]

if len(orgs) > 0 or len(mids) > 0:
    seen_texts.add(line_str)
    total_org_count += len(orgs)
    total_mid_count += len(mids)

    selected_sentences.append({ ... })

    if total_org_count >= target_orgs and total_mid_count >= target_middle and len(selected_sentences) >= 30:
        break
```
*Effect*: Sentences were only selected if they had $\ge 2$ entities after the rule-tagger injection, and had at least one ORG or middle entity.

---

## 3. Why Konkani and Sinhala Saturated with ORGs

The loop termination condition explains the massive sentence counts and ORG saturation in Konkani (72 sentences, 161 ORGs) and Sinhala (78 sentences, 173 ORGs) versus Assamese (30 sentences, 65 ORGs):
1. **Target**: `target_orgs = 25`, `target_middle = 25`, `min_sentences = 30`.
2. **Assamese**: Natural middle entities in long sentences were relatively dense; Assamese reached 43 middle entities and 65 ORGs by sentence 30, triggering the break condition immediately.
3. **Konkani & Sinhala**: Middle-list entities were rare in the raw corpus. It required scanning through **72 sentences in Konkani** and **78 sentences in Sinhala** to reach the 25 middle-entity target.
4. **Cumulative Injection**: Because `DedicatedORGRuleTagger` was executed on every accepted sentence, as the loop struggled to accumulate middle entities, it accumulated ~2.2 rule-injected ORGs per sentence, blowing the total ORG counts up to 161 (Konkani) and 173 (Sinhala).

---

## 4. Conclusion

The long multi-entity evaluation set was not sampled representatively from naturally occurring multi-entity sentences. It was filtered for ORG keywords and systematically injected with weak-supervision rule tags from `DedicatedORGRuleTagger`.
