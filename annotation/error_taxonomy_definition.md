# Error Taxonomy Specification & Deterministic Decision Rules

> **Phase 1 — Root-Cause Error Taxonomy Study**  
> **Repository**: `Explainable NER for Low-Resource Languages Using Attention Attribution`  
> **Target Languages**: Konkani, Assamese, Sinhala  
> **Model Under Study**: Frozen Baseline v1 (`baseline_v1/checkpoints/final_{lang}.pt`)  
> **Status**: AUTHORITATIVE SPECIFICATION — FIXED RULE SET  

---

## 1. Executive Summary & Objective

This document defines the formal, deterministic error taxonomy for Phase 1 of the v2 reliability overhaul. 
Historically, narrow interventions (such as gazetteer patching or manual test-case expansion) targeted isolated anecdotal errors without a census of underlying error mechanisms. 

To replace anecdote with rigorous quantitative evidence, this taxonomy partitions all discrepancies between gold annotations and model predictions into **8 mutually exclusive and exhaustive categories**. Every error observed in the diagnostic sample must map to **exactly one** primary category, preventing double-counting or orphaned discrepancies.

---

## 2. Mathematical Notation & Definitions

Let a sentence $S = (w_1, w_2, \dots, w_N)$ be a sequence of $N$ tokens.
- **Gold Entities**: $G = \{g_1, g_2, \dots, g_M\}$, where each gold entity $g_k = (T_k, s_k, e_k)$ denotes entity class $T_k \in \{\text{PER}, \text{LOC}, \text{ORG}, \text{MISC}\}$, token start index $s_k$, and token end index $e_k$ (half-open interval $[s_k, e_k)$, where $0 \le s_k < e_k \le N$).
- **Predicted Entities**: $P = \{p_1, p_2, \dots, p_K\}$, where each predicted entity $p_j = (T'_j, s'_j, e'_j)$ denotes predicted entity class $T'_j$, token start index $s'_j$, and token end index $e'_j$ ($[s'_j, e'_j)$).
- **Token Tags**: Gold tag sequence $\mathbf{y} = (y_1, \dots, y_N)$ and predicted tag sequence $\mathbf{\hat{y}} = (\hat{y}_1, \dots, \hat{y}_N)$ over the standard 9-tag BIO scheme:
  $$\mathcal{Y} = \{\text{O}, \text{B-PER}, \text{I-PER}, \text{B-LOC}, \text{I-LOC}, \text{B-ORG}, \text{I-ORG}, \text{B-MISC}, \text{I-MISC}\}$$

---

## 3. The 8 Concrete Error Categories

### Category 1: Missed Entity (`missed_entity` / False Negative)
- **Definition**: A gold entity $g = (T, s, e)$ has **zero token overlap** with any predicted entity span in $P$, in a sentence that is **not** completely collapsed.
- **Formal Condition**:
  $$\forall p \in P, \quad [s, e) \cap [s', e') = \emptyset \quad \wedge \quad \exists t \in \{1, \dots, N\} \text{ s.t. } \hat{y}_t \neq \text{O}$$
- **Linguistic / Model Root Cause**: Lexical absence, extreme morphological inflection (unseen case marker), low token prior probability, or lack of strong contextual cues.

### Category 2: False Positive (`false_positive` / Spurious Entity)
- **Definition**: A predicted entity $p = (T', s', e')$ has **zero token overlap** with any gold entity span in $G$, and exhibits a syntactically valid BIO transition.
- **Formal Condition**:
  $$\forall g \in G, \quad [s', e') \cap [s, e) = \emptyset \quad \wedge \quad \hat{y}_{s'} = \text{B-}T' \quad \wedge \quad \forall t \in (s', e'), \hat{y}_t = \text{I-}T'$$
- **Linguistic / Model Root Cause**: False trigger from capitalization/orthographic salience, common noun resembling named entity, or weak context hallucination.

### Category 3: Wrong Entity Type (`wrong_entity_type` / Type Confusion)
- **Definition**: A predicted entity $p = (T', s', e')$ matches the exact span boundaries of a gold entity $g = (T, s, e)$ ($s' = s \wedge e' = e$), but assigns the incorrect entity class ($T' \neq T$).
- **Formal Condition**:
  $$s' = s \quad \wedge \quad e' = e \quad \wedge \quad T' \neq T$$
- **Linguistic / Model Root Cause**: Semantic ambiguity or metonymy (e.g. country name used as Organization / Government vs Location; University classified as Location vs Organization).

### Category 4: Incomplete Span (`incomplete_span` / Boundary Under-Extraction)
- **Definition**: A predicted entity $p = (T', s', e')$ and gold entity $g = (T, s, e)$ share the same entity class ($T' = T$) and have non-empty token overlap ($[s', e') \cap [s, e) \neq \emptyset$), but the predicted span is strictly shorter than the gold entity span ($e' - s' < e - s$).
- **Formal Condition**:
  $$T' = T \quad \wedge \quad [s', e') \cap [s, e) \neq \emptyset \quad \wedge \quad (e' - s' < e - s) \quad \wedge \quad (s' \ge s \wedge e' \le e)$$
- **Linguistic / Model Root Cause**: Dropping case-inflected clitics/postpositions, truncating multi-word honorifics or compound headwords.

### Category 5: Overextended Span (`overextended_span` / Boundary Over-Extraction)
- **Definition**: A predicted entity $p = (T', s', e')$ and gold entity $g = (T, s, e)$ share the same entity class ($T' = T$) and have non-empty token overlap ($[s', e') \cap [s, e) \neq \emptyset$), but the predicted span includes adjacent non-entity tokens ($e' - s' > e - s$).
- **Formal Condition**:
  $$T' = T \quad \wedge \quad [s', e') \cap [s, e) \neq \emptyset \quad \wedge \quad (e' - s' > e - s) \quad \wedge \quad (s' \le s \wedge e' \ge e)$$
- **Linguistic / Model Root Cause**: Swallowing preceding titles, adjacent adjectives, or postpositional particles into the entity span.

### Category 6: B-/I- Transition Error (`bi_transition_error` / Structural Tag Flaw)
- **Definition**: An invalid BIO sequence transition occurs in the predicted tags $\mathbf{\hat{y}}$:
  1. An `I-TYPE` tag immediately preceded by `O` (orphan `I-` tag).
  2. An `I-TYPE_A` tag immediately preceded by `B-TYPE_B` or `I-TYPE_B` where $\text{TYPE}_A \neq \text{TYPE}_B$.
- **Formal Condition**:
  $$(\hat{y}_t = \text{I-}T_A \wedge \hat{y}_{t-1} = \text{O}) \quad \vee \quad (\hat{y}_t = \text{I-}T_A \wedge \hat{y}_{t-1} \in \{\text{B-}T_B, \text{I-}T_B\} \text{ with } T_A \neq T_B)$$
- **Linguistic / Model Root Cause**: Defect in transition matrix calibration or decoding constraints (e.g. CRF transition under-penalization).

### Category 7: O-Predicted-Inside-Entity (`o_inside_entity` / Entity Puncture)
- **Definition**: Within a multi-token gold entity $g = (T, s, e)$ where $e - s \ge 2$, the model predicts an entity token at the boundary or prefix, but predicts `O` on one or more intermediate tokens inside $[s, e)$, puncturing or fragmenting the entity.
- **Formal Condition**:
  $$e - s \ge 2 \quad \wedge \quad \exists t_1, t_2 \in [s, e) \text{ with } t_1 < t_2 \text{ s.t. } \hat{y}_{t_1} \neq \text{O} \wedge \hat{y}_{t_2} \neq \text{O} \quad \wedge \quad \exists t_{\text{mid}} \in (t_1, t_2) \text{ s.t. } \hat{y}_{t_{\text{mid}}} = \text{O}$$
  *(Also covers cases where $[s, e)$ is predicted as $[s, t) \cup [t+1, e)$ split by an internal `O`)*.
- **Linguistic / Model Root Cause**: Unknown internal modifier, hyphen, or conjunct character causing the sequence model to drop to background `O`.

### Category 8: Complete-Sentence Collapse (`complete_sentence_collapse` / Zero-Entity Collapse)
- **Definition**: A sentence contains one or more gold entities ($|G| \ge 1$), but the model predicts **zero entities** across the entire sentence ($\mathbf{\hat{y}} = (\text{O}, \dots, \text{O})$).
- **Formal Condition**:
  $$|G| \ge 1 \quad \wedge \quad \forall t \in \{1, \dots, N\}, \quad \hat{y}_t = \text{O}$$
- **Linguistic / Model Root Cause**: Global sequence collapse, often triggered by sentence length exceeding training distribution, dense multi-clause structures, or high cumulative token OOV rates triggering CRF all-O decoding paths.

---

## 4. Precedence & Ambiguity Resolution Rules

To guarantee strict mathematical consistency and eliminate ad-hoc, case-by-case re-decisions, the following deterministic rules govern edge cases:

1. **Rule 1 (Truncation vs False Negative)**:
   A truncated gold span (e.g. gold `[0, 3)` predicted as `[1, 3)`) is categorized strictly as `incomplete_span`. It is **never** double-counted as a `missed_entity` for the missing token or a `false_positive` for the partial span.

2. **Rule 2 (Expansion vs False Positive)**:
   An overextended span (e.g. gold `[1, 2)` predicted as `[0, 2)`) is categorized strictly as `overextended_span`. The extra token is **never** counted as a separate `false_positive`.

3. **Rule 3 (Type Mismatch on Partial Overlap)**:
   If a predicted span partially overlaps a gold span ($P \cap G \neq \emptyset$) but their entity types differ ($T' \neq T$):
   - If the overlap covers $\ge 50\%$ of both spans, it is categorized as `wrong_entity_type` (semantic misclassification dominates boundary jitter).
   - If overlap is $< 50\%$, the predicted span is categorized as `false_positive` and the gold entity as `missed_entity`.

4. **Rule 4 (Sentence Collapse Hierarchy)**:
   If a sentence satisfies the condition for `complete_sentence_collapse`:
   - At the sentence level, the sentence is recorded as a `complete_sentence_collapse`.
   - At the entity level, every gold entity in that sentence is categorized under `complete_sentence_collapse` (rather than generic `missed_entity`). This isolates length/density-driven systemic collapses from ordinary lexical misses.

5. **Rule 5 (Mutual Exclusivity & Partition Completeness)**:
   Every discrepancy between gold and predicted tag sequences maps to **exactly one** category in the primary partition:
   $$\text{Total Errors} = \sum_{k=1}^8 \text{Count}(\text{Category}_k)$$
   No discrepancy is omitted; no discrepancy is double-counted.

---

## 5. Statistical Reliability Threshold

In alignment with project governance:
- Any category, language, or cross-tabulation slice with $N < 10$ instances is explicitly flagged as **[Statistically Fragile (N < 10)]**.
- If a category fails to reach $N \ge 30$ instances after pooling available evaluation sets and held-out diagnostic material, the actual achievable count is reported honestly with this fragility flag.
