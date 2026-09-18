# Faithfulness Test Design Audit: Asymmetry & Target Token Erasure Mechanism

**Audit Date**: 2026-09-18  
**Auditor**: Senior ML/NLP Research Engineer  
**Target Component**: [`explainability/erasure_faithfulness_test.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/explainability/erasure_faithfulness_test.py)  
**Scope**: Pre-submission verification of Finding 2 (Negative Faithfulness Gap & Inverted Method Selection)  

---

## 1. Executive Summary & Audit Verdict

### Verdict: **STRUCTURAL SELECTION ASYMMETRY CONFIRMED WITH DIRECT CODE & EMPIRICAL PROOF**

The reported negative faithfulness gaps ($\text{Gap} = \text{Drop}_{\text{top}} - \text{Drop}_{\text{rand}} < 0$) in `reports/phase13_faithfulness_results.md` were **not caused by unfaithful attribution**, but by an **asymmetric candidate pool design flaw** in `explainability/erasure_faithfulness_test.py`.

1. **Near-Zero Target Token Attribution**: Across all $N = 395$ gold-dev entity instances across Konkani, Assamese, and Sinhala, the target entity token itself receives an attribution score $\le 0.001$ in **$96.7\%$** of cases.
2. **Top-$k$ Behavior**: Because the target entity token has near-zero attribution, it is **never selected in Top-1** ($0.0\%$ in Konkani and Sinhala, $0.72\%$ in Assamese). Top-$k$ erasure exclusively replaces *context tokens* (syntactic connectors, punctuation, or verbs) with `<UNK>`, which causes a minimal confidence drop ($\sim 0.001$ to $0.01$).
3. **Random-$k$ Behavior**: Because the target entity token is not in Top-$k$, it is placed into `non_top_pool`. Random-$k$ selection draws from this pool and inadvertently erases the **target entity token itself** in **$7.0\% - 10.0\%$** of instances for $k=1$, **$18.0\% - 23.0\%$** for $k=2$, and **$25.0\% - 29.0\%$** for $k=3$.
4. **Catastrophic Drop Disparity**: Erasing the entity token itself collapses prediction confidence by **$0.9161$ to $0.9838$** (from $\sim 0.99$ to $\sim 0.04$). Erasing context tokens only drops confidence by **$0.0603$ to $0.0936$**.
5. **The Artifact**: The $7\% - 29\%$ frequency of erasing the entity token in the control group inflated $\text{Drop}_{\text{rand}}$ to $0.09 - 0.38$, mathematically guaranteeing a negative gap.

---

## 2. Code Evidence of Selection Asymmetry

In [`explainability/erasure_faithfulness_test.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/explainability/erasure_faithfulness_test.py), lines 180–226 reveal the asymmetric selection logic:

```python
# (a) Top-k erasure for Grad x Attn
grad_ranked_indices = torch.argsort(grad_scores, descending=True).tolist()
top_k_grad = grad_ranked_indices[:k]
tokens_grad_erased = list(tokens)
for idx in top_k_grad:
    tokens_grad_erased[idx] = "<UNK>"
conf_grad_erased = get_token_confidence(model, extractor, tokens_grad_erased, q_idx, target_tag_id)
drop_grad = max(0.0, orig_conf - conf_grad_erased)

# (b) Top-k erasure for Raw Attention
raw_ranked_indices = torch.argsort(raw_scores, descending=True).tolist()
top_k_raw = raw_ranked_indices[:k]
...

# (c) Random-k erasure (control)
non_top_pool = [i for i in range(seq_len) if i not in top_k_grad and i not in top_k_raw]
if len(non_top_pool) >= k:
    rand_indices = rng.sample(non_top_pool, k)
else:
    rand_indices = rng.sample(range(seq_len), min(k, seq_len))

tokens_rand_erased = list(tokens)
for idx in rand_indices:
    tokens_rand_erased[idx] = "<UNK>"
conf_rand_erased = get_token_confidence(model, extractor, tokens_rand_erased, q_idx, target_tag_id)
drop_rand = max(0.0, orig_conf - conf_rand_erased)
```

### Critical Code Flaw:
- `q_idx` is the index of the entity token being predicted.
- `top_k_grad` is the top $k$ highest attributed positions. Because `grad_scores[q_idx]` $\approx 0.0000$, `q_idx` is virtually never in `top_k_grad`.
- `non_top_pool` contains all tokens NOT in `top_k_grad` or `top_k_raw`. Therefore, **`q_idx` is almost always included in `non_top_pool`**.
- `rand_indices` samples from `non_top_pool` with replacement probability $\frac{k}{|\text{non\_top\_pool}|} \approx \frac{k}{15-25} \approx 5\% - 20\%$.
- When `q_idx` is in `rand_indices`, `tokens_rand_erased[q_idx] = "<UNK>"`, erasing the very entity whose tag is being predicted.

---

## 3. Programmatic Census of Target-Token Attribution ($N = 395$)

Evaluated across all gold-dev entity instances in [`reports/experiment_logs/target_token_attribution_audit.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/experiment_logs/target_token_attribution_audit.json):

| Language | Total Entities ($N$) | Mean Attribution (Grad $\times$ Attn) | Mean Attention (Raw) | Attribution $\le 0.001$ Rate | Attribution $\le 0.01$ Rate | Ranked #1 Rate | Ranked in Top-3 Rate | Mean Rank (out of seq_len) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 125 | 0.003278 | 0.002845 | **92.8%** (116/125) | 94.4% (118/125) | 0.0% (0/125) | 1.6% (2/125) | 16.94 |
| **Assamese** | 139 | 0.002364 | 0.000969 | **98.56%** (137/139) | 98.56% (137/139) | 0.72% (1/139) | 1.44% (2/139) | 17.17 |
| **Sinhala** | 131 | 0.000204 | 0.000043 | **98.47%** (129/131) | 99.24% (130/131) | 0.0% (0/131) | 0.0% (0/131) | 20.57 |
| **ALL / MACRO** | **395** | **0.001949** | **0.001286** | **96.7%** (382/395) | **97.5%** (385/395) | **0.25%** (1/395) | **1.01%** (4/395) | **18.23** |

---

## 4. Measured Asymmetry in Entity Token Selection ($N = 100$)

Across 100 random evaluation samples per language:

| Condition | $k$ | Konkani Entity Inclusions | Assamese Entity Inclusions | Sinhala Entity Inclusions |
| :--- | :---: | :---: | :---: | :---: |
| **Top-$k$ (Grad $\times$ Attn)** | $k=1$ | **0.0%** (0/100) | **1.0%** (1/100) | **0.0%** (0/100) |
| | $k=2$ | **1.0%** (1/100) | **2.0%** (2/100) | **0.0%** (0/100) |
| | $k=3$ | **2.0%** (2/100) | **2.0%** (2/100) | **0.0%** (0/100) |
| **Top-$k$ (Raw Attention)** | $k=1$ | **0.0%** (0/100) | **0.0%** (0/100) | **0.0%** (0/100) |
| | $k=2$ | **0.0%** (0/100) | **0.0%** (0/100) | **0.0%** (0/100) |
| | $k=3$ | **1.0%** (1/100) | **0.0%** (0/100) | **0.0%** (0/100) |
| **Random-$k$ Pool (Control)** | $k=1$ | **7.0%** (7/100) | **9.0%** (9/100) | **10.0%** (10/100) |
| | $k=2$ | **23.0%** (23/100) | **22.0%** (22/100) | **18.0%** (18/100) |
| | $k=3$ | **29.0%** (29/100) | **28.0%** (28/100) | **25.0%** (25/100) |

### Impact on Measured Drops:
- **Mean drop when ENTITY TOKEN ITSELF is erased**:
  - Konkani: **0.9161**
  - Assamese: **0.9518**
  - Sinhala: **0.9838**
- **Mean drop when ONLY CONTEXT TOKENS are erased**:
  - Konkani: **0.0693**
  - Assamese: **0.0603**
  - Sinhala: **0.0936**

---

## 5. Corrected Symmetric Methodology Design

To evaluate **contextual attribution faithfulness** validly:
1. **Candidate Pool Definition**:
   $$\mathcal{C} = \{j \in \{0, \dots, T-1\} \mid j \neq q\_\text{idx}\}$$
   The query entity token $q\_\text{idx}$ is **strictly excluded** from candidate selection for BOTH Top-$k$ and Random-$k$.
2. **Top-$k$ Selection**:
   $$S_{\text{top}} = \arg\max_{S \subset \mathcal{C}, |S|=k} \sum_{j \in S} \text{Score}(q \to j)$$
3. **Random-$k$ Selection**:
   $$S_{\text{rand}} \sim \text{Uniform}(\mathcal{C} \setminus S_{\text{top}}, k)$$
4. **Scientific Integrity**:
   Under this symmetric design, neither Top-$k$ nor Random-$k$ is permitted to destroy the query token. Both methods test the same causal question: *Does erasing the model's top-attributed context words degrade entity confidence more than erasing random context words?*
