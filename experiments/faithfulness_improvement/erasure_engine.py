"""
Symmetric Erasure Evaluation Engine for Faithfulness Benchmarking.

Evaluates causal degradation of prediction confidence under top-k vs random-k token erasure.
Supports:
1. Perturbation strategies: '<UNK>' token replacement vs Zero-Embedding replacement.
2. Confidence modes: Local Emission Softmax vs Exact CRF Marginal Probability.
3. Strict symmetric candidate pool (context only, target entity token strictly excluded).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from explainability.attention_extractor import AttentionExtractor
from experiments.faithfulness_improvement.crf_marginal_confidence import compute_crf_marginals
from model.ner_model_v4_multilingual import NERMultilingualModel


def evaluate_confidence(
    model: NERMultilingualModel,
    extractor: AttentionExtractor,
    tokens: Sequence[str],
    target_token_idx: int,
    target_tag_id: int,
    confidence_mode: str = "crf_marginal",  # 'crf_marginal' or 'emission_softmax'
    zero_embed_indices: Sequence[int] | None = None,
) -> float:
    """
    Evaluate model prediction confidence at target_token_idx for target_tag_id.

    Args:
        model: Frozen NERMultilingualModel instance.
        extractor: AttentionExtractor instance.
        tokens: Sequence of tokens.
        target_token_idx: Index of query token.
        target_tag_id: Target class ID.
        confidence_mode: 'crf_marginal' (true model posterior) or 'emission_softmax' (baseline).
        zero_embed_indices: Optional list of token indices to zero out at the embedding layer.
    """
    w, c, l, mask = extractor.prepare_input_tensors(tokens)
    seq_len = len(tokens)

    with torch.no_grad():
        word_repr = model.word_embedding(w)
        lang_repr = model.lang_embedding(l, seq_len)
        if model.use_char_cnn and model.char_cnn is not None:
            char_repr = model.char_cnn(c)
            combined = torch.cat([word_repr, char_repr, lang_repr], dim=-1)
        else:
            combined = torch.cat([word_repr, lang_repr], dim=-1)

        # Apply zero-embedding perturbation if specified
        if zero_embed_indices:
            for z_idx in zero_embed_indices:
                if 0 <= z_idx < seq_len:
                    combined[0, z_idx, :] = 0.0

        combined_drop = model.concat_dropout(combined)
        lstm_hidden = model.encoder(combined_drop, mask=mask, return_hidden=True)
        if model.use_attention and model.attention is not None:
            attended, _ = model.attention(lstm_hidden, mask=mask)
        else:
            attended = lstm_hidden

        emissions = model.fc_shared(attended)  # (1, seq_len, num_tags)

        if target_tag_id is None:
            target_tag_id = int(emissions[0, target_token_idx].argmax(dim=-1).item())

        if confidence_mode == "emission_softmax":
            probs = torch.softmax(emissions[0, target_token_idx], dim=-1)
            return float(probs[target_tag_id].item())
        elif confidence_mode == "crf_marginal":
            marginals = compute_crf_marginals(model.crf_shared, emissions, mask=mask)
            return float(marginals[0, target_token_idx, target_tag_id].item())
        else:
            raise ValueError(f"Unknown confidence_mode: {confidence_mode}")


def run_symmetric_erasure_trial(
    model: NERMultilingualModel,
    extractor: AttentionExtractor,
    tokens: Sequence[str],
    target_token_idx: int,
    target_tag_id: int,
    attribution_scores: torch.Tensor,
    k_list: Sequence[int] = (1, 2, 3),
    erasure_strategy: str = "zero_embedding",  # 'zero_embedding' or 'unk_token'
    confidence_mode: str = "crf_marginal",     # 'crf_marginal' or 'emission_softmax'
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Run one symmetric erasure trial on a single sentence instance for given attribution scores.
    """
    if rng is None:
        rng = random.Random(42)

    seq_len = len(tokens)
    # 1. Base confidence
    orig_conf = evaluate_confidence(
        model, extractor, tokens, target_token_idx, target_tag_id, confidence_mode=confidence_mode
    )

    # 2. Strict candidate pool: exclude target entity token itself
    valid_candidates = [i for i in range(seq_len) if i != target_token_idx]
    if not valid_candidates:
        return {}

    # Rank context candidates descending by attribution score
    scores_list = attribution_scores.tolist()
    ranked_candidates = sorted(valid_candidates, key=lambda idx: scores_list[idx], reverse=True)

    trial_results: dict[str, Any] = {"orig_conf": orig_conf}

    for k in k_list:
        if k > len(valid_candidates):
            continue

        # (a) Top-k context tokens
        top_k = ranked_candidates[:k]

        # (b) Random-k context tokens (sampled from candidates NOT in top_k)
        non_top_pool = [i for i in valid_candidates if i not in top_k]
        if len(non_top_pool) >= k:
            rand_k = rng.sample(non_top_pool, k)
        else:
            rand_k = rng.sample(valid_candidates, min(k, len(valid_candidates)))

        # Evaluate Top-k drop
        if erasure_strategy == "zero_embedding":
            conf_top = evaluate_confidence(
                model, extractor, tokens, target_token_idx, target_tag_id,
                confidence_mode=confidence_mode, zero_embed_indices=top_k
            )
            conf_rand = evaluate_confidence(
                model, extractor, tokens, target_token_idx, target_tag_id,
                confidence_mode=confidence_mode, zero_embed_indices=rand_k
            )
        elif erasure_strategy == "unk_token":
            tokens_top = list(tokens)
            for idx in top_k:
                tokens_top[idx] = "<UNK>"
            conf_top = evaluate_confidence(
                model, extractor, tokens_top, target_token_idx, target_tag_id,
                confidence_mode=confidence_mode
            )
            tokens_rand = list(tokens)
            for idx in rand_k:
                tokens_rand[idx] = "<UNK>"
            conf_rand = evaluate_confidence(
                model, extractor, tokens_rand, target_token_idx, target_tag_id,
                confidence_mode=confidence_mode
            )
        else:
            raise ValueError(f"Unknown erasure_strategy: {erasure_strategy}")

        drop_top = max(0.0, orig_conf - conf_top)
        drop_rand = max(0.0, orig_conf - conf_rand)
        gap = drop_top - drop_rand
        win = 1 if gap > 0.0 else 0

        trial_results[f"k={k}"] = {
            "top_k_indices": top_k,
            "rand_k_indices": rand_k,
            "conf_top": conf_top,
            "conf_rand": conf_rand,
            "drop_top": drop_top,
            "drop_rand": drop_rand,
            "gap": gap,
            "win": win,
        }

    return trial_results
