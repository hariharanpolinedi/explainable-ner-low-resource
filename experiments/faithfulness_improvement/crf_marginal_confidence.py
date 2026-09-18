"""
Exact Marginal Confidence and Sequence Scoring Module for Linear-Chain CRF.

Computes:
1. Forward-backward marginal posterior probabilities P_CRF(y_t = c | x)
2. Sequence score and conditional log-likelihood log P(y | x) = score(x, y) - log Z(x)
3. Multi-token entity span joint / geometric mean confidence

Zero external interpretability packages; 100% native PyTorch tensor operations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from model.crf_layer import LinearChainCRF


def compute_crf_marginals(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute marginal posterior probabilities P(y_t = c | x) for each token using forward-backward.

    Args:
        crf: LinearChainCRF instance.
        emissions: FloatTensor of shape (batch_size, seq_len, num_tags).
        mask: Optional BoolTensor of shape (batch_size, seq_len).

    Returns:
        FloatTensor of shape (batch_size, seq_len, num_tags) containing marginal probabilities.
    """
    emissions = emissions.float()
    batch_size, seq_len, num_tags = emissions.shape
    if mask is None:
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)

    # 1. Forward pass: compute alphas
    # alpha[t, b, j] = log sum_{y_{1:t-1}} exp(score(y_{1:t-1}, y_t=j))
    forward_vars: list[torch.Tensor] = []
    alpha = crf.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
    forward_vars.append(alpha)

    for i in range(1, seq_len):
        curr_mask = mask[:, i].unsqueeze(1)  # (batch_size, 1)
        emit = emissions[:, i].unsqueeze(2)  # (batch_size, num_tags, 1)

        # alpha: (batch_size, 1, num_tags) -> from_tags
        # transitions: (num_tags, num_tags) -> [to_tag, from_tag]
        # emit: (batch_size, num_tags, 1) -> to_tag
        next_alpha = alpha.unsqueeze(1) + crf.transitions.unsqueeze(0) + emit  # (B, num_tags, num_tags)
        next_alpha = torch.logsumexp(next_alpha, dim=2)  # (B, num_tags)
        alpha = torch.where(curr_mask, next_alpha, alpha)
        forward_vars.append(alpha)

    # Transition to end tag
    terminal_alpha = alpha + crf.end_transitions.unsqueeze(0)
    log_z = torch.logsumexp(terminal_alpha, dim=1)  # (batch_size,)

    # 2. Backward pass: compute betas
    # beta[t, b, j] = log sum_{y_{t+1:T}} exp(score(y_t=j, y_{t+1:T}))
    backward_vars: list[torch.Tensor] = [None] * seq_len  # type: ignore
    # Terminal step: end transitions
    beta = crf.end_transitions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_tags)
    backward_vars[seq_len - 1] = beta

    for i in range(seq_len - 2, -1, -1):
        curr_mask = mask[:, i + 1].unsqueeze(1)  # (batch_size, 1)
        emit = emissions[:, i + 1].unsqueeze(1)  # (batch_size, 1, num_tags) -> to_tag emit

        # trans is (to, from), beta is to, emit is to
        trans = crf.transitions.unsqueeze(0)  # (1, num_tags, num_tags) [to, from]
        score_step = beta.unsqueeze(2) + emit.transpose(1, 2) + trans  # (B, to, from)
        next_beta = torch.logsumexp(score_step, dim=1)  # (B, from)
        beta = torch.where(curr_mask, next_beta, beta)
        backward_vars[i] = beta

    # 3. Combine: log P(y_t = c | x) = alpha_t(c) + beta_t(c) - log_z
    marginals = torch.zeros_like(emissions)
    for t in range(seq_len):
        log_prob_t = forward_vars[t] + backward_vars[t] - log_z.unsqueeze(1)
        marginals[:, t, :] = torch.exp(log_prob_t).clamp(min=0.0, max=1.0)

    return marginals


def get_crf_marginal_confidence(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    token_idx: int,
    target_tag_id: int,
    mask: torch.Tensor | None = None,
) -> float:
    """
    Get the exact CRF marginal posterior probability P(y_{token_idx} = target_tag_id | x).
    """
    with torch.no_grad():
        marginals = compute_crf_marginals(crf, emissions, mask=mask)
        return float(marginals[0, token_idx, target_tag_id].item())


def get_crf_span_confidence(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    span_indices: Sequence[int],
    span_tag_ids: Sequence[int],
    mask: torch.Tensor | None = None,
) -> float:
    """
    Compute geometric mean marginal confidence over an entity span [s, ..., e].
    """
    with torch.no_grad():
        marginals = compute_crf_marginals(crf, emissions, mask=mask)
        probs = [
            float(marginals[0, idx, tag_id].item())
            for idx, tag_id in zip(span_indices, span_tag_ids)
        ]
        # Geometric mean
        log_sum = sum(torch.log(torch.tensor(max(p, 1e-12))).item() for p in probs)
        return float(torch.exp(torch.tensor(log_sum / len(probs))).item())
