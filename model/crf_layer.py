"""
Linear-chain Conditional Random Field (CRF) layer implemented from scratch in PyTorch.

Provides:
1. Exact negative log-likelihood (NLL) loss computation via the forward-backward algorithm (log-sum-exp).
2. Global Viterbi decoding for maximum-likelihood sequence inference.
3. Masking support for variable-length padded sequences.
4. Numerical stability guarantees (computed in float32 to prevent FP16 underflow).

100% pure algorithm implementation with zero pretrained weights or external dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class LinearChainCRF(nn.Module if nn else object):
    """
    Linear-chain Conditional Random Field.
    Learns tag transition dynamics and enforces valid global sequence decodings.
    """

    def __init__(self, num_tags: int) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate LinearChainCRF.")
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j]: transition score FROM tag j TO tag i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize transition parameters uniformly."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood (NLL) loss.
        loss = log(Z(x)) - score(x, y)

        Args:
            emissions: FloatTensor of shape (batch_size, seq_len, num_tags)
            tags: LongTensor of shape (batch_size, seq_len)
            mask: BoolTensor of shape (batch_size, seq_len) with True for valid tokens
            reduction: 'none', 'sum', or 'mean'
        Returns:
            Scalar loss tensor (or per-example tensor if reduction='none')
        """
        # Ensure float32 for CRF numerical stability under AMP
        emissions = emissions.float()
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        # 1. Numerator: Gold sequence score
        gold_score = self._compute_score(emissions, tags, mask)
        # 2. Denominator: Partition function log(Z)
        forward_score = self._compute_forward_score(emissions, mask)

        nll = forward_score - gold_score

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        elif reduction == "none":
            return nll
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score for the target tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            curr_mask = mask[:, i]
            curr_tags = tags[:, i]
            prev_tags = tags[:, i - 1]

            # Transition score from prev_tags to curr_tags
            trans_score = self.transitions[curr_tags, prev_tags]
            # Emission score for curr_tags
            emit_score = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze(1)

            step_score = trans_score + emit_score
            score = score + step_score * curr_mask.float()

        # Add transition to end tag for the last valid token
        # Find last token index for each sequence in batch
        last_valid_indices = mask.long().sum(dim=1) - 1
        last_valid_indices = torch.clamp(last_valid_indices, min=0)
        last_tags = tags.gather(1, last_valid_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        return score

    def _compute_forward_score(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the partition function log(Z) using the forward algorithm."""
        batch_size, seq_len, num_tags = emissions.shape

        # Initial forward state: start_transitions + first emission
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)

        for i in range(1, seq_len):
            curr_mask = mask[:, i].unsqueeze(1)  # (batch_size, 1)
            emit = emissions[:, i].unsqueeze(2)  # (batch_size, num_tags, 1)

            # Broadcast for all transitions:
            # alpha: (batch_size, 1, num_tags) -> from_tags
            # transitions: (num_tags, num_tags) -> to_tags, from_tags
            # emit: (batch_size, num_tags, 1) -> to_tags
            alpha_broadcast = alpha.unsqueeze(1)  # (batch_size, 1, num_tags)
            trans_broadcast = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)

            # Score of transitioning from 'from_tag' to 'to_tag' plus emitting 'to_tag'
            next_alpha = alpha_broadcast + trans_broadcast + emit  # (batch_size, num_tags, num_tags)
            next_alpha = torch.logsumexp(next_alpha, dim=2)  # (batch_size, num_tags)

            # Update alpha if valid token, keep previous alpha if masked
            alpha = torch.where(curr_mask, next_alpha, alpha)

        # Transition to end tag
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Viterbi decoding for maximum-likelihood sequence inference.
        Returns the best tag ID path for each example in the batch.
        """
        emissions = emissions.float()
        batch_size, seq_len, num_tags = emissions.shape
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)

        best_paths: list[list[int]] = []

        for b in range(batch_size):
            seq_l = int(mask[b].sum().item())
            if seq_l == 0:
                best_paths.append([])
                continue

            # Forward Viterbi
            b_emissions = emissions[b, :seq_l]  # (seq_l, num_tags)
            viterbi_vars = self.start_transitions + b_emissions[0]  # (num_tags,)
            backpointers: list[torch.Tensor] = []

            for i in range(1, seq_l):
                # viterbi_vars: (num_tags,) -> from_tags
                # transitions: (num_tags, num_tags) -> [to_tag, from_tag]
                next_tag_var = viterbi_vars.unsqueeze(0) + self.transitions  # (num_tags, num_tags)
                best_prev, best_tag_ids = torch.max(next_tag_var, dim=1)  # (num_tags,)
                viterbi_vars = best_prev + b_emissions[i]
                backpointers.append(best_tag_ids)

            # Add end transition
            terminal_vars = viterbi_vars + self.end_transitions
            best_last_tag = torch.argmax(terminal_vars).item()

            # Backtrack
            best_path = [best_last_tag]
            for bp in reversed(backpointers):
                best_last_tag = bp[best_last_tag].item()
                best_path.append(best_last_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
