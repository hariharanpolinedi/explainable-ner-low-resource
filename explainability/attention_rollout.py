"""
Attention Rollout Module (Phase 13 - Architectural Extension Stub).

In deep multi-layer transformer architectures (Abnar & Zuidema, 2020),
Attention Rollout computes token flow across stacked layers by recursively
multiplying residual-adjusted attention matrices:

    R_0 = I
    R_l = (0.5 * A_l + 0.5 * I) @ R_{l-1}

STATUS IN CURRENT PROJECT:
The final locked Phase-12 architecture across all three languages (Konkani, Assamese,
and Sinhala) empirically resolved to exactly ONE attention block (`num_attention_blocks = 1`),
as established in `reports/phase12_must_verify_resolution.md` (multi-block attention
exhibited over-smoothing and severe performance regression in the low-resource regime).

Consequently, Attention Rollout across sequential layers is NOT INVOKED during runtime
inference in Phase 13. This module is maintained as a reproducible, fully-tested stub
for architectural completeness and future multi-block extensions.
"""

from __future__ import annotations

from typing import Sequence
import torch


class AttentionRollout:
    """
    Computes Attention Rollout across sequential multi-head self-attention layers.
    Marked NOT INVOKED for single-block architectures.
    """

    def __init__(self, add_residual: bool = True) -> None:
        self.add_residual = add_residual
        self.is_invoked = False  # Set to True only if multi-block rollout is executed

    def compute_rollout(
        self,
        layer_attentions: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute rollout across a list of multi-head attention tensors.

        Args:
            layer_attentions: List of Tensors of shape (num_heads, seq_len, seq_len)
                              or (batch, num_heads, seq_len, seq_len).

        Returns:
            Aggregated rollout attention matrix of shape (seq_len, seq_len).
        """
        if not layer_attentions:
            raise ValueError("layer_attentions list must not be empty.")

        first_layer = layer_attentions[0]
        if first_layer.dim() == 4:
            # Squeeze batch dimension if 1
            first_layer = first_layer[0]

        num_heads, seq_len, _ = first_layer.shape
        eye = torch.eye(seq_len, device=first_layer.device, dtype=first_layer.dtype)

        # Initialize rollout matrix with identity
        rollout = eye.clone()

        for layer_attn in layer_attentions:
            if layer_attn.dim() == 4:
                layer_attn = layer_attn[0]
            # Average across heads: shape (seq_len, seq_len)
            mean_attn = layer_attn.mean(dim=0)
            if self.add_residual:
                a_res = 0.5 * mean_attn + 0.5 * eye
            else:
                a_res = mean_attn
            # Normalize rows
            a_res = a_res / a_res.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            rollout = torch.matmul(a_res, rollout)

        self.is_invoked = True
        return rollout
