"""
From-scratch Multi-Head Self-Attention module with residual connection, LayerNorm,
and attention-weight instrumentation for explainable NER (Phase 7).

Architecture:
Input Hidden States H in R^{B x S x D}
-> Multi-Head Scaled Dot-Product Attention:
   Q = H W_Q, K = H W_K, V = H W_V
   Scores = (Q K^T) / sqrt(d_k)
   Padding Mask Application (keys outside mask masked to -1e9, query rows zeroed)
   Attention Weights A = Softmax(Scores)
   Context = A V
   Output Projection = Context W_O
-> Dropout
-> Residual Connection: H + Output
-> LayerNorm(H + Output)
-> Exposes full per-head, per-token attention weight tensor for Phase 13 attribution.
"""

from __future__ import annotations

import math
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
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


class MultiHeadSelfAttentionLayer(nn.Module if nn else object):
    """
    Single-block Multi-Head Self-Attention with residual connection and LayerNorm.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate MultiHeadSelfAttentionLayer.")
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections Q, K, V
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output projection
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Instrumentation: persists attention weights for explainability inspection
        self.last_attention_weights: torch.Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear projections using Xavier uniform."""
        for proj in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with padding mask and attention weight extraction.

        Args:
            hidden_states: FloatTensor of shape (batch_size, seq_len, hidden_dim).
            mask: BoolTensor of shape (batch_size, seq_len) with True for valid tokens.

        Returns:
            Tuple of:
            - output: LayerNorm(H + Attn(H)) of shape (batch_size, seq_len, hidden_dim)
            - attn_weights: Softmax attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        # Shape: (B, S, H, D_k) -> (B, H, S, D_k)
        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Robust Padding Mask Application
        if mask is not None:
            # key_mask: True where key is valid. Shape: (B, 1, 1, S)
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            # Mask out attention to padding tokens: set scores to large negative value
            scores = scores.masked_fill(~key_mask, -1e9)

        # Softmax over key dimension (last dimension)
        attn_weights = F.softmax(scores, dim=-1)

        # If query position is padding, zero out entire attention distribution from that position
        if mask is not None:
            # query_mask: True where query is valid. Shape: (B, 1, S, 1)
            query_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn_weights = attn_weights * query_mask.to(dtype=attn_weights.dtype)

        # Store for explainability retrieval
        self.last_attention_weights = attn_weights.detach()

        # Apply dropout to attention distribution
        attn_dropped = self.attn_dropout(attn_weights)

        # 4. Context aggregation: (B, H, S, D_k)
        context = torch.matmul(attn_dropped, v)

        # 5. Concatenate heads and project: (B, S, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.w_o(context)
        attn_output = self.proj_dropout(attn_output)

        # 6. Residual Connection + LayerNorm: z_i = LayerNorm(h_i + attention_i)
        residual_sum = hidden_states + attn_output
        output = self.layer_norm(residual_sum)

        # If mask provided, zero out padding representations for strict hygiene
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(dtype=output.dtype)

        return output, attn_weights


class MultiHeadSelfAttentionBlock(nn.Module if nn else object):
    """
    Stacked multi-head self-attention module supporting 1 or more sequential blocks.
    Stores and returns attention weights across all blocks.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 1,
        dropout: float = 0.1,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate MultiHeadSelfAttentionBlock.")
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            MultiHeadSelfAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Exposed attribute for Phase 13 explainability access
        self.last_attention_weights: list[torch.Tensor] | torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through sequential self-attention blocks.

        Returns:
            Tuple of:
            - output: FloatTensor of shape (batch_size, seq_len, hidden_dim)
            - all_attn_weights: List of attention weight tensors, one per block.
        """
        curr_states = hidden_states
        all_attn_weights: list[torch.Tensor] = []

        for block in self.blocks:
            curr_states, attn_w = block(curr_states, mask=mask)
            all_attn_weights.append(attn_w)

        # Store on the module for downstream explainability inspection
        self.last_attention_weights = all_attn_weights if self.num_blocks > 1 else all_attn_weights[0]
        return curr_states, all_attn_weights
