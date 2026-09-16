"""
Bidirectional LSTM sequence encoder with tag emission projection.

Encodes embedded token representations into forward and backward contextual states,
projected into tag emission logits for linear-chain CRF decoding.
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
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class BiLSTMEncoder(nn.Module if nn else object):
    """
    Bidirectional LSTM sequence encoder.
    Supports 1-layer and 2-layer configurations with configurable hidden dimensions and dropout.
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_tags: int = 9,
        dropout: float = 0.3,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate BiLSTMEncoder.")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear projection weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: FloatTensor of shape (batch_size, seq_len, input_dim)
            mask: BoolTensor of shape (batch_size, seq_len) with True for valid tokens
        Returns:
            Emission logits of shape (batch_size, seq_len, num_tags)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # Enforce minimum length 1 for safety
            lengths = torch.clamp(lengths, min=1)
            packed = pack_padded_sequence(
                embeddings,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=embeddings.size(1),
            )
        else:
            out, _ = self.lstm(embeddings)

        out = self.output_dropout(out)
        emissions = self.fc(out)
        return emissions
