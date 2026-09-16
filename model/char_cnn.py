"""
Character-level CNN word encoder module for low-resource NER (Phase 6).

Architecture:
Character/Grapheme Embedding (50d, randomly initialized, no pretrained weights)
-> 1D Convolutions with multiple filter widths (e.g. [3, 4, 5], 30 filters per width)
-> ReLU activation
-> Max-over-time pooling per filter width
-> Concatenation of pooled features into a single char-derived word vector (90d)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

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


class CharCNNWordEncoder(nn.Module if nn else object):
    """
    Character/grapheme CNN word encoder.

    Extracts morphological and subword features from character index sequences,
    producing a fixed-dimension character-derived representation per word.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 50,
        filter_widths: Sequence[int] = (3, 4, 5),
        filters_per_width: int = 30,
        dropout: float = 0.3,
        padding_idx: int = 0,
        output_dim: int | None = None,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate CharCNNWordEncoder.")
        super().__init__()

        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.filter_widths = tuple(filter_widths)
        self.filters_per_width = filters_per_width
        self.total_conv_dim = len(self.filter_widths) * filters_per_width
        self.output_dim = output_dim if output_dim is not None else self.total_conv_dim
        self.max_filter_width = max(self.filter_widths) if self.filter_widths else 3

        # Randomly initialized character embedding (no pretrained vectors)
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=padding_idx,
        )
        # Standard normal initialization scaled by embedding dim
        nn.init.uniform_(self.char_embedding.weight, -0.1, 0.1)
        if padding_idx is not None and 0 <= padding_idx < char_vocab_size:
            with torch.no_grad():
                self.char_embedding.weight[padding_idx].fill_(0.0)

        self.char_dropout = nn.Dropout(dropout)

        # 1D Convolutions per filter width
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=filters_per_width,
                kernel_size=k,
                bias=True,
            )
            for k in self.filter_widths
        ])

        # Optional projection layer if configured output_dim differs from total_conv_dim
        if self.output_dim != self.total_conv_dim:
            self.projection = nn.Linear(self.total_conv_dim, self.output_dim)
        else:
            self.projection = None

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            char_ids: Tensor of shape (batch_size, seq_len, max_word_len) containing char IDs.

        Returns:
            Tensor of shape (batch_size, seq_len, output_dim) containing char-derived word vectors.
        """
        batch_size, seq_len, word_len = char_ids.shape

        # Flatten batch and sequence dimensions: (batch_size * seq_len, word_len)
        flat_chars = char_ids.view(batch_size * seq_len, word_len)

        # Ensure word length is at least max_filter_width to prevent Conv1d underflow
        if word_len < self.max_filter_width:
            pad_amount = self.max_filter_width - word_len
            flat_chars = F.pad(flat_chars, (0, pad_amount), value=0)
            word_len = self.max_filter_width

        # Character embedding: (B * S, word_len, char_emb_dim)
        char_embeds = self.char_embedding(flat_chars)
        char_embeds = self.char_dropout(char_embeds)

        # Transpose to channel-first for Conv1d: (B * S, char_emb_dim, word_len)
        char_embeds = char_embeds.transpose(1, 2)

        # Apply convolutions and max-over-time pooling
        pooled_outputs = []
        for conv in self.convs:
            # conv output: (B * S, filters_per_width, word_len - kernel_size + 1)
            convolved = F.relu(conv(char_embeds))
            # max pool over time: (B * S, filters_per_width)
            pooled, _ = convolved.max(dim=-1)
            pooled_outputs.append(pooled)

        # Concatenate pooled features: (B * S, total_conv_dim)
        char_repr = torch.cat(pooled_outputs, dim=-1)

        if self.projection is not None:
            char_repr = self.projection(char_repr)

        # Reshape back to (batch_size, seq_len, output_dim)
        return char_repr.view(batch_size, seq_len, self.output_dim)
