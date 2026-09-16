"""
Named Entity Recognition Model v2 with Character-Level CNN Word Representation (Phase 6).

Architecture:
[Word Embedding (100d)]  +  [Char-CNN Word Vector (90d)]
               \\            /
            [Concatenation (190d)]
                     |
            [Concat Dropout (0.3)]
                     |
        [BiLSTM Sequence Encoder (128 hidden)]
                     |
         [Linear Emission Projection (9 tags)]
                     |
             [Linear-Chain CRF]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

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

from model.bilstm_encoder import BiLSTMEncoder
from model.char_cnn import CharCNNWordEncoder
from model.crf_layer import LinearChainCRF
from model.word_embedding import WordEmbedding


class NERCharCNNModel(nn.Module if nn else object):
    """
    Combined Word-and-Character BiLSTM-CRF model for low-resource Named Entity Recognition.
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_tags: int = 9,
        word_embedding_dim: int = 100,
        char_embedding_dim: int = 50,
        filter_widths: Sequence[int] = (3, 4, 5),
        filters_per_width: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        word_dropout: float = 0.3,
        char_dropout: float = 0.3,
        concat_dropout: float = 0.3,
        bilstm_dropout: float = 0.3,
        word_pad_idx: int = 0,
        char_pad_idx: int = 0,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate NERCharCNNModel.")
        super().__init__()

        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.num_tags = num_tags
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.filter_widths = tuple(filter_widths)
        self.filters_per_width = filters_per_width
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Word embedding layer (reused from Phase 5)
        self.word_embedding = WordEmbedding(
            vocab_size=word_vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_pad_idx,
            dropout=word_dropout,
        )

        # 2. Character-CNN encoder
        self.char_cnn = CharCNNWordEncoder(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            filter_widths=filter_widths,
            filters_per_width=filters_per_width,
            dropout=char_dropout,
            padding_idx=char_pad_idx,
        )

        # 3. Concatenated representation dimension
        self.char_repr_dim = self.char_cnn.output_dim
        self.combined_dim = word_embedding_dim + self.char_repr_dim
        self.concat_dropout = nn.Dropout(concat_dropout)

        # 4. BiLSTM encoder (reused from Phase 5)
        self.encoder = BiLSTMEncoder(
            input_dim=self.combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tags=num_tags,
            dropout=bilstm_dropout,
        )

        # 5. Linear-Chain CRF layer (reused from Phase 5)
        self.crf = LinearChainCRF(num_tags=num_tags)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        """
        Forward pass.

        Args:
            word_ids: Tensor of shape (batch_size, seq_len) with token IDs.
            char_ids: Tensor of shape (batch_size, seq_len, max_word_len) with char IDs.
            tags: Optional ground-truth tag sequence (batch_size, seq_len).
            mask: Optional boolean mask (batch_size, seq_len) indicating non-pad tokens.

        Returns:
            Scalar NLL loss if tags is provided, else list of predicted tag sequences.
        """
        if mask is None:
            mask = word_ids != 0

        # Word representation: (batch_size, seq_len, word_embedding_dim)
        word_repr = self.word_embedding(word_ids)

        # Char-CNN representation: (batch_size, seq_len, char_repr_dim)
        char_repr = self.char_cnn(char_ids)

        # Combined representation: (batch_size, seq_len, combined_dim)
        combined = torch.cat([word_repr, char_repr], dim=-1)
        combined = self.concat_dropout(combined)

        # BiLSTM contextual emissions: (batch_size, seq_len, num_tags)
        emissions = self.encoder(combined, mask=mask)

        # CRF loss or decoding
        if tags is not None:
            return self.crf(emissions, tags, mask=mask)
        else:
            return self.crf.decode(emissions, mask=mask)
