"""
Baseline Word-Level BiLSTM-CRF Named Entity Recognition Model (Phase 5).

Architecture:
Word Embedding (randomly initialized, no pretrained weights)
-> BiLSTM Encoder (1 or 2 layers, dropout 0.3)
-> Linear-Chain CRF Layer (negative log-likelihood loss, Viterbi decoding)
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

from model.bilstm_encoder import BiLSTMEncoder
from model.crf_layer import LinearChainCRF
from model.word_embedding import WordEmbedding


class NERBaselineModel(nn.Module if nn else object):
    """
    Complete Word-Level BiLSTM-CRF model for low-resource Named Entity Recognition.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int = 9,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate NERBaselineModel.")
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = WordEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            dropout=dropout,
        )
        self.encoder = BiLSTMEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tags=num_tags,
            dropout=dropout,
        )
        self.crf = LinearChainCRF(num_tags=num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            tags: Optional LongTensor of shape (batch_size, seq_len)
            mask: Optional BoolTensor of shape (batch_size, seq_len)
        Returns:
            Scalar NLL loss if tags is provided, else list of decoded tag ID sequences.
        """
        if mask is None:
            mask = input_ids != 0

        embeds = self.embedding(input_ids)
        emissions = self.encoder(embeds, mask)

        if tags is not None:
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            decoded = self.crf.decode(emissions, mask)
            return decoded
