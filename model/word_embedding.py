"""
Randomly initialized word embedding layer for low-resource NER.

Strict project constraint:
NO pretrained embeddings (Word2Vec, GloVe, FastText) or external checkpoints.
Embeddings are randomly initialized and trained end-to-end purely from scratch.
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
except ImportError:
    # Handled when torch finishes installation
    torch = None  # type: ignore
    nn = None  # type: ignore


class WordEmbedding(nn.Module if nn else object):
    """
    Randomly initialized word embedding table.
    Learned end-to-end strictly from NER training signal.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        padding_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate WordEmbedding.")
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights uniformly within [-bound, bound] where bound = sqrt(3 / dim)."""
        bound = math.sqrt(3.0 / self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bound, bound)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        embeds = self.embedding(input_ids)
        return self.dropout(embeds)
