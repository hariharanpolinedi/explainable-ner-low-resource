"""
Language Conditioning Module for Multilingual NER (Phase 8).

Provides a learned dense embedding vector representing the source language identity,
designed to be concatenated with lexical (word) and morphological (char-CNN) representations
at the input stage of the sequence encoder.
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
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class LanguageEmbedding(nn.Module if nn else object):
    """
    Learned language embedding layer for multilingual conditioning.

    Maps integer language IDs to dense vectors of dimension `embedding_dim`,
    projecting a discrete language tag into the contextual continuous representation.
    """

    def __init__(
        self,
        num_languages: int = 4,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate LanguageEmbedding.")
        super().__init__()

        self.num_languages = num_languages
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_languages,
            embedding_dim=embedding_dim,
        )
        # Initialize randomly with standard normal scaled
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        lang_ids: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for language embedding lookup.

        Args:
            lang_ids: LongTensor of shape (batch_size,) or (batch_size, seq_len).
            seq_len: Optional sequence length to expand (batch_size, 1, embedding_dim)
                     to (batch_size, seq_len, embedding_dim) when lang_ids is 1D.

        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim).
        """
        if lang_ids.dim() == 1:
            # Shape: (batch_size, embedding_dim)
            lang_emb = self.embedding(lang_ids)
            lang_emb = self.dropout(lang_emb)
            if seq_len is not None:
                # Expand across sequence length: (batch_size, seq_len, embedding_dim)
                lang_emb = lang_emb.unsqueeze(1).expand(-1, seq_len, -1)
            return lang_emb
        elif lang_ids.dim() == 2:
            # Shape: (batch_size, seq_len, embedding_dim)
            lang_emb = self.embedding(lang_ids)
            return self.dropout(lang_emb)
        else:
            raise ValueError(f"lang_ids must be 1D or 2D tensor, got shape {tuple(lang_ids.shape)}")
