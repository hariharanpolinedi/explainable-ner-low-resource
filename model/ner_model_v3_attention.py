"""
Named Entity Recognition Model v3 with Multi-Head Self-Attention (Phase 7).

Architecture:
[Word Embedding (100d)]  +  [Char-CNN Word Vector (90d)]
               \\            /
            [Concatenation (190d)]
                     |
            [Concat Dropout (0.3)]
                     |
        [BiLSTM Sequence Encoder (128 hidden x 2 = 256d)]
                     |
       [Multi-Head Self-Attention Block (4 heads, 1 block)]
                     |
            [Residual + LayerNorm (256d)]
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
from model.self_attention import MultiHeadSelfAttentionBlock
from model.word_embedding import WordEmbedding


class NERAttentionModel(nn.Module if nn else object):
    """
    State-of-the-art Low-Resource Named Entity Recognition Model integrating
    Word Embeddings, Character-level CNN, BiLSTM Contextual Encoder,
    Multi-Head Self-Attention with Residual LayerNorm, and Linear-Chain CRF.
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
        num_heads: int = 4,
        num_attention_blocks: int = 1,
        attention_dropout: float = 0.1,
        word_dropout: float = 0.3,
        char_dropout: float = 0.3,
        concat_dropout: float = 0.3,
        bilstm_dropout: float = 0.3,
        word_pad_idx: int = 0,
        char_pad_idx: int = 0,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate NERAttentionModel.")
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
        self.num_heads = num_heads
        self.num_attention_blocks = num_attention_blocks

        # 1. Word embedding layer (reused from Phase 5)
        self.word_embedding = WordEmbedding(
            vocab_size=word_vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_pad_idx,
            dropout=word_dropout,
        )

        # 2. Character-CNN encoder (reused from Phase 6)
        self.char_cnn = CharCNNWordEncoder(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            filter_widths=filter_widths,
            filters_per_width=filters_per_width,
            dropout=char_dropout,
            padding_idx=char_pad_idx,
        )

        # 3. Concatenation and dropout
        self.char_repr_dim = self.char_cnn.output_dim
        self.combined_dim = word_embedding_dim + self.char_repr_dim
        self.concat_dropout = nn.Dropout(concat_dropout)

        # 4. BiLSTM sequence encoder (reused from Phase 5/6)
        # BiLSTM produces 2 * hidden_dim contextual representations
        self.bilstm_out_dim = hidden_dim * 2
        self.encoder = BiLSTMEncoder(
            input_dim=self.combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tags=num_tags,
            dropout=bilstm_dropout,
        )

        # 5. Multi-Head Self-Attention Block with residual LayerNorm (Phase 7 core addition)
        self.attention = MultiHeadSelfAttentionBlock(
            hidden_dim=self.bilstm_out_dim,
            num_heads=num_heads,
            num_blocks=num_attention_blocks,
            dropout=attention_dropout,
        )

        # 6. Linear-Chain CRF (reused from Phase 5)
        self.crf = LinearChainCRF(num_tags=num_tags)

    @property
    def last_attention_weights(self) -> Any:
        """Convenience property retrieving attention weights from the most recent forward pass."""
        return self.attention.last_attention_weights

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Any:
        """
        Forward pass with attention extraction.

        Args:
            word_ids: LongTensor of shape (batch_size, seq_len) with token IDs.
            char_ids: LongTensor of shape (batch_size, seq_len, max_word_len) with char IDs.
            tags: Optional LongTensor of shape (batch_size, seq_len) with ground-truth BIO tag IDs.
            mask: Optional BoolTensor of shape (batch_size, seq_len) with True for valid tokens.
            return_attention: If True, returns tuple (result, attention_weights).

        Returns:
            - If tags is provided: NLL loss scalar (or (loss, attn_weights) if return_attention=True).
            - If tags is None: list of decoded tag sequences (or (paths, attn_weights) if return_attention=True).
        """
        if mask is None:
            mask = word_ids != 0

        # Step 1: Subword and lexical embeddings
        word_repr = self.word_embedding(word_ids)
        char_repr = self.char_cnn(char_ids)
        combined = torch.cat([word_repr, char_repr], dim=-1)
        combined = self.concat_dropout(combined)

        # Step 2: BiLSTM recurrent contextual encoding (obtain hidden states before projection)
        lstm_hidden = self.encoder(combined, mask=mask, return_hidden=True)

        # Step 3: Multi-Head Self-Attention + Residual Connection + LayerNorm
        attended_states, attn_weights = self.attention(lstm_hidden, mask=mask)

        # Step 4: Tag emission projection
        emissions = self.encoder.fc(attended_states)

        # Step 5: CRF decoding or loss computation
        if tags is not None:
            loss = self.crf(emissions, tags, mask=mask)
            if return_attention:
                return loss, attn_weights
            return loss
        else:
            paths = self.crf.decode(emissions, mask=mask)
            if return_attention:
                return paths, attn_weights
            return paths
