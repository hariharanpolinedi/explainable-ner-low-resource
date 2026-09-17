"""
Named Entity Recognition Model v4 with Multilingual Language Conditioning (Phase 8).

Architecture:
[Word Embedding (100d)]  +  [Char-CNN (90d)]  +  [Language Embedding (16d)]
               \\                   |                   /
                            [Concatenation (206d)]
                                     |
                            [Concat Dropout (0.3)]
                                     |
                    [BiLSTM Sequence Encoder (256d)]
                                     |
               [Multi-Head Self-Attention Block (4 heads, 1 block)]
                                     |
                            [Residual + LayerNorm (256d)]
                                     |
            [CRF Head: Shared Linear+CRF  OR  Per-Language Linear+CRF]
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
from model.language_conditioning import LanguageEmbedding
from model.self_attention import MultiHeadSelfAttentionBlock
from model.word_embedding import WordEmbedding


class NERMultilingualModel(nn.Module if nn else object):
    """
    Multilingual NER Model with language-conditioning embeddings, shared
    Char-CNN + BiLSTM + Self-Attention encoder backbone, and configurable
    (shared vs per-language) CRF sequence labeling heads.
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_tags: int = 9,
        num_languages: int = 3,
        word_embedding_dim: int = 100,
        char_embedding_dim: int = 50,
        lang_embedding_dim: int = 16,
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
        crf_config: str = "shared",
        word_pad_idx: int = 0,
        char_pad_idx: int = 0,
        use_char_cnn: bool = True,
        use_attention: bool = True,
        use_crf: bool = True,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch must be installed to instantiate NERMultilingualModel.")
        super().__init__()

        if crf_config not in {"shared", "per_language"}:
            raise ValueError(f"crf_config must be 'shared' or 'per_language', got '{crf_config}'")

        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.num_tags = num_tags
        self.num_languages = num_languages
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.lang_embedding_dim = lang_embedding_dim
        self.filter_widths = tuple(filter_widths)
        self.filters_per_width = filters_per_width
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_attention_blocks = num_attention_blocks
        self.crf_config = crf_config
        self.use_char_cnn = use_char_cnn
        self.use_attention = use_attention
        self.use_crf = use_crf

        # 1. Word lexical embedding
        self.word_embedding = WordEmbedding(
            vocab_size=word_vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_pad_idx,
            dropout=word_dropout,
        )

        # 2. Character-level CNN encoder (optional ablation)
        if self.use_char_cnn:
            self.char_cnn = CharCNNWordEncoder(
                char_vocab_size=char_vocab_size,
                char_embedding_dim=char_embedding_dim,
                filter_widths=filter_widths,
                filters_per_width=filters_per_width,
                dropout=char_dropout,
                padding_idx=char_pad_idx,
            )
            self.char_repr_dim = self.char_cnn.output_dim
        else:
            self.char_cnn = None
            self.char_repr_dim = 0

        # 3. Language conditioning embedding
        self.lang_embedding = LanguageEmbedding(
            num_languages=num_languages,
            embedding_dim=lang_embedding_dim,
            dropout=0.1,
        )

        # 4. Concatenation and dropout: word + (char_cnn if enabled) + lang
        self.combined_dim = word_embedding_dim + self.char_repr_dim + lang_embedding_dim
        self.concat_dropout = nn.Dropout(concat_dropout)

        # 5. Shared BiLSTM contextual sequence encoder
        self.bilstm_out_dim = hidden_dim * 2
        self.encoder = BiLSTMEncoder(
            input_dim=self.combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tags=num_tags,
            dropout=bilstm_dropout,
        )

        # 6. Shared Multi-Head Self-Attention Block (optional ablation)
        if self.use_attention:
            self.attention = MultiHeadSelfAttentionBlock(
                hidden_dim=self.bilstm_out_dim,
                num_heads=num_heads,
                num_blocks=num_attention_blocks,
                dropout=attention_dropout,
            )
        else:
            self.attention = None

        # 7. Configurable Output Head (CRF vs Softmax, Shared vs Per-Language)
        if self.crf_config == "shared":
            self.fc_shared = nn.Linear(self.bilstm_out_dim, num_tags)
            self.crf_shared = LinearChainCRF(num_tags=num_tags) if self.use_crf else None
            self.fc_heads = None
            self.crf_heads = None
        else:
            self.fc_shared = None
            self.crf_shared = None
            self.fc_heads = nn.ModuleList([
                nn.Linear(self.bilstm_out_dim, num_tags) for _ in range(num_languages)
            ])
            self.crf_heads = (
                nn.ModuleList([LinearChainCRF(num_tags=num_tags) for _ in range(num_languages)])
                if self.use_crf
                else None
            )

    @property
    def last_attention_weights(self) -> Any:
        """Retrieve stored attention weights from the most recent forward pass."""
        if self.attention is not None:
            return self.attention.last_attention_weights
        return None

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor | None = None,
        lang_ids: torch.Tensor | None = None,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Any:
        """
        Forward pass with multilingual language conditioning and sequence decoding.

        Args:
            word_ids: LongTensor of shape (batch_size, seq_len)
            char_ids: Optional LongTensor of shape (batch_size, seq_len, max_word_len)
            lang_ids: Optional LongTensor of shape (batch_size,) indicating source language ID
            tags: Optional LongTensor of shape (batch_size, seq_len)
            mask: Optional BoolTensor of shape (batch_size, seq_len)
            return_attention: If True, also returns attention weights

        Returns:
            - If tags is provided: scalar loss (or (loss, attn_weights))
            - If tags is None: list of decoded paths (or (paths, attn_weights))
        """
        if mask is None:
            mask = word_ids != 0

        batch_size, seq_len = word_ids.shape
        if lang_ids is None:
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=word_ids.device)

        # Step 1: Lexical, Morphological, and Language Embeddings
        word_repr = self.word_embedding(word_ids)               # (B, T, word_dim)
        lang_repr = self.lang_embedding(lang_ids, seq_len)       # (B, T, lang_dim)

        if self.use_char_cnn and self.char_cnn is not None:
            if char_ids is None:
                raise ValueError("char_ids must be provided when use_char_cnn is True")
            char_repr = self.char_cnn(char_ids)                 # (B, T, char_dim)
            combined = torch.cat([word_repr, char_repr, lang_repr], dim=-1)
        else:
            combined = torch.cat([word_repr, lang_repr], dim=-1)

        # Step 2: Input Dropout
        combined = self.concat_dropout(combined)

        # Step 3: Shared Contextual BiLSTM Encoding (return unprojected hidden)
        lstm_hidden = self.encoder(combined, mask=mask, return_hidden=True)

        # Step 4: Shared Multi-Head Self-Attention + Residual LayerNorm (or bypass)
        if self.use_attention and self.attention is not None:
            attended_states, attn_weights = self.attention(lstm_hidden, mask=mask)
        else:
            attended_states = lstm_hidden
            attn_weights = None

        # Step 5: Sequence Tagging (CRF vs Softmax)
        if self.use_crf:
            if self.crf_config == "shared":
                emissions = self.fc_shared(attended_states)
                if tags is not None:
                    loss = self.crf_shared(emissions, tags, mask=mask)
                    if return_attention:
                        return loss, attn_weights
                    return loss
                else:
                    paths = self.crf_shared.decode(emissions, mask=mask)
                    if return_attention:
                        return paths, attn_weights
                    return paths
            else:
                # Per-Language CRF Head Routing
                if tags is not None:
                    total_loss = torch.tensor(0.0, device=word_ids.device)
                    num_sub_batches = 0
                    for l_id in range(self.num_languages):
                        idx = (lang_ids == l_id).nonzero(as_tuple=True)[0]
                        if len(idx) == 0:
                            continue
                        sub_emissions = self.fc_heads[l_id](attended_states[idx])
                        sub_tags = tags[idx]
                        sub_mask = mask[idx]
                        loss_l = self.crf_heads[l_id](sub_emissions, sub_tags, mask=sub_mask)
                        total_loss = total_loss + loss_l * len(idx)
                        num_sub_batches += len(idx)

                    norm_loss = total_loss / max(num_sub_batches, 1)
                    if return_attention:
                        return norm_loss, attn_weights
                    return norm_loss
                else:
                    # Decoding per sample
                    paths: list[list[int]] = [None] * batch_size  # type: ignore
                    for l_id in range(self.num_languages):
                        idx = (lang_ids == l_id).nonzero(as_tuple=True)[0]
                        if len(idx) == 0:
                            continue
                        sub_emissions = self.fc_heads[l_id](attended_states[idx])
                        sub_mask = mask[idx]
                        sub_paths = self.crf_heads[l_id].decode(sub_emissions, mask=sub_mask)
                        for k, orig_idx in enumerate(idx.tolist()):
                            paths[orig_idx] = sub_paths[k]

                    if return_attention:
                        return paths, attn_weights
                    return paths

        else:
            # Softmax cross-entropy (Ablation c: No CRF)
            if self.crf_config == "shared":
                logits = self.fc_shared(attended_states)
            else:
                logits = torch.zeros(batch_size, seq_len, self.num_tags, device=word_ids.device)
                for l_id in range(self.num_languages):
                    idx = (lang_ids == l_id).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        logits[idx] = self.fc_heads[l_id](attended_states[idx])

            if tags is not None:
                # Compute Cross-Entropy loss over valid unmasked tokens
                active_mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_tags)[active_mask]
                active_tags = tags.view(-1)[active_mask]
                loss = nn.functional.cross_entropy(active_logits, active_tags)
                if return_attention:
                    return loss, attn_weights
                return loss
            else:
                preds = logits.argmax(dim=-1)
                paths = []
                for i in range(batch_size):
                    seq_m = mask[i]
                    paths.append(preds[i][seq_m].tolist())
                if return_attention:
                    return paths, attn_weights
                return paths
