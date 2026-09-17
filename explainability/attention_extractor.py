"""
Formalized Attention Extraction Module (Phase 13).

Provides a clean, reusable interface for extracting multi-head self-attention
weight tensors and per-token contextual attention distributions from frozen
NERMultilingualModel checkpoints.

Strictly operates on local tensors; NO external pretrained packages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from preprocessing.tokenizer import ScriptTokenizer
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID


class AttentionExtractor:
    """
    Extracts and organizes attention weight matrices from frozen NER models.
    """

    def __init__(
        self,
        model: nn.Module,
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        lang: str,
        lang_id: int,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.lang = lang
        self.lang_id = lang_id
        self.device = device or next(model.parameters()).device
        self.tokenizer = ScriptTokenizer(lang)
        self.model.eval()

    def prepare_input_tensors(
        self,
        tokens: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert tokens into padded word, character, language, and mask tensors.
        """
        seq_len = len(tokens)
        word_ids = [self.word_to_id.get(t, 1) for t in tokens]
        char_ids_list: list[list[int]] = []
        for t in tokens:
            graphemes = self.tokenizer.tokenize_characters(t)
            c_ids = [self.char_to_id.get(g, 1) for g in graphemes]
            if not c_ids:
                c_ids = [1]
            char_ids_list.append(c_ids)

        max_word_len = max(len(c) for c in char_ids_list)
        char_tensor = torch.zeros(1, seq_len, max_word_len, dtype=torch.long, device=self.device)
        for i, c_ids in enumerate(char_ids_list):
            char_tensor[0, i, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)

        word_tensor = torch.tensor([word_ids], dtype=torch.long, device=self.device)
        lang_tensor = torch.tensor([self.lang_id], dtype=torch.long, device=self.device)
        mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

        return word_tensor, char_tensor, lang_tensor, mask

    def extract_attention(
        self,
        tokens: Sequence[str],
    ) -> dict[str, Any]:
        """
        Run a forward pass on tokens and return decoded tags, raw emissions,
        and per-head attention weight tensors.

        Returns:
            Dict containing:
            - 'tokens': input token list
            - 'predicted_tags': list of BIO tag strings
            - 'predicted_tag_ids': list of tag integer IDs
            - 'attention_weights': FloatTensor of shape (num_heads, seq_len, seq_len)
            - 'head_avg_attention': FloatTensor of shape (seq_len, seq_len)
            - 'emissions': FloatTensor of shape (seq_len, num_tags)
        """
        w, c, l, mask = self.prepare_input_tensors(tokens)

        with torch.no_grad():
            paths, attn_weights_list = self.model(
                w, c, lang_ids=l, mask=mask, return_attention=True
            )

        predicted_path = paths[0]
        predicted_tags = [ID_TO_TAG.get(tid, "O") for tid in predicted_path]

        # attn_weights is shape (batch_size, num_heads, seq_len, seq_len)
        if isinstance(attn_weights_list, list):
            raw_attn = attn_weights_list[0][0]  # (num_heads, seq_len, seq_len)
        else:
            raw_attn = attn_weights_list[0]

        head_avg = raw_attn.mean(dim=0)  # (seq_len, seq_len)

        # Get emissions for confidence analysis
        with torch.no_grad():
            word_repr = self.model.word_embedding(w)
            lang_repr = self.model.lang_embedding(l, len(tokens))
            if self.model.use_char_cnn and self.model.char_cnn is not None:
                char_repr = self.model.char_cnn(c)
                combined = torch.cat([word_repr, char_repr, lang_repr], dim=-1)
            else:
                combined = torch.cat([word_repr, lang_repr], dim=-1)
            combined = self.model.concat_dropout(combined)
            lstm_hidden = self.model.encoder(combined, mask=mask, return_hidden=True)
            if self.model.use_attention and self.model.attention is not None:
                attended, _ = self.model.attention(lstm_hidden, mask=mask)
            else:
                attended = lstm_hidden
            emissions = self.model.fc_shared(attended)[0]  # (seq_len, num_tags)

        return {
            "tokens": list(tokens),
            "predicted_tags": predicted_tags,
            "predicted_tag_ids": predicted_path,
            "attention_weights": raw_attn,
            "head_avg_attention": head_avg,
            "emissions": emissions,
        }

    def get_token_attribution_raw_attention(
        self,
        query_token_idx: int,
        attention_weights: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Extract attribution vector for a target query token using raw attention weights.

        Args:
            query_token_idx: position of the predicted token (query)
            attention_weights: Tensor of shape (num_heads, seq_len, seq_len)

        Returns:
            Dict with:
            - 'per_head': (num_heads, seq_len) attention vectors from query to all keys
            - 'head_average': (seq_len,) averaged attention vector across heads
        """
        # query_token_idx attends to key j: weights[head, query_token_idx, :]
        per_head = attention_weights[:, query_token_idx, :]
        head_avg = per_head.mean(dim=0)
        return {
            "per_head": per_head,
            "head_average": head_avg,
        }
