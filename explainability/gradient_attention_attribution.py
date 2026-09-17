"""
Gradient × Attention Attribution Pipeline (Phase 13).

Computes the attribution of each input token towards the model's prediction of a
target entity token by evaluating the gradient of the predicted emission score
with respect to the multi-head self-attention weights:

    Attribution(q -> j) = Grad(q -> j) ⊙ Attn(q -> j)

where q is the query (entity token) position and j is the attended key position.
Zero external interpretability packages; 100% direct PyTorch autograd.
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

from explainability.attention_extractor import AttentionExtractor


class GradientAttentionAttributor:
    """
    Computes Gradient × Attention attribution maps for tokens predicted by a frozen NER model.
    """

    def __init__(
        self,
        extractor: AttentionExtractor,
    ) -> None:
        self.extractor = extractor
        self.model = extractor.model
        self.device = extractor.device

    def attribute_token(
        self,
        tokens: Sequence[str],
        target_token_idx: int,
        target_tag_id: int | None = None,
        normalize: bool = True,
    ) -> dict[str, Any]:
        """
        Compute Gradient × Attention attribution scores for a target token position.

        Args:
            tokens: Sequence of input sentence tokens.
            target_token_idx: Integer index of the target query token (0 <= idx < len(tokens)).
            target_tag_id: Optional integer tag ID. If None, uses the model's argmax/CRF prediction.
            normalize: If True, applies ReLU to retain positive attribution and normalizes by L1 sum.

        Returns:
            Dict containing:
            - 'tokens': input tokens
            - 'target_token': tokens[target_token_idx]
            - 'target_token_idx': target_token_idx
            - 'target_tag_id': predicted or specified tag ID
            - 'grad_x_attn_per_head': Tensor of shape (num_heads, seq_len)
            - 'attribution_scores': Tensor of shape (seq_len,) aggregated across heads
            - 'ranked_tokens': list of (token, score, index) sorted by attribution descending
        """
        seq_len = len(tokens)
        if not (0 <= target_token_idx < seq_len):
            raise IndexError(f"target_token_idx ({target_token_idx}) out of range for sequence of length {seq_len}")

        # Temporarily disable cuDNN to allow LSTM backward in eval mode
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        self.model.zero_grad()
        self.model.eval()

        w, c, l, mask = self.extractor.prepare_input_tensors(tokens)

        # 1. Forward through embedding and BiLSTM
        word_repr = self.model.word_embedding(w)
        lang_repr = self.model.lang_embedding(l, seq_len)

        if self.model.use_char_cnn and self.model.char_cnn is not None:
            char_repr = self.model.char_cnn(c)
            combined = torch.cat([word_repr, char_repr, lang_repr], dim=-1)
        else:
            combined = torch.cat([word_repr, lang_repr], dim=-1)

        combined = self.model.concat_dropout(combined)
        lstm_hidden = self.model.encoder(combined, mask=mask, return_hidden=True)

        # 2. Forward through attention with gradient hook on attention weights
        if not self.model.use_attention or self.model.attention is None:
            torch.backends.cudnn.enabled = prev_cudnn
            raise RuntimeError("Gradient x Attention requires use_attention=True")

        layer = self.model.attention.blocks[0]
        batch_size = 1
        q = layer.w_q(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
        k = layer.w_k(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
        v = layer.w_v(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * layer.scale
        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights.retain_grad()

        context = torch.matmul(layer.attn_dropout(attn_weights), v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, layer.hidden_dim)
        attn_output = layer.proj_dropout(layer.w_o(context))
        attended_states = layer.layer_norm(lstm_hidden + attn_output)

        # 3. Project to emissions
        emissions = self.model.fc_shared(attended_states)  # (1, seq_len, num_tags)

        # 4. Resolve target tag ID
        if target_tag_id is None:
            # Greedy argmax or CRF path
            target_tag_id = int(emissions[0, target_token_idx].argmax(dim=-1).item())

        # 5. Backpropagate from the target emission score
        target_score = emissions[0, target_token_idx, target_tag_id]
        target_score.backward()

        # 6. Compute element-wise product: Grad * Attn
        # attn_weights.grad has shape (1, num_heads, seq_len, seq_len)
        if attn_weights.grad is None:
            torch.backends.cudnn.enabled = prev_cudnn
            raise RuntimeError("attn_weights.grad is None after backward pass.")

        grad_tensor = attn_weights.grad[0]        # (num_heads, seq_len, seq_len)
        weights_tensor = attn_weights.detach()[0]  # (num_heads, seq_len, seq_len)

        # Slice query row at target_token_idx: query attending to all keys j
        q_grad = grad_tensor[:, target_token_idx, :]       # (num_heads, seq_len)
        q_attn = weights_tensor[:, target_token_idx, :]     # (num_heads, seq_len)
        grad_x_attn_per_head = q_grad * q_attn              # (num_heads, seq_len)

        # Head aggregation: mean across heads
        raw_attribution = grad_x_attn_per_head.mean(dim=0)  # (seq_len,)

        if normalize:
            # Use magnitude (absolute impact) and normalize to unit sum
            abs_attr = torch.abs(raw_attribution)
            attr_sum = abs_attr.sum()
            if attr_sum > 1e-12:
                final_attribution = abs_attr / attr_sum
            else:
                final_attribution = torch.ones_like(abs_attr) / len(abs_attr)
        else:
            final_attribution = raw_attribution


        # Build ranked token list
        scores_list = final_attribution.tolist()
        ranked = sorted(
            [(tokens[i], scores_list[i], i) for i in range(seq_len)],
            key=lambda x: x[1],
            reverse=True,
        )

        torch.backends.cudnn.enabled = prev_cudnn

        return {
            "tokens": list(tokens),
            "target_token": tokens[target_token_idx],
            "target_token_idx": target_token_idx,
            "target_tag_id": target_tag_id,
            "raw_attention": q_attn.mean(dim=0).detach(),
            "grad_x_attn_per_head": grad_x_attn_per_head.detach(),
            "attribution_scores": final_attribution.detach(),
            "ranked_tokens": ranked,
        }
