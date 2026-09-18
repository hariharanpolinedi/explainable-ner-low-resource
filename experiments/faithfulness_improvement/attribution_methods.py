"""
Unified Attribution Methods Suite for BiLSTM-Attention-CRF NER Models.

Provides:
1. Raw Attention Attribution (Head-average, Head-selective)
2. Gradient x Attention (Baseline Absolute, Signed ReLU, Log-Softmax, CRF-Aware)
3. Gradient x Input Embeddings (End-to-end BiLSTM+Attention+CRF flow)
4. Integrated Gradients on Input Embeddings (Path-integrated axiomatic attribution)

Strictly native PyTorch autograd; NO external interpretability libraries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from explainability.attention_extractor import AttentionExtractor
from model.ner_model_v4_multilingual import NERMultilingualModel


class UnifiedAttributor:
    """
    Unified attribution evaluator supporting multiple targets, aggregations,
    and representations across the frozen multilingual NER model.
    """

    def __init__(self, extractor: AttentionExtractor) -> None:
        self.extractor = extractor
        self.model: NERMultilingualModel = extractor.model
        self.device = extractor.device

    def attribute(
        self,
        tokens: Sequence[str],
        target_token_idx: int,
        target_tag_id: int | None = None,
        method: str = "raw_attention",
        target_type: str = "emission",
        aggregation: str = "abs",
        head_idx: int | None = None,
        ig_steps: int = 10,
    ) -> dict[str, Any]:
        """
        Compute attribution scores for all tokens in sentence toward the target entity token.

        Args:
            tokens: List of tokens.
            target_token_idx: Index of entity token to explain.
            target_tag_id: Target class ID (if None, model argmax/CRF prediction is used).
            method: 'raw_attention', 'grad_x_attn', 'grad_x_input', 'integrated_gradients'.
            target_type: 'emission', 'log_softmax', 'crf_sequence', 'crf_path'.
            aggregation: 'abs' (baseline), 'relu' (positive only), 'signed'.
            head_idx: Optional specific head index (0..3) for head-selective attention.
            ig_steps: Steps for Integrated Gradients.

        Returns:
            Dict containing:
            - 'tokens': input tokens
            - 'target_token': tokens[target_token_idx]
            - 'target_token_idx': target_token_idx
            - 'target_tag_id': target_tag_id
            - 'attribution_scores': Tensor of shape (seq_len,) normalized to sum to 1.0
            - 'raw_scores': Tensor of shape (seq_len,) before normalization
        """
        seq_len = len(tokens)
        if not (0 <= target_token_idx < seq_len):
            raise IndexError(f"target_token_idx {target_token_idx} out of bounds for seq_len {seq_len}")

        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        self.model.zero_grad()
        self.model.eval()

        w, c, l, mask = self.extractor.prepare_input_tensors(tokens)

        # -------------------------------------------------------------
        # 1. Forward to Combined Embeddings
        # -------------------------------------------------------------
        word_repr = self.model.word_embedding(w)
        lang_repr = self.model.lang_embedding(l, seq_len)
        if self.model.use_char_cnn and self.model.char_cnn is not None:
            char_repr = self.model.char_cnn(c)
            combined = torch.cat([word_repr, char_repr, lang_repr], dim=-1)
        else:
            combined = torch.cat([word_repr, lang_repr], dim=-1)

        # -------------------------------------------------------------
        # Method: Raw Attention (No backward required)
        # -------------------------------------------------------------
        if method == "raw_attention":
            with torch.no_grad():
                combined_drop = self.model.concat_dropout(combined)
                lstm_hidden = self.model.encoder(combined_drop, mask=mask, return_hidden=True)
                attended, attn_weights = self.model.attention(lstm_hidden, mask=mask)
                emissions = self.model.fc_shared(attended)

                if target_tag_id is None:
                    target_tag_id = int(emissions[0, target_token_idx].argmax(dim=-1).item())

                # attn_weights is list of blocks; block 0 has shape (batch_size, num_heads, seq_len, seq_len)
                if isinstance(attn_weights, list):
                    raw_attn = attn_weights[0][0]  # (num_heads, seq_len, seq_len)
                else:
                    raw_attn = attn_weights[0]

                if head_idx is not None:
                    q_attn = raw_attn[head_idx, target_token_idx, :]  # (seq_len,)
                else:
                    q_attn = raw_attn[:, target_token_idx, :].mean(dim=0)  # (seq_len,)

                # Normalize to unit sum
                attr_sum = q_attn.sum()
                norm_attr = q_attn / attr_sum if attr_sum > 1e-12 else torch.ones_like(q_attn) / seq_len

            torch.backends.cudnn.enabled = prev_cudnn
            return {
                "tokens": list(tokens),
                "target_token": tokens[target_token_idx],
                "target_token_idx": target_token_idx,
                "target_tag_id": target_tag_id,
                "attribution_scores": norm_attr.detach(),
                "raw_scores": q_attn.detach(),
            }

        # -------------------------------------------------------------
        # Helper: Compute forward pass from combined embedding
        # -------------------------------------------------------------
        def _forward_from_combined(
            emb: torch.Tensor,
            retain_attn_grad: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            emb_drop = self.model.concat_dropout(emb)
            lstm_hidden = self.model.encoder(emb_drop, mask=mask, return_hidden=True)

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
            if retain_attn_grad:
                attn_weights.retain_grad()

            context = torch.matmul(layer.attn_dropout(attn_weights), v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, layer.hidden_dim)
            attn_output = layer.proj_dropout(layer.w_o(context))
            attended_states = layer.layer_norm(lstm_hidden + attn_output)

            emissions = self.model.fc_shared(attended_states)
            return emissions, attn_weights, attended_states

        # -------------------------------------------------------------
        # Helper: Compute target score S for backprop
        # -------------------------------------------------------------
        def _compute_target_scalar(
            emissions: torch.Tensor,
            resolved_tag_id: int,
        ) -> torch.Tensor:
            if target_type == "emission":
                return emissions[0, target_token_idx, resolved_tag_id]
            elif target_type == "log_softmax":
                lsm = torch.log_softmax(emissions[0, target_token_idx], dim=-1)
                return lsm[resolved_tag_id]
            elif target_type in ("crf_sequence", "crf_path"):
                # Use greedy or Viterbi predicted sequence for full sequence score
                with torch.no_grad():
                    viterbi_paths = self.model.crf_shared.decode(emissions, mask=mask)
                    best_path = viterbi_paths[0]
                    # Ensure target_token_idx has resolved_tag_id
                    best_path[target_token_idx] = resolved_tag_id
                    path_tensor = torch.tensor([best_path], dtype=torch.long, device=self.device)

                gold_score = self.model.crf_shared._compute_score(emissions, path_tensor, mask)
                if target_type == "crf_path":
                    return gold_score.squeeze()
                else:
                    # CRF conditional log-likelihood: score - log_Z
                    forward_score = self.model.crf_shared._compute_forward_score(emissions, mask)
                    return (gold_score - forward_score).squeeze()
            else:
                raise ValueError(f"Unknown target_type: {target_type}")

        # -------------------------------------------------------------
        # Method: Gradient x Attention
        # -------------------------------------------------------------
        if method == "grad_x_attn":
            emissions, attn_weights, _ = _forward_from_combined(combined, retain_attn_grad=True)

            if target_tag_id is None:
                target_tag_id = int(emissions[0, target_token_idx].argmax(dim=-1).item())

            scalar = _compute_target_scalar(emissions, target_tag_id)
            scalar.backward()

            grad_tensor = attn_weights.grad[0]        # (num_heads, seq_len, seq_len)
            weights_tensor = attn_weights.detach()[0]  # (num_heads, seq_len, seq_len)

            q_grad = grad_tensor[:, target_token_idx, :]   # (num_heads, seq_len)
            q_attn = weights_tensor[:, target_token_idx, :] # (num_heads, seq_len)
            grad_x_attn = q_grad * q_attn

            if head_idx is not None:
                raw_attr = grad_x_attn[head_idx]
            else:
                raw_attr = grad_x_attn.mean(dim=0)

            # Apply aggregation
            if aggregation == "abs":
                processed = torch.abs(raw_attr)
            elif aggregation == "relu":
                processed = torch.clamp(raw_attr, min=0.0)
            elif aggregation == "signed":
                processed = raw_attr
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            attr_sum = processed.sum()
            if attr_sum > 1e-12:
                final_attr = processed / attr_sum
            else:
                final_attr = torch.ones_like(processed) / seq_len

            torch.backends.cudnn.enabled = prev_cudnn
            return {
                "tokens": list(tokens),
                "target_token": tokens[target_token_idx],
                "target_token_idx": target_token_idx,
                "target_tag_id": target_tag_id,
                "attribution_scores": final_attr.detach(),
                "raw_scores": raw_attr.detach(),
            }

        # -------------------------------------------------------------
        # Method: Gradient x Input Embedding
        # -------------------------------------------------------------
        elif method == "grad_x_input":
            combined_grad = combined.clone().detach().requires_grad_(True)
            emissions, _, _ = _forward_from_combined(combined_grad, retain_attn_grad=False)

            if target_tag_id is None:
                target_tag_id = int(emissions[0, target_token_idx].argmax(dim=-1).item())

            scalar = _compute_target_scalar(emissions, target_tag_id)
            scalar.backward()

            input_grad = combined_grad.grad[0]  # (seq_len, combined_dim)
            input_emb = combined[0].detach()    # (seq_len, combined_dim)

            # Dot product along embedding dimension for each token
            dot_prod = (input_emb * input_grad).sum(dim=-1)  # (seq_len,)

            if aggregation == "abs":
                processed = torch.abs(dot_prod)
            elif aggregation == "relu":
                processed = torch.clamp(dot_prod, min=0.0)
            else:
                processed = dot_prod

            attr_sum = processed.sum()
            if attr_sum > 1e-12:
                final_attr = processed / attr_sum
            else:
                final_attr = torch.ones_like(processed) / seq_len

            torch.backends.cudnn.enabled = prev_cudnn
            return {
                "tokens": list(tokens),
                "target_token": tokens[target_token_idx],
                "target_token_idx": target_token_idx,
                "target_tag_id": target_tag_id,
                "attribution_scores": final_attr.detach(),
                "raw_scores": dot_prod.detach(),
            }

        # -------------------------------------------------------------
        # Method: Integrated Gradients on Input Embeddings
        # -------------------------------------------------------------
        elif method == "integrated_gradients":
            input_emb = combined.clone().detach()  # (1, seq_len, combined_dim)
            baseline_emb = torch.zeros_like(input_emb)

            # Determine target_tag_id first
            with torch.no_grad():
                init_emissions, _, _ = _forward_from_combined(input_emb, retain_attn_grad=False)
                if target_tag_id is None:
                    target_tag_id = int(init_emissions[0, target_token_idx].argmax(dim=-1).item())

            total_grad = torch.zeros_like(input_emb)

            for step in range(1, ig_steps + 1):
                alpha = float(step) / ig_steps
                interpolated = (baseline_emb + alpha * (input_emb - baseline_emb)).requires_grad_(True)
                self.model.zero_grad()
                step_emissions, _, _ = _forward_from_combined(interpolated, retain_attn_grad=False)
                step_scalar = _compute_target_scalar(step_emissions, target_tag_id)
                step_scalar.backward()
                total_grad = total_grad + interpolated.grad

            avg_grad = total_grad[0] / ig_steps  # (seq_len, combined_dim)
            diff = (input_emb[0] - baseline_emb[0])  # (seq_len, combined_dim)
            ig_scores = (diff * avg_grad).sum(dim=-1)  # (seq_len,)

            if aggregation == "abs":
                processed = torch.abs(ig_scores)
            elif aggregation == "relu":
                processed = torch.clamp(ig_scores, min=0.0)
            else:
                processed = ig_scores

            attr_sum = processed.sum()
            if attr_sum > 1e-12:
                final_attr = processed / attr_sum
            else:
                final_attr = torch.ones_like(processed) / seq_len

            torch.backends.cudnn.enabled = prev_cudnn
            return {
                "tokens": list(tokens),
                "target_token": tokens[target_token_idx],
                "target_token_idx": target_token_idx,
                "target_tag_id": target_tag_id,
                "attribution_scores": final_attr.detach(),
                "raw_scores": ig_scores.detach(),
            }

        else:
            torch.backends.cudnn.enabled = prev_cudnn
            raise ValueError(f"Unknown attribution method: {method}")
