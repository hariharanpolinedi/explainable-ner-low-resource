"""
Class-Weighted and Focal Loss Modules for NER Imbalance Handling (Phase 11).

Provides two swappable loss variants targeting 'O'-tag dominance and entity recall suppression:
1. Inverse-frequency class-weighted CRF loss: upweights underrepresented entity tags (PER, LOC, ORG, MISC)
   relative to majority 'O' tokens via calibrated tag emission scaling.
2. Focal CRF loss: dynamically down-weights easy, well-classified tokens (typically 'O')
   via modulating factor (1 - p_t)^gamma, concentrating gradient updates on hard entity boundaries.
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
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from model.crf_layer import LinearChainCRF
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID


def compute_inverse_frequency_weights(
    tag_counts: dict[int, int] | None = None,
    num_tags: int = 9,
    o_tag_id: int = 0,
    max_weight: float = 5.0,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights for NER tagset.
    Tags for entities are upweighted relative to the majority 'O' tag.
    """
    if tag_counts is None:
        # Default empirical prior: 'O' is ~90% of tokens, entities are ~10%
        # PER: 2.5, LOC: 2.5, ORG: 4.0, MISC: 3.0
        weights = torch.ones(num_tags, dtype=torch.float32)
        for tid, tag in ID_TO_TAG.items():
            if tag == "O":
                weights[tid] = 1.0
            elif "ORG" in tag:
                weights[tid] = 3.5
            elif "MISC" in tag:
                weights[tid] = 3.0
            elif "PER" in tag:
                weights[tid] = 2.5
            elif "LOC" in tag:
                weights[tid] = 2.5
        return weights

    total_tokens = sum(tag_counts.values())
    weights = torch.ones(num_tags, dtype=torch.float32)
    o_count = tag_counts.get(o_tag_id, 1)

    for tid in range(num_tags):
        cnt = tag_counts.get(tid, 0)
        if cnt == 0:
            weights[tid] = max_weight
        elif tid == o_tag_id:
            weights[tid] = 1.0
        else:
            raw_w = (o_count / (cnt + 1)) ** smoothing
            weights[tid] = min(float(raw_w), max_weight)

    return weights


class ClassWeightedCRFLoss(nn.Module if nn else object):
    """
    Class-weighted CRF loss. Scales emission logits by inverse-frequency weights
    before computing sequence partition function and gold sequence scores,
    effectively penalizing entity omission during transition competition.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.reduction = reduction

    def forward(
        self,
        crf: LinearChainCRF,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            crf: LinearChainCRF module
            emissions: Tensor of shape (B, T, num_tags)
            tags: LongTensor of shape (B, T)
            mask: BoolTensor of shape (B, T)
        """
        # Scale emissions by class weights broadcasted across tokens
        # emissions: (B, T, num_tags), class_weights: (num_tags,)
        weights = self.class_weights.to(emissions.device)
        weighted_emissions = emissions * weights.unsqueeze(0).unsqueeze(0)

        # Standard forward-backward on weighted emissions
        return crf(weighted_emissions, tags, mask=mask, reduction=self.reduction)


class FocalCRFLoss(nn.Module if nn else object):
    """
    Focal CRF Loss for sequence labeling.
    Modulates token loss by (1 - p_t)^gamma to focus learning on hard, uncertain
    entity tokens while down-weighting well-classified majority 'O' tokens.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        crf_weight: float = 1.0,
        focal_aux_weight: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.crf_weight = crf_weight
        self.focal_aux_weight = focal_aux_weight
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        crf: LinearChainCRF,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines global CRF NLL loss with token-level focal modulating penalty.
        """
        # 1. Global CRF sequence loss
        crf_loss = crf(emissions, tags, mask=mask, reduction="none")  # (B,)

        # 2. Token-level focal probabilities
        probs = F.softmax(emissions, dim=-1)  # (B, T, num_tags)
        # Gather probabilities of the gold tags
        gold_probs = probs.gather(2, tags.unsqueeze(-1)).squeeze(-1)  # (B, T)
        gold_probs = torch.clamp(gold_probs, 1e-6, 1.0 - 1e-6)

        # Focal modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - gold_probs) ** self.gamma  # (B, T)

        if self.alpha is not None:
            alpha = self.alpha.to(emissions.device)
            alpha_t = alpha[tags]  # (B, T)
            modulating_factor = modulating_factor * alpha_t

        # Token-level cross-entropy
        token_ce = -torch.log(gold_probs)  # (B, T)
        token_focal = modulating_factor * token_ce * mask.float()  # (B, T)

        seq_lengths = torch.clamp(mask.sum(dim=1).float(), min=1.0)
        mean_focal_per_seq = token_focal.sum(dim=1) / seq_lengths  # (B,)

        combined_loss = self.crf_weight * crf_loss + self.focal_aux_weight * mean_focal_per_seq

        if self.reduction == "mean":
            return combined_loss.mean()
        elif self.reduction == "sum":
            return combined_loss.sum()
        else:
            return combined_loss


if __name__ == "__main__":
    crf = LinearChainCRF(num_tags=9)
    em = torch.randn(2, 4, 9, requires_grad=True)
    tags = torch.tensor([[0, 1, 2, 0], [0, 3, 4, 0]])
    mask = torch.ones((2, 4), dtype=torch.bool)

    cw = compute_inverse_frequency_weights()
    cw_loss = ClassWeightedCRFLoss(cw)
    loss1 = cw_loss(crf, em, tags, mask)
    loss1.backward(retain_graph=True)
    print("ClassWeightedCRFLoss output:", loss1.item())

    focal_loss = FocalCRFLoss(gamma=2.0)
    loss2 = focal_loss(crf, em, tags, mask)
    print("FocalCRFLoss output:", loss2.item())
