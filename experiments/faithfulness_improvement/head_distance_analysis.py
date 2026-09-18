"""
Context-Distance and 4-Head Quantitative Analysis Module.

Analyzes:
1. Context distance breakdown (dist 1, dist 2, dist 3-5, dist >5)
2. Attention-head specialization metrics across all 4 self-attention heads:
   - Shannon entropy
   - Average attention distance (span radius)
   - Proportion of attention to self/entity-internal
   - Proportion to immediate context (|j - q| = 1)
   - Proportion to distant context (|j - q| > 3)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from explainability.attention_extractor import AttentionExtractor


def compute_entropy(distribution: torch.Tensor) -> float:
    """Compute Shannon entropy in nats."""
    dist = distribution.clamp(min=1e-12)
    return float(-(dist * torch.log(dist)).sum().item())


def analyze_context_distance_profile(
    attribution_scores: torch.Tensor,
    target_token_idx: int,
) -> dict[str, float]:
    """
    Partition attribution mass across distance buckets from the target entity token.
    Excludes the target token itself (context-only).
    """
    seq_len = len(attribution_scores)
    dist_buckets = {
        "dist_1": 0.0,    # |j - q| == 1
        "dist_2": 0.0,    # |j - q| == 2
        "dist_3_5": 0.0,  # 3 <= |j - q| <= 5
        "dist_gt_5": 0.0, # |j - q| > 5
    }
    context_sum = 0.0

    for j in range(seq_len):
        if j == target_token_idx:
            continue
        dist = abs(j - target_token_idx)
        score = float(attribution_scores[j].item())
        context_sum += score

        if dist == 1:
            dist_buckets["dist_1"] += score
        elif dist == 2:
            dist_buckets["dist_2"] += score
        elif 3 <= dist <= 5:
            dist_buckets["dist_3_5"] += score
        else:
            dist_buckets["dist_gt_5"] += score

    # Normalize by total context mass
    if context_sum > 1e-12:
        return {k: round(v / context_sum * 100.0, 2) for k, v in dist_buckets.items()}
    return {k: 0.0 for k in dist_buckets}


def analyze_head_profiles(
    attention_weights: torch.Tensor,
    target_token_idx: int,
) -> dict[int, dict[str, float]]:
    """
    Compute head-specific properties for a single entity prediction instance.

    Args:
        attention_weights: Tensor of shape (num_heads, seq_len, seq_len).
        target_token_idx: Query token index.

    Returns:
        Dict mapping head_idx (0..3) to metrics.
    """
    num_heads, seq_len, _ = attention_weights.shape
    head_metrics: dict[int, dict[str, float]] = {}

    for h in range(num_heads):
        row = attention_weights[h, target_token_idx, :]  # (seq_len,)
        ent = compute_entropy(row)

        self_prop = float(row[target_token_idx].item())

        # Distance distribution
        avg_dist = 0.0
        immediate_prop = 0.0
        distant_prop = 0.0

        for j in range(seq_len):
            w = float(row[j].item())
            d = abs(j - target_token_idx)
            avg_dist += w * d
            if d == 1:
                immediate_prop += w
            elif d > 3:
                distant_prop += w

        head_metrics[h] = {
            "entropy": round(ent, 4),
            "avg_distance": round(avg_dist, 2),
            "self_prop": round(self_prop * 100.0, 2),
            "immediate_prop": round(immediate_prop * 100.0, 2),
            "distant_prop": round(distant_prop * 100.0, 2),
        }

    return head_metrics
