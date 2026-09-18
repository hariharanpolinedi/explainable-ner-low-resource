"""
Statistical Analysis & Paired Bootstrap Confidence Interval Module.

Computes:
1. Descriptive metrics: mean, median, standard deviation of drops and gaps.
2. Win rates (percentage of predictions where Drop_top > Drop_rand).
3. Paired percentile bootstrap 95% confidence intervals (B=1,000 resamples, fixed seed).
4. Checks whether the 95% confidence interval straddles zero.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def compute_descriptive_stats(values: Sequence[float]) -> dict[str, float]:
    """Compute mean, median, and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    sorted_vals = sorted(values)
    mean_val = sum(values) / n
    median_val = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in values) / max(1, n - 1)
    std_val = math.sqrt(variance)

    return {
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
    }


def compute_paired_bootstrap_ci(
    gaps: Sequence[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute paired percentile bootstrap confidence interval for mean faithfulness gap.
    """
    n = len(gaps)
    if n == 0:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "crosses_zero": True}

    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(num_bootstraps):
        resample = [rng.choice(gaps) for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)

    bootstrap_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * num_bootstraps)
    upper_idx = int((1.0 - alpha) * num_bootstraps) - 1

    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]
    crosses_zero = bool(ci_lower <= 0.0 <= ci_upper)

    return {
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "crosses_zero": crosses_zero,
    }


def compute_full_metric_suite(
    top_drops: Sequence[float],
    rand_drops: Sequence[float],
    gaps: Sequence[float],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute complete statistical summary for an evaluation condition.
    """
    n = len(gaps)
    if n == 0:
        return {}

    top_stats = compute_descriptive_stats(top_drops)
    rand_stats = compute_descriptive_stats(rand_drops)
    gap_stats = compute_descriptive_stats(gaps)
    bootstrap_ci = compute_paired_bootstrap_ci(gaps, seed=seed)

    win_count = sum(1 for g in gaps if g > 0.0)
    win_rate = round(win_count / n * 100.0, 2)

    return {
        "n": n,
        "mean_top_drop": top_stats["mean"],
        "median_top_drop": top_stats["median"],
        "std_top_drop": top_stats["std"],
        "mean_rand_drop": rand_stats["mean"],
        "median_rand_drop": rand_stats["median"],
        "std_rand_drop": rand_stats["std"],
        "mean_gap": gap_stats["mean"],
        "median_gap": gap_stats["median"],
        "std_gap": gap_stats["std"],
        "win_rate": win_rate,
        "ci_lower_95": bootstrap_ci["ci_lower"],
        "ci_upper_95": bootstrap_ci["ci_upper"],
        "crosses_zero": bootstrap_ci["crosses_zero"],
    }
