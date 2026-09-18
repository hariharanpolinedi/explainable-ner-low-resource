"""
Checkpoint Weight Divergence Gate.

Enforces a hard gate before writing any retrained/patched checkpoint:
1. Computes parameter-wise difference (L2 norm, changed parameter count, % changed).
2. Rejects any candidate checkpoint that has not diverged by at least `min_pct_changed` (default: 1.0%).
3. Prevents silent re-saving of baseline weights under patched names.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


def compute_parameter_divergence(
    state_candidate: dict[str, torch.Tensor],
    state_baseline: dict[str, torch.Tensor],
    allow_shape_mismatch: bool = False,
) -> dict[str, Any]:
    """
    Compute parameter-wise divergence between two model state dicts.
    """
    total_params = 0
    changed_params = 0
    sq_diff_sum = 0.0
    max_diff = 0.0

    common_keys = [k for k in state_candidate.keys() if k in state_baseline]
    if not common_keys:
        raise ValueError("State dicts have no common keys to compare!")

    for k in common_keys:
        t_cand = state_candidate[k].detach().cpu().float()
        t_base = state_baseline[k].detach().cpu().float()

        if t_cand.shape != t_base.shape:
            if allow_shape_mismatch:
                continue
            raise ValueError(f"Shape mismatch for parameter '{k}': {t_cand.shape} vs {t_base.shape}")

        num_el = t_cand.numel()
        total_params += num_el

        diff = torch.abs(t_cand - t_base)
        changed = (diff > 1e-6).sum().item()
        changed_params += changed

        sq_diff = torch.sum((t_cand - t_base) ** 2).item()
        sq_diff_sum += sq_diff

        m_diff = torch.max(diff).item()
        if m_diff > max_diff:
            max_diff = m_diff

    l2_diff = float(sq_diff_sum ** 0.5)
    pct_changed = (changed_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "changed_params": changed_params,
        "pct_changed": round(pct_changed, 4),
        "l2_diff": round(l2_diff, 6),
        "max_diff": round(max_diff, 6),
    }


def verify_weight_divergence(
    state_candidate: dict[str, torch.Tensor],
    state_baseline: dict[str, torch.Tensor],
    min_pct_changed: float = 1.0,
    min_l2_diff: float = 0.01,
    allow_shape_mismatch: bool = False,
) -> tuple[bool, dict[str, Any], str]:
    """
    Verify if candidate weights have diverged meaningfully from baseline.
    Returns (passed, divergence_metrics, message).
    """
    div = compute_parameter_divergence(
        state_candidate,
        state_baseline,
        allow_shape_mismatch=allow_shape_mismatch,
    )

    if div["pct_changed"] < min_pct_changed or div["l2_diff"] < min_l2_diff:
        msg = (
            f"DIVERGENCE GATE REJECTED: Candidate weights failed divergence threshold! "
            f"Changed {div['pct_changed']:.2f}% parameters (threshold: >={min_pct_changed:.2f}%), "
            f"L2 diff = {div['l2_diff']:.6f} (threshold: >={min_l2_diff:.4f}). "
            f"Retraining did not update weights meaningfully."
        )
        return False, div, msg

    msg = (
        f"DIVERGENCE GATE PASSED: Genuinely retrained weights verified. "
        f"{div['pct_changed']:.2f}% parameters changed ({div['changed_params']:,}/{div['total_params']:,}), "
        f"L2 diff = {div['l2_diff']:.4f}, max diff = {div['max_diff']:.4f}."
    )
    return True, div, msg


def save_patched_checkpoint_with_gate(
    candidate_state: dict[str, torch.Tensor],
    baseline_state: dict[str, torch.Tensor],
    output_path: Path | str,
    metadata: dict[str, Any],
    min_pct_changed: float = 1.0,
    min_l2_diff: float = 0.01,
) -> tuple[bool, dict[str, Any]]:
    """
    Saves the checkpoint ONLY IF the divergence gate passes.
    If gate fails, raises an explicit error and refuses to write the checkpoint.
    """
    passed, div, msg = verify_weight_divergence(
        candidate_state,
        baseline_state,
        min_pct_changed=min_pct_changed,
        min_l2_diff=min_l2_diff,
    )

    out_p = Path(output_path)
    if not passed:
        print(f"\n[FATAL ERROR] {msg}")
        print(f"[FATAL ERROR] Refusing to save un-updated model to: {out_p.name}\n")
        return False, div

    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["model_state_dict"] = candidate_state
    payload["divergence_from_baseline"] = div

    torch.save(payload, out_p)
    print(f"\n[SUCCESS] {msg}")
    print(f"[SUCCESS] Saved verified patched checkpoint to: {out_p.name}\n")
    return True, div


def run_divergence_gate_tests() -> bool:
    """
    Unit test verifying divergence gate behavior on both identical and modified weights.
    """
    print("Testing Checkpoint Divergence Gate...")

    # Synthetic baseline
    base = {
        "layer1.weight": torch.randn(50, 50),
        "layer1.bias": torch.zeros(50),
        "layer2.weight": torch.randn(10, 50),
    }

    # Case 1: Identical weights -> MUST FAIL
    identical_cand = {k: v.clone() for k, v in base.items()}
    passed_1, div_1, _ = verify_weight_divergence(identical_cand, base, min_pct_changed=1.0)
    assert not passed_1, "Gate erroneously passed identical weights!"
    assert div_1["pct_changed"] == 0.0
    print("  [PASS] Correctly rejected identical candidate weights (0.00% diff).")

    # Case 2: Minimal/insufficient divergence (0.05% of params changed) -> MUST FAIL
    minor_cand = {k: v.clone() for k, v in base.items()}
    minor_cand["layer1.weight"][0, 0] += 0.001
    passed_2, div_2, _ = verify_weight_divergence(minor_cand, base, min_pct_changed=1.0)
    assert not passed_2, "Gate erroneously passed insufficiently diverged weights!"
    print(f"  [PASS] Correctly rejected sub-threshold candidate weights ({div_2['pct_changed']:.2f}% diff).")

    # Case 3: Genuine divergence (10% of params changed with L2 > 0.1) -> MUST PASS
    divergent_cand = {k: v.clone() for k, v in base.items()}
    divergent_cand["layer1.weight"] += torch.randn(50, 50) * 0.1
    passed_3, div_3, _ = verify_weight_divergence(divergent_cand, base, min_pct_changed=1.0)
    assert passed_3, "Gate erroneously rejected genuinely diverged weights!"
    print(f"  [PASS] Correctly accepted genuinely diverged weights ({div_3['pct_changed']:.2f}% diff, L2={div_3['l2_diff']:.4f}).")

    print("Checkpoint Divergence Gate unit tests PASSED.\n")
    return True


if __name__ == "__main__":
    run_divergence_gate_tests()
