"""
Verify Assamese Ablation Checkpoint Divergence.

Computes pairwise parameter divergence (% changed and L2 norm diff) between:
- checkpoints/final_assamese.pt (Reference)
- checkpoints/fromscratch_control_assamese.pt (Condition d)
- checkpoints/finetuned_assamese.pt (Condition e)
- checkpoints/selftrained_assamese_best.pt (Condition f)

Determines whether Assamese's 98.91% flat-line scores reflect:
1. Accidental duplicate re-saves of the reference model, OR
2. Genuinely distinct trained models reaching an annotation ceiling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def verify_assamese_checkpoints() -> dict[str, Any]:
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    ckpt_paths = {
        "final_reference": checkpoints_dir / "final_assamese.pt",
        "(d) fromscratch_control": checkpoints_dir / "fromscratch_control_assamese.pt",
        "(e) finetuned": checkpoints_dir / "finetuned_assamese.pt",
        "(f) selftrained_best": checkpoints_dir / "selftrained_assamese_best.pt",
    }

    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, path in ckpt_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        c = torch.load(path, map_location="cpu", weights_only=False)
        st = c.get("model_state_dict", c)
        states[name] = st

    results: dict[str, Any] = {"comparisons": {}}
    names = list(states.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            st1, st2 = states[n1], states[n2]
            common_keys = set(st1.keys()) & set(st2.keys())

            tot_params = 0
            diff_params = 0
            sq_diff = 0.0

            for k in common_keys:
                t1, t2 = st1[k].float(), st2[k].float()
                if t1.shape == t2.shape:
                    num = t1.numel()
                    tot_params += num
                    d = (t1 != t2).sum().item()
                    diff_params += d
                    sq_diff += ((t1 - t2) ** 2).sum().item()

            l2_diff = round(sq_diff ** 0.5, 4)
            pct_changed = round((diff_params / tot_params * 100), 2) if tot_params > 0 else 0.0

            is_duplicate = (pct_changed == 0.0 and l2_diff == 0.0)
            pair_key = f"{n1}__vs__{n2}"
            results["comparisons"][pair_key] = {
                "pct_params_changed": pct_changed,
                "l2_norm_diff": l2_diff,
                "is_duplicate": is_duplicate,
                "verdict": "DUPLICATE" if is_duplicate else "GENUINELY DISTINCT",
            }
            print(f"[{n1}] vs [{n2}]:")
            print(f"  % Changed: {pct_changed}% | L2 Diff: {l2_diff} | Verdict: {'DUPLICATE' if is_duplicate else 'GENUINELY DISTINCT'}")

    out_file = PROJECT_ROOT / "reports" / "experiment_logs" / "assamese_checkpoint_divergence.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    verify_assamese_checkpoints()
