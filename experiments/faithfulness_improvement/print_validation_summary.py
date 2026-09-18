"""
Print complete validation suite tables.
"""

from __future__ import annotations

import json
from pathlib import Path

val_path = Path("experiments/faithfulness_improvement/results/validation_suite_results.json")
with open(val_path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

languages = ["konkani", "assamese", "sinhala"]

print("=" * 115)
print("  SANITY CHECKS & CONTROL BASELINES (k=1, zero_crf)")
print("=" * 115)
print(f"{'Language':<10} | {'Method':<22} | {'Top-1':<7} | {'Rand-1':<7} | {'Gap':<8} | {'Win Rate':<9} | {'95% CI':<18}")
print("-" * 115)
for lang in languages:
    for m, m_data in data["sanity_checks"][lang].items():
        ci_str = f"[{m_data['ci_lower_95']:+.4f}, {m_data['ci_upper_95']:+.4f}]"
        print(
            f"{lang.capitalize():<10} | {m:<22} | "
            f"{m_data['mean_top_drop']:<7.4f} | {m_data['mean_rand_drop']:<7.4f} | "
            f"{m_data['mean_gap']:<+8.4f} | {m_data['win_rate']:<8.1f}% | {ci_str:<18}"
        )
    print("-" * 115)

print("\n--- MACRO SANITY CHECKS ---")
print(f"{'Method':<22} | {'Macro Gap':<12} | {'Macro Win Rate':<14}")
print("-" * 55)
for m, vals in data["macro_sanity"].items():
    print(f"{m:<22} | {vals['macro_gap']:<+12.4f} | {vals['macro_win_rate']:<13.1f}%")

print("\n" + "=" * 115)
print("  INTEGRATED GRADIENTS STEP CONVERGENCE & COMPLETENESS ERROR")
print("=" * 115)
for lang in languages:
    print(f"\n[{lang.capitalize()}]")
    for s, s_data in data["ig_sweep"][lang].items():
        print(f"  {s:<10} -> Mean Gap: {s_data['mean_gap']:+.4f} | Completeness Relative Error: {s_data['mean_completeness_error']*100:.2f}%")

print("\n" + "=" * 115)
print("  INPUT BASELINE SENSITIVITY SWEEP (Mean Gap at k=1)")
print("=" * 115)
for lang in languages:
    bs = data["baseline_sensitivity"][lang]
    print(f"{lang.capitalize():<10} | Zero Baseline: {bs['zero_baseline']:+.4f} | Mean Baseline: {bs['mean_baseline']:+.4f} | UNK Baseline: {bs['unk_baseline']:+.4f}")

print("\n" + "=" * 115)
print("  MULTI-SEED REPRODUCIBILITY SWEEP (Seeds 42, 123, 2026)")
print("=" * 115)
for lang in languages:
    print(f"\n[{lang.capitalize()}]")
    for m in ["grad_x_attn_relu", "grad_x_input"]:
        s_vals = data["multi_seed_reproducibility"][lang][m]
        v_list = list(s_vals.values())
        mean_v = sum(v_list) / len(v_list)
        std_v = (sum((x - mean_v)**2 for x in v_list) / (len(v_list) - 1))**0.5
        print(f"  {m:<20} | Seed 42: {s_vals['42']:+.4f} | Seed 123: {s_vals['123']:+.4f} | Seed 2026: {s_vals['2026']:+.4f} | Mean: {mean_v:+.4f} ± {std_v:.4f}")

print("\n" + "=" * 115)
print("  PAIRED PERMUTATION TEST P-VALUES (P=1,000 permutations)")
print("=" * 115)
for lang in languages:
    pt = data["permutation_tests"][lang]
    print(f"{lang.capitalize():<10} | GradInput vs RawAttn: p = {pt['p_val_grad_input_vs_raw']:.4f} | GradInput vs Random: p = {pt['p_val_grad_input_vs_random']:.4f} | GradReLU vs RawAttn: p = {pt['p_val_grad_relu_vs_raw']:.4f} | GradReLU vs Random: p = {pt['p_val_grad_relu_vs_random']:.4f}")

print("\n" + "=" * 115)
print("  INDIVIDUAL HEAD-WISE FAITHFULNESS GAPS (k=1)")
print("=" * 115)
for lang in languages:
    hw = data["head_wise_faithfulness"][lang]
    print(f"{lang.capitalize():<10} | Head 0: {hw['head_0']:+.4f} | Head 1: {hw['head_1']:+.4f} | Head 2: {hw['head_2']:+.4f} | Head 3: {hw['head_3']:+.4f}")

print("\n" + "=" * 115)
print("  CONTEXT DISTANCE COMPARISON (Attribution Mass %: Attention vs Input)")
print("=" * 115)
for lang in languages:
    dc = data["distance_attribution_comparison"][lang]
    print(f"\n[{lang.capitalize()}]")
    print(f"  {'Distance Bin':<15} | {'Attention Mass':<18} | {'Input Attribution Mass':<22}")
    print(f"  {'-'*60}")
    for k in ["dist_0_self", "dist_1", "dist_2", "dist_3", "dist_4_5", "dist_gt_5"]:
        print(f"  {k:<15} | {dc['attention'][k]:<17.2f}% | {dc['input_attribution'][k]:<21.2f}%")
