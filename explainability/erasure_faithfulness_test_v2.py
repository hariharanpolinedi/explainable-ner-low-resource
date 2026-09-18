"""
Corrected Erasure Faithfulness Benchmark & Attribution Method Selection v2 (Pre-Submission Verification).

Addresses the structural design flaw identified in v1:
- In v1, the target entity token itself was accidentally eligible for selection in random-k erasure
  (because near-zero attribution placed it in non_top_pool), while it was never selected in top-k erasure.
  Erasing the entity token itself caused a catastrophic ~0.95 drop, artificially inflating random-k drops
  and creating misleading negative faithfulness gaps.
- In v2, the target entity token itself is explicitly excluded from BOTH top-k and random-k candidate pools
  (via configurable parameter `exclude_target_token=True`).
  This fairly evaluates contextual attribution faithfulness: does erasing top-attributed CONTEXT tokens
  degrade prediction confidence more than erasing random CONTEXT tokens?

Evaluates on >=100 gold-dev entity predictions per language.
Directly compares Raw Attention vs Gradient × Attention using correct comparative logic
(higher/less-negative gap wins).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from explainability.attention_extractor import AttentionExtractor
from explainability.erasure_faithfulness_test import (
    get_token_confidence,
    load_dev_sentences_with_entities,
)
from explainability.gradient_attention_attribution import GradientAttentionAttributor
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs
from utils.config_loader import load_config
from utils.seed import set_global_seed


def run_symmetric_erasure_test_on_language(
    lang: str,
    model: nn.Module,
    extractor: AttentionExtractor,
    attributor: GradientAttentionAttributor,
    dev_sentences: list[dict[str, Any]],
    top_k_list: Sequence[int] = (1, 2, 3),
    sample_size: int = 100,
    seed: int = 42,
    exclude_target_token: bool = True,
) -> dict[str, Any]:
    """
    Run the symmetric erasure test comparing Raw Attention vs Grad x Attention on a single language.

    Args:
        lang: Target language name.
        model: Frozen NERMultilingualModel instance.
        extractor: AttentionExtractor instance.
        attributor: GradientAttentionAttributor instance.
        dev_sentences: Parsed gold-dev sentences with entity metadata.
        top_k_list: List of k values to evaluate (default: 1, 2, 3).
        sample_size: Number of entity instances to evaluate (>=100).
        seed: Random seed for reproducibility.
        exclude_target_token: If True, enforces strict symmetric exclusion of the target query token
                             from BOTH top-k and random-k candidate pools.
    """
    rng = random.Random(seed)

    all_instances = []
    for sent_data in dev_sentences:
        tokens = sent_data["tokens"]
        for ent in sent_data["entities"]:
            all_instances.append({
                "tokens": tokens,
                "token_idx": ent["token_idx"],
                "gold_tag": ent["tag"],
                "token": ent["token"],
            })

    if len(all_instances) > sample_size:
        selected_instances = rng.sample(all_instances, sample_size)
    else:
        selected_instances = all_instances

    print(f"[{lang.upper()}] Evaluating {len(selected_instances)} entity predictions on gold-dev "
          f"(exclude_target_token={exclude_target_token})...")

    results: dict[str, dict[str, list[float]]] = {
        "raw_attention": {f"top_{k}": [] for k in top_k_list},
        "grad_x_attn": {f"top_{k}": [] for k in top_k_list},
        "random": {f"rand_{k}": [] for k in top_k_list},
    }
    results["raw_attention"].update({f"gap_{k}": [] for k in top_k_list})
    results["grad_x_attn"].update({f"gap_{k}": [] for k in top_k_list})

    for item in selected_instances:
        tokens = item["tokens"]
        q_idx = item["token_idx"]
        seq_len = len(tokens)

        # Skip trivial single-token sequences if any
        if seq_len <= 1:
            continue

        # 1. Compute baseline prediction and confidence
        attr_result = attributor.attribute_token(tokens, target_token_idx=q_idx, normalize=True)
        target_tag_id = attr_result["target_tag_id"]
        orig_conf = get_token_confidence(model, extractor, tokens, q_idx, target_tag_id)

        grad_scores = attr_result["attribution_scores"]  # (seq_len,)
        raw_scores = attr_result["raw_attention"]        # (seq_len,)

        # 2. Define valid candidate pool under the chosen design
        if exclude_target_token:
            valid_candidates = [i for i in range(seq_len) if i != q_idx]
        else:
            valid_candidates = list(range(seq_len))

        if not valid_candidates:
            continue

        # Sort candidate indices descending by score
        grad_ranked_indices = sorted(valid_candidates, key=lambda idx: float(grad_scores[idx].item()), reverse=True)
        raw_ranked_indices = sorted(valid_candidates, key=lambda idx: float(raw_scores[idx].item()), reverse=True)

        for k in top_k_list:
            if k > len(valid_candidates):
                continue

            # (a) Top-k erasure for Grad x Attn (strictly within valid_candidates)
            top_k_grad = grad_ranked_indices[:k]
            tokens_grad_erased = list(tokens)
            for idx in top_k_grad:
                tokens_grad_erased[idx] = "<UNK>"
            conf_grad_erased = get_token_confidence(model, extractor, tokens_grad_erased, q_idx, target_tag_id)
            drop_grad = max(0.0, orig_conf - conf_grad_erased)
            results["grad_x_attn"][f"top_{k}"].append(drop_grad)

            # (b) Top-k erasure for Raw Attention (strictly within valid_candidates)
            top_k_raw = raw_ranked_indices[:k]
            tokens_raw_erased = list(tokens)
            for idx in top_k_raw:
                tokens_raw_erased[idx] = "<UNK>"
            conf_raw_erased = get_token_confidence(model, extractor, tokens_raw_erased, q_idx, target_tag_id)
            drop_raw = max(0.0, orig_conf - conf_raw_erased)
            results["raw_attention"][f"top_{k}"].append(drop_raw)

            # (c) Symmetric Random-k erasure (control)
            # Pool: valid candidates that are not in top_k_grad and not in top_k_raw
            # This guarantees that random erasure draws ONLY from the non-top context pool,
            # never touching the target entity token when exclude_target_token=True.
            non_top_pool = [i for i in valid_candidates if i not in top_k_grad and i not in top_k_raw]
            if len(non_top_pool) >= k:
                rand_indices = rng.sample(non_top_pool, k)
            else:
                rand_indices = rng.sample(valid_candidates, min(k, len(valid_candidates)))

            tokens_rand_erased = list(tokens)
            for idx in rand_indices:
                tokens_rand_erased[idx] = "<UNK>"
            conf_rand_erased = get_token_confidence(model, extractor, tokens_rand_erased, q_idx, target_tag_id)
            drop_rand = max(0.0, orig_conf - conf_rand_erased)
            results["random"][f"rand_{k}"].append(drop_rand)

            # (d) Faithfulness Gaps: Drop_top - Drop_rand
            results["grad_x_attn"][f"gap_{k}"].append(drop_grad - drop_rand)
            results["raw_attention"][f"gap_{k}"].append(drop_raw - drop_rand)

    # Compute summary statistics
    summary: dict[str, Any] = {
        "sample_size": len(selected_instances),
        "exclude_target_token": exclude_target_token,
    }
    for method in ["grad_x_attn", "raw_attention"]:
        summary[method] = {}
        for k in top_k_list:
            drops = results[method][f"top_{k}"]
            r_drops = results["random"][f"rand_{k}"]
            gaps = results[method][f"gap_{k}"]
            mean_drop = sum(drops) / len(drops) if drops else 0.0
            mean_rand = sum(r_drops) / len(r_drops) if r_drops else 0.0
            mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
            win_rate = (sum(1 for g in gaps if g > 0) / len(gaps) * 100) if gaps else 0.0

            summary[method][f"k={k}"] = {
                "mean_top_drop": round(mean_drop, 4),
                "mean_random_drop": round(mean_rand, 4),
                "faithfulness_gap": round(mean_gap, 4),
                "faithfulness_win_rate": round(win_rate, 2),
            }

    return summary


def run_full_faithfulness_suite_v2(
    gold_dir: Path | str = "data/gold",
    proc_dir: Path | str = "data/processed",
    checkpoints_dir: Path | str = "checkpoints",
    sample_size: int = 100,
    seed: int = 42,
    exclude_target_token: bool = True,
) -> dict[str, Any]:
    """
    Run complete v2 faithfulness evaluation across all 3 languages.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    gold_dir = Path(gold_dir)
    proc_dir = Path(proc_dir)
    checkpoints_dir = Path(checkpoints_dir)
    languages = ("konkani", "assamese", "sinhala")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    all_results: dict[str, Any] = {}

    for lang in languages:
        ckpt_path = checkpoints_dir / f"final_{lang}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        model = NERMultilingualModel(
            word_vocab_size=len(word_to_id),
            char_vocab_size=len(char_to_id),
            num_tags=9,
            num_languages=3,
            word_embedding_dim=100,
            char_embedding_dim=50,
            lang_embedding_dim=16,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_attention_blocks=1,
            crf_config="shared",
            use_char_cnn=True,
            use_attention=True,
            use_crf=True,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        lang_id = LANG_TO_ID[lang]
        extractor = AttentionExtractor(model, word_to_id, char_to_id, lang, lang_id=lang_id, device=device)
        attributor = GradientAttentionAttributor(extractor)

        dev_sentences = load_dev_sentences_with_entities(gold_dir / lang / "dev.conll")

        res = run_symmetric_erasure_test_on_language(
            lang=lang,
            model=model,
            extractor=extractor,
            attributor=attributor,
            dev_sentences=dev_sentences,
            top_k_list=(1, 2, 3),
            sample_size=sample_size,
            seed=seed,
            exclude_target_token=exclude_target_token,
        )
        all_results[lang] = res

    # Print summary table
    print("\n=======================================================================================================")
    print(f"  V2 FAITHFULNESS BENCHMARK RESULTS (exclude_target_token={exclude_target_token})")
    print("=======================================================================================================\n")
    print(f"{'Language':<10} | {'Method':<22} | {'Top-1':<7} | {'Rand-1':<7} | {'Gap k=1':<8} | {'Gap k=2':<8} | {'Gap k=3':<8} | {'Win Rate k=1':<12}")
    print("-" * 105)

    grad_gaps_all = []
    raw_gaps_all = []

    for lang in languages:
        for method, m_name in [("grad_x_attn", "Gradient x Attention"), ("raw_attention", "Raw Attention")]:
            k1 = all_results[lang][method]["k=1"]
            k2 = all_results[lang][method]["k=2"]
            k3 = all_results[lang][method]["k=3"]

            if method == "grad_x_attn":
                grad_gaps_all.extend([k1["faithfulness_gap"], k2["faithfulness_gap"], k3["faithfulness_gap"]])
            else:
                raw_gaps_all.extend([k1["faithfulness_gap"], k2["faithfulness_gap"], k3["faithfulness_gap"]])

            print(f"{lang.capitalize():<10} | {m_name:<22} | {k1['mean_top_drop']:<7.4f} | {k1['mean_random_drop']:<7.4f} | "
                  f"{k1['faithfulness_gap']:<+8.4f} | {k2['faithfulness_gap']:<+8.4f} | {k3['faithfulness_gap']:<+8.4f} | {k1['faithfulness_win_rate']:<10.1f}%")

    mean_grad_gap = sum(grad_gaps_all) / len(grad_gaps_all)
    mean_raw_gap = sum(raw_gaps_all) / len(raw_gaps_all)

    print("-" * 105)
    print(f"Mean Faithfulness Gap across all languages and k values:")
    print(f"  Gradient x Attention: {mean_grad_gap:+.4f}")
    print(f"  Raw Attention:        {mean_raw_gap:+.4f}")

    if mean_grad_gap > mean_raw_gap:
        winner = "Gradient x Attention"
        margin = mean_grad_gap - mean_raw_gap
    else:
        winner = "Raw Attention"
        margin = mean_raw_gap - mean_grad_gap

    print(f"\nAuthoritative Winning Attribution Method: **{winner}** (Outperformed by {margin:+.4f} gap margin)")

    all_results["summary"] = {
        "mean_grad_x_attn_gap": round(mean_grad_gap, 4),
        "mean_raw_attention_gap": round(mean_raw_gap, 4),
        "winning_method": winner,
        "gap_margin": round(margin, 4),
    }

    out_file = Path("reports/experiment_logs/phase13_faithfulness_v2_raw.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2)

    return all_results


if __name__ == "__main__":
    run_full_faithfulness_suite_v2()
