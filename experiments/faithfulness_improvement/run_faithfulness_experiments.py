"""
Master Orchestration Script for Faithfulness Improvement Experiment.

Executes controlled, reproducible evaluation on locked Gold-Dev (N=300, 100/lang).
Evaluates multiple attribution methods, targets, and erasure strategies.
Strictly zero access to Gold-Test.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from explainability.attention_extractor import AttentionExtractor
from explainability.erasure_faithfulness_test import load_dev_sentences_with_entities
from experiments.faithfulness_improvement.attribution_methods import UnifiedAttributor
from experiments.faithfulness_improvement.erasure_engine import evaluate_confidence, run_symmetric_erasure_trial
from experiments.faithfulness_improvement.head_distance_analysis import (
    analyze_context_distance_profile,
    analyze_head_profiles,
)
from experiments.faithfulness_improvement.statistical_bootstrap import compute_full_metric_suite
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs


def run_faithfulness_investigation(
    gold_dir: Path | str = "data/gold",
    proc_dir: Path | str = "data/processed",
    checkpoints_dir: Path | str = "checkpoints",
    sample_size: int = 100,
    seed: int = 42,
    output_dir: Path | str = "experiments/faithfulness_improvement/results",
) -> dict[str, Any]:
    """
    Execute complete faithfulness improvement evaluation on Gold-Dev.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    gold_dir = Path(gold_dir)
    proc_dir = Path(proc_dir)
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = ("konkani", "assamese", "sinhala")
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    # Methods to evaluate:
    # 1. raw_attention: Head-averaged Raw Attention (baseline winner)
    # 2. grad_x_attn_abs: Grad x Attn (Baseline abs, emission logit)
    # 3. grad_x_attn_relu: Grad x Attn (Signed ReLU, emission logit)
    # 4. grad_x_attn_lsm: Grad x Attn (Signed ReLU, log-softmax)
    # 5. grad_x_attn_crf: Grad x Attn (CRF-aware sequence score)
    # 6. grad_x_input: Gradient x Input Embedding (end-to-end BiLSTM+Attn+CRF)
    # 7. integrated_gradients: Integrated Gradients on Input Embeddings (m=10)
    # 8. raw_head1: Raw Attention Head 1 (Local Boundary specialist)
    methods_config = [
        {"name": "raw_attention", "method": "raw_attention", "target": "emission", "agg": "abs", "head": None},
        {"name": "grad_x_attn_abs", "method": "grad_x_attn", "target": "emission", "agg": "abs", "head": None},
        {"name": "grad_x_attn_relu", "method": "grad_x_attn", "target": "emission", "agg": "relu", "head": None},
        {"name": "grad_x_attn_lsm", "method": "grad_x_attn", "target": "log_softmax", "agg": "relu", "head": None},
        {"name": "grad_x_attn_crf", "method": "grad_x_attn", "target": "crf_sequence", "agg": "relu", "head": None},
        {"name": "grad_x_input", "method": "grad_x_input", "target": "emission", "agg": "relu", "head": None},
        {"name": "integrated_gradients", "method": "integrated_gradients", "target": "emission", "agg": "relu", "head": None},
        {"name": "raw_head1", "method": "raw_attention", "target": "emission", "agg": "abs", "head": 1},
    ]

    top_k_list = [1, 2, 3]

    master_results: dict[str, Any] = {
        "metadata": {
            "sample_size_per_lang": sample_size,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "languages": {},
        "macro_summary": {},
        "entity_type_analysis": {},
        "distance_analysis": {},
        "head_analysis": {},
        "qualitative_cases": [],
    }

    rng = random.Random(seed)

    for lang in languages:
        print(f"\n=========================================================================")
        print(f"  Evaluating Faithfulness Investigation on {lang.upper()} (Gold-Dev)")
        print(f"=========================================================================")

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
        attributor = UnifiedAttributor(extractor)

        dev_sentences = load_dev_sentences_with_entities(gold_dir / lang / "dev.conll")

        all_instances = []
        for sent_idx, sent_data in enumerate(dev_sentences):
            tokens = sent_data["tokens"]
            for ent in sent_data["entities"]:
                all_instances.append({
                    "sentence_idx": sent_idx,
                    "tokens": tokens,
                    "token_idx": ent["token_idx"],
                    "gold_tag": ent["tag"],
                    "token": ent["token"],
                    "entity_type": ent["tag"].split("-")[-1],
                })

        if len(all_instances) > sample_size:
            selected_instances = rng.sample(all_instances, sample_size)
        else:
            selected_instances = all_instances

        print(f"Selected {len(selected_instances)} entity instances on gold-dev.")

        # Data structures to accumulate drops across instances
        # lang -> method_name -> k -> [drops]
        method_data: dict[str, dict[str, dict[str, list[float]]]] = {
            m["name"]: {
                # Protocol A: zero_embedding + crf_marginal (primary proposed)
                "zero_crf": {f"top_{k}": [] for k in top_k_list},
                # Protocol B: unk_token + emission_softmax (exact historical baseline match)
                "unk_emission": {f"top_{k}": [] for k in top_k_list},
            }
            for m in methods_config
        }
        for m in methods_config:
            for proto in ["zero_crf", "unk_emission"]:
                method_data[m["name"]][proto].update({f"rand_{k}": [] for k in top_k_list})
                method_data[m["name"]][proto].update({f"gap_{k}": [] for k in top_k_list})

        # Entity type breakdown containers
        entity_breakdown: dict[str, dict[str, list[float]]] = {}
        # Distance analysis container
        distance_profile_list: list[dict[str, float]] = []
        # Head analysis container
        head_metrics_list: list[dict[int, dict[str, float]]] = []

        # Run instance loop
        for inst_i, inst in enumerate(selected_instances):
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            gold_tag = inst["gold_tag"]
            ent_type = inst["entity_type"]

            if len(tokens) <= 1:
                continue

            # Shared trial RNG for matched random-k selection across methods
            instance_seed = seed + inst_i

            # Extract attention weights for head analysis
            res_attn = extractor.extract_attention(tokens)
            attn_weights = res_attn["attention_weights"]
            head_stats = analyze_head_profiles(attn_weights, q_idx)
            head_metrics_list.append(head_stats)

            # Evaluate each attribution method
            for m_cfg in methods_config:
                m_name = m_cfg["name"]
                attr_res = attributor.attribute(
                    tokens=tokens,
                    target_token_idx=q_idx,
                    method=m_cfg["method"],
                    target_type=m_cfg["target"],
                    aggregation=m_cfg["agg"],
                    head_idx=m_cfg["head"],
                    ig_steps=10,
                )
                attr_scores = attr_res["attribution_scores"]
                target_tag_id = attr_res["target_tag_id"]

                # Distance profiling for Raw Attention and GradxInput
                if m_name == "raw_attention":
                    dist_prof = analyze_context_distance_profile(attr_scores, q_idx)
                    distance_profile_list.append(dist_prof)

                # 1. Primary Protocol: Zero-Embedding + CRF Marginal Confidence
                trial_rng_zero = random.Random(instance_seed)
                trial_zero = run_symmetric_erasure_trial(
                    model=model,
                    extractor=extractor,
                    tokens=tokens,
                    target_token_idx=q_idx,
                    target_tag_id=target_tag_id,
                    attribution_scores=attr_scores,
                    k_list=top_k_list,
                    erasure_strategy="zero_embedding",
                    confidence_mode="crf_marginal",
                    rng=trial_rng_zero,
                )

                # 2. Baseline Protocol: UNK Token + Emission Softmax Confidence
                trial_rng_unk = random.Random(instance_seed)
                trial_unk = run_symmetric_erasure_trial(
                    model=model,
                    extractor=extractor,
                    tokens=tokens,
                    target_token_idx=q_idx,
                    target_tag_id=target_tag_id,
                    attribution_scores=attr_scores,
                    k_list=top_k_list,
                    erasure_strategy="unk_token",
                    confidence_mode="emission_softmax",
                    rng=trial_rng_unk,
                )

                # Record metrics
                for k in top_k_list:
                    k_key = f"k={k}"
                    if k_key in trial_zero:
                        method_data[m_name]["zero_crf"][f"top_{k}"].append(trial_zero[k_key]["drop_top"])
                        method_data[m_name]["zero_crf"][f"rand_{k}"].append(trial_zero[k_key]["drop_rand"])
                        method_data[m_name]["zero_crf"][f"gap_{k}"].append(trial_zero[k_key]["gap"])

                    if k_key in trial_unk:
                        method_data[m_name]["unk_emission"][f"top_{k}"].append(trial_unk[k_key]["drop_top"])
                        method_data[m_name]["unk_emission"][f"rand_{k}"].append(trial_unk[k_key]["drop_rand"])
                        method_data[m_name]["unk_emission"][f"gap_{k}"].append(trial_unk[k_key]["gap"])

                # Entity type tracking for k=1 under zero_crf
                if f"k=1" in trial_zero:
                    if ent_type not in entity_breakdown:
                        entity_breakdown[ent_type] = {}
                    if m_name not in entity_breakdown[ent_type]:
                        entity_breakdown[ent_type][m_name] = []
                    entity_breakdown[ent_type][m_name].append(trial_zero["k=1"]["gap"])

            # Collect qualitative case studies (select first 2 successful, 2 weak, 1 failure)
            if len(master_results["qualitative_cases"]) < 9:
                raw_attr_res = attributor.attribute(tokens, q_idx, method="raw_attention")
                top_ctx_idx = sorted(
                    [i for i in range(len(tokens)) if i != q_idx],
                    key=lambda idx: float(raw_attr_res["attribution_scores"][idx].item()),
                    reverse=True,
                )[:3]
                top_ctx_words = [tokens[i] for i in top_ctx_idx]
                master_results["qualitative_cases"].append({
                    "lang": lang,
                    "tokens": tokens,
                    "target_token": tokens[q_idx],
                    "target_token_idx": q_idx,
                    "gold_tag": gold_tag,
                    "top_attributed_context": top_ctx_words,
                })

        # Summarize language stats
        lang_summary: dict[str, Any] = {}
        for m_cfg in methods_config:
            m_name = m_cfg["name"]
            lang_summary[m_name] = {}
            for proto in ["zero_crf", "unk_emission"]:
                lang_summary[m_name][proto] = {}
                for k in top_k_list:
                    top_d = method_data[m_name][proto][f"top_{k}"]
                    rand_d = method_data[m_name][proto][f"rand_{k}"]
                    gaps = method_data[m_name][proto][f"gap_{k}"]
                    lang_summary[m_name][proto][f"k={k}"] = compute_full_metric_suite(top_d, rand_d, gaps, seed=seed)

        master_results["languages"][lang] = lang_summary

        # Summarize distance profile for language
        if distance_profile_list:
            avg_dist = {
                k: round(sum(d[k] for d in distance_profile_list) / len(distance_profile_list), 2)
                for k in ["dist_1", "dist_2", "dist_3_5", "dist_gt_5"]
            }
            master_results["distance_analysis"][lang] = avg_dist

        # Summarize head metrics for language
        if head_metrics_list:
            head_avg: dict[int, dict[str, float]] = {}
            for h in range(4):
                head_avg[h] = {
                    metric: round(sum(inst[h][metric] for inst in head_metrics_list) / len(head_metrics_list), 2)
                    for metric in ["entropy", "avg_distance", "self_prop", "immediate_prop", "distant_prop"]
                }
            master_results["head_analysis"][lang] = head_avg

        # Entity breakdown summary
        master_results["entity_type_analysis"][lang] = {
            ent: {
                m_name: round(sum(vals) / len(vals), 4) if vals else 0.0
                for m_name, vals in m_dict.items()
            }
            for ent, m_dict in entity_breakdown.items()
        }

    # Macro summary across languages
    macro_summary: dict[str, Any] = {}
    for m_cfg in methods_config:
        m_name = m_cfg["name"]
        macro_summary[m_name] = {}
        for proto in ["zero_crf", "unk_emission"]:
            macro_summary[m_name][proto] = {}
            for k in top_k_list:
                gaps_k = [master_results["languages"][l][m_name][proto][f"k={k}"]["mean_gap"] for l in languages]
                wins_k = [master_results["languages"][l][m_name][proto][f"k={k}"]["win_rate"] for l in languages]
                macro_summary[m_name][proto][f"k={k}"] = {
                    "macro_mean_gap": round(sum(gaps_k) / len(gaps_k), 4),
                    "macro_win_rate": round(sum(wins_k) / len(wins_k), 2),
                }

    master_results["macro_summary"] = macro_summary

    # Save to disk
    out_file = output_dir / "faithfulness_experiment_master.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(master_results, fp, indent=2)
    print(f"\n[SUCCESS] Master results successfully saved to {out_file}")

    return master_results


if __name__ == "__main__":
    run_faithfulness_investigation()
