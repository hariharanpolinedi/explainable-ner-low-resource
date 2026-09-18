"""
Validation Suite & Scientific Verification for Faithfulness Experiments.

Executes:
1. Attribution Sanity Checks & Controls (Random, Uniform, Gradient-only)
2. Integrated Gradients Completeness & Step Convergence Sweep (m in {10, 20, 50})
3. Input Baseline Reference Sensitivity (Zero vs Mean vs UNK)
4. Multi-Seed Reproducibility Sweep (Seeds 42, 123, 2026)
5. Paired Permutation / Randomization Significance Testing (P=1,000)
6. Individual Head-Wise Faithfulness (Heads 0, 1, 2, 3)
7. Context Distance Profiling (Dist 0, 1, 2, 3, 4-5, >5)

Strictly on Gold-Dev (N=300, 100/lang). Zero access to Gold-Test.
"""

from __future__ import annotations

import json
import math
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
from experiments.faithfulness_improvement.erasure_engine import evaluate_confidence
from experiments.faithfulness_improvement.statistical_bootstrap import compute_full_metric_suite
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs


def compute_paired_permutation_pvalue(
    gaps_a: Sequence[float],
    gaps_b: Sequence[float],
    num_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired permutation test between method A and method B.
    H0: Mean(gap_A - gap_B) == 0.
    """
    diffs = [a - b for a, b in zip(gaps_a, gaps_b)]
    n = len(diffs)
    if n == 0:
        return 1.0

    observed_diff = abs(sum(diffs) / n)
    rng = random.Random(seed)
    count_extreme = 0

    for _ in range(num_permutations):
        perm_diffs = [d if rng.random() > 0.5 else -d for d in diffs]
        perm_mean = abs(sum(perm_diffs) / n)
        if perm_mean >= observed_diff:
            count_extreme += 1

    return count_extreme / num_permutations


def run_validation_suite(
    gold_dir: Path | str = "data/gold",
    proc_dir: Path | str = "data/processed",
    checkpoints_dir: Path | str = "checkpoints",
    sample_size: int = 100,
    output_dir: Path | str = "experiments/faithfulness_improvement/results",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Disable cuDNN to permit eval-mode backward passes through BiLSTM
    torch.backends.cudnn.enabled = False

    gold_dir = Path(gold_dir)
    proc_dir = Path(proc_dir)
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = ("konkani", "assamese", "sinhala")
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    validation_results: dict[str, Any] = {
        "metadata": {
            "sample_size": sample_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "sanity_checks": {},
        "ig_sweep": {},
        "baseline_sensitivity": {},
        "multi_seed_reproducibility": {},
        "permutation_tests": {},
        "head_wise_faithfulness": {},
        "distance_attribution_comparison": {},
    }

    for lang in languages:
        print(f"\n=========================================================================")
        print(f"  Running Scientific Validation Suite on {lang.upper()} (Gold-Dev)")
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

        # Extract dev instances
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

        rng_select = random.Random(42)
        if len(all_instances) > sample_size:
            selected_instances = rng_select.sample(all_instances, sample_size)
        else:
            selected_instances = all_instances

        print(f"Selected {len(selected_instances)} instances for validation on {lang}.")

        # -----------------------------------------------------------------
        # TEST 1: Sanity Checks & Control Attribution Methods
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Running Sanity Checks & Control Baselines...")
        sanity_methods = [
            "random_control",
            "uniform_control",
            "raw_attention",
            "grad_only_attn",
            "grad_only_input",
            "grad_x_attn_abs",
            "grad_x_attn_relu",
            "grad_x_input",
            "integrated_gradients",
        ]
        sanity_gaps: dict[str, list[float]] = {m: [] for m in sanity_methods}
        sanity_drops: dict[str, list[float]] = {m: [] for m in sanity_methods}
        sanity_rand_drops: dict[str, list[float]] = {m: [] for m in sanity_methods}

        for inst_i, inst in enumerate(selected_instances):
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            seq_len = len(tokens)
            if seq_len <= 1:
                continue

            valid_candidates = [i for i in range(seq_len) if i != q_idx]
            if not valid_candidates:
                continue

            trial_rng = random.Random(42 + inst_i)

            # Get target prediction
            w, c, l, mask = extractor.prepare_input_tensors(tokens)
            with torch.no_grad():
                paths = model.crf_shared.decode(model.fc_shared(model.encoder(model.concat_dropout(torch.cat([
                    model.word_embedding(w),
                    model.char_cnn(c) if model.use_char_cnn else torch.empty(1, seq_len, 0, device=device),
                    model.lang_embedding(l, seq_len)
                ], dim=-1)), mask=mask, return_hidden=True)), mask=mask)
                target_tag_id = paths[0][q_idx]

            orig_conf = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal")

            # Compute attribution vectors for each sanity condition
            attr_vectors: dict[str, list[float]] = {}

            # (a) Random control: random score for each token
            attr_vectors["random_control"] = [trial_rng.random() for _ in range(seq_len)]

            # (b) Uniform control: equal score
            attr_vectors["uniform_control"] = [1.0 / seq_len] * seq_len

            # (c) Raw attention
            raw_res = attributor.attribute(tokens, q_idx, target_tag_id, method="raw_attention")
            attr_vectors["raw_attention"] = raw_res["attribution_scores"].tolist()

            # (d) Grad-only attention: ||grad|| without attention weighting
            emb_drop = model.concat_dropout(torch.cat([
                model.word_embedding(w),
                model.char_cnn(c) if model.use_char_cnn else torch.empty(1, seq_len, 0, device=device),
                model.lang_embedding(l, seq_len)
            ], dim=-1))
            lstm_hidden = model.encoder(emb_drop, mask=mask, return_hidden=True)
            layer = model.attention.blocks[0]
            batch_size = 1
            q = layer.w_q(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = layer.w_k(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            v = layer.w_v(lstm_hidden).view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * layer.scale
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
            attn_w = torch.softmax(scores, dim=-1)
            attn_w.retain_grad()
            ctx = torch.matmul(layer.attn_dropout(attn_w), v).transpose(1, 2).contiguous().view(batch_size, seq_len, layer.hidden_dim)
            emiss = model.fc_shared(layer.layer_norm(lstm_hidden + layer.proj_dropout(layer.w_o(ctx))))
            emiss[0, q_idx, target_tag_id].backward()
            grad_only_a = attn_w.grad[0, :, q_idx, :].abs().mean(dim=0)
            attr_vectors["grad_only_attn"] = (grad_only_a / (grad_only_a.sum() + 1e-12)).tolist()

            # (e) Grad-only input: ||grad_E|| without E dot product
            comb = torch.cat([
                model.word_embedding(w),
                model.char_cnn(c) if model.use_char_cnn else torch.empty(1, seq_len, 0, device=device),
                model.lang_embedding(l, seq_len)
            ], dim=-1).clone().detach().requires_grad_(True)
            emiss_in = model.fc_shared(model.attention(model.encoder(model.concat_dropout(comb), mask=mask, return_hidden=True), mask=mask)[0])
            emiss_in[0, q_idx, target_tag_id].backward()
            grad_only_in = comb.grad[0].norm(p=2, dim=-1)
            attr_vectors["grad_only_input"] = (grad_only_in / (grad_only_in.sum() + 1e-12)).tolist()

            # (f) Grad x Attn Abs
            res_abs = attributor.attribute(tokens, q_idx, target_tag_id, method="grad_x_attn", aggregation="abs")
            attr_vectors["grad_x_attn_abs"] = res_abs["attribution_scores"].tolist()

            # (g) Grad x Attn Signed ReLU
            res_relu = attributor.attribute(tokens, q_idx, target_tag_id, method="grad_x_attn", aggregation="relu")
            attr_vectors["grad_x_attn_relu"] = res_relu["attribution_scores"].tolist()

            # (h) Grad x Input
            res_in = attributor.attribute(tokens, q_idx, target_tag_id, method="grad_x_input", aggregation="abs")
            attr_vectors["grad_x_input"] = res_in["attribution_scores"].tolist()

            # (i) Integrated Gradients
            res_ig = attributor.attribute(tokens, q_idx, target_tag_id, method="integrated_gradients", ig_steps=10)
            attr_vectors["integrated_gradients"] = res_ig["attribution_scores"].tolist()

            # Run k=1 erasure for all sanity methods under identical random control
            for m_name in sanity_methods:
                scores = attr_vectors[m_name]
                ranked = sorted(valid_candidates, key=lambda idx: scores[idx], reverse=True)
                top_1 = [ranked[0]]
                non_top = [i for i in valid_candidates if i not in top_1]
                rand_1 = trial_rng.sample(non_top, 1) if non_top else valid_candidates[:1]

                conf_top = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1)
                conf_rand = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1)

                drop_t = max(0.0, orig_conf - conf_top)
                drop_r = max(0.0, orig_conf - conf_rand)
                gap_1 = drop_t - drop_r

                sanity_drops[m_name].append(drop_t)
                sanity_rand_drops[m_name].append(drop_r)
                sanity_gaps[m_name].append(gap_1)

        # Summarize Sanity Checks
        validation_results["sanity_checks"][lang] = {
            m: compute_full_metric_suite(sanity_drops[m], sanity_rand_drops[m], sanity_gaps[m], seed=42)
            for m in sanity_methods
        }

        # -----------------------------------------------------------------
        # TEST 2: Integrated Gradients Step Sweep & Completeness Error Check
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Running Integrated Gradients Step Sweep & Completeness...")
        ig_steps_list = [10, 20, 50]
        ig_results: dict[int, list[float]] = {s: [] for s in ig_steps_list}
        completeness_errors: dict[int, list[float]] = {s: [] for s in ig_steps_list}

        for inst_i, inst in enumerate(selected_instances[:30]):  # Subset of 30 for high-step IG efficiency
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            seq_len = len(tokens)
            if seq_len <= 1:
                continue

            valid_candidates = [i for i in range(seq_len) if i != q_idx]
            if not valid_candidates:
                continue

            trial_rng = random.Random(100 + inst_i)

            w, c, l, mask = extractor.prepare_input_tensors(tokens)
            word_repr = model.word_embedding(w)
            lang_repr = model.lang_embedding(l, seq_len)
            char_repr = model.char_cnn(c) if model.use_char_cnn else torch.empty(1, seq_len, 0, device=device)
            input_emb = torch.cat([word_repr, char_repr, lang_repr], dim=-1).detach()
            baseline_emb = torch.zeros_like(input_emb)

            with torch.no_grad():
                init_emiss = model.fc_shared(model.attention(model.encoder(model.concat_dropout(input_emb), mask=mask, return_hidden=True), mask=mask)[0])
                base_emiss = model.fc_shared(model.attention(model.encoder(model.concat_dropout(baseline_emb), mask=mask, return_hidden=True), mask=mask)[0])
                target_tag_id = int(init_emiss[0, q_idx].argmax(dim=-1).item())
                delta_f = float((init_emiss[0, q_idx, target_tag_id] - base_emiss[0, q_idx, target_tag_id]).item())

            orig_conf = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal")

            for steps in ig_steps_list:
                total_grad = torch.zeros_like(input_emb)
                for step in range(1, steps + 1):
                    alpha = float(step) / steps
                    interpolated = (baseline_emb + alpha * (input_emb - baseline_emb)).requires_grad_(True)
                    model.zero_grad()
                    emiss_step = model.fc_shared(model.attention(model.encoder(model.concat_dropout(interpolated), mask=mask, return_hidden=True), mask=mask)[0])
                    emiss_step[0, q_idx, target_tag_id].backward()
                    total_grad = total_grad + interpolated.grad

                avg_grad = total_grad[0] / steps
                ig_scores = (input_emb[0] * avg_grad).sum(dim=-1)  # (seq_len,)
                ig_sum = float(ig_scores.sum().item())

                # Completeness relative error: |sum(IG) - delta_F| / (|delta_F| + 1e-6)
                comp_err = abs(ig_sum - delta_f) / (abs(delta_f) + 1e-6)
                completeness_errors[steps].append(comp_err)

                # Faithfulness gap at k=1
                scores_list = ig_scores.abs().tolist()
                ranked = sorted(valid_candidates, key=lambda idx: scores_list[idx], reverse=True)
                top_1 = [ranked[0]]
                non_top = [i for i in valid_candidates if i not in top_1]
                rand_1 = trial_rng.sample(non_top, 1) if non_top else valid_candidates[:1]

                conf_top = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1)
                conf_rand = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1)
                ig_results[steps].append((max(0.0, orig_conf - conf_top) - max(0.0, orig_conf - conf_rand)))

        validation_results["ig_sweep"][lang] = {
            f"steps_{s}": {
                "mean_gap": round(sum(ig_results[s]) / len(ig_results[s]), 4),
                "mean_completeness_error": round(sum(completeness_errors[s]) / len(completeness_errors[s]), 4),
            }
            for s in ig_steps_list
        }

        # -----------------------------------------------------------------
        # TEST 3: Input Baseline Reference Sensitivity
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Running Input Baseline Sensitivity...")
        # Compute mean embedding across dev tokens
        all_embs = []
        for inst in selected_instances[:20]:
            w, c, l, mask = extractor.prepare_input_tensors(inst["tokens"])
            with torch.no_grad():
                comb = torch.cat([
                    model.word_embedding(w),
                    model.char_cnn(c) if model.use_char_cnn else torch.empty(1, len(inst["tokens"]), 0, device=device),
                    model.lang_embedding(l, len(inst["tokens"]))
                ], dim=-1)
                all_embs.append(comb[0])
        mean_emb_vec = torch.cat(all_embs, dim=0).mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)

        # UNK embedding
        w_unk = torch.tensor([[1]], dtype=torch.long, device=device)
        c_unk = torch.tensor([[[1, 1, 1, 1, 1]]], dtype=torch.long, device=device)
        l_unk = torch.tensor([lang_id], dtype=torch.long, device=device)
        with torch.no_grad():
            unk_emb_vec = torch.cat([
                model.word_embedding(w_unk),
                model.char_cnn(c_unk) if model.use_char_cnn else torch.empty(1, 1, 0, device=device),
                model.lang_embedding(l_unk, 1)
            ], dim=-1)  # (1, 1, D)

        baseline_configs = {
            "zero_baseline": lambda emb: torch.zeros_like(emb),
            "mean_baseline": lambda emb: mean_emb_vec.expand_as(emb).clone(),
            "unk_baseline": lambda emb: unk_emb_vec.expand_as(emb).clone(),
        }
        base_gaps: dict[str, list[float]] = {b: [] for b in baseline_configs}

        for inst_i, inst in enumerate(selected_instances[:40]):
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            seq_len = len(tokens)
            valid_candidates = [i for i in range(seq_len) if i != q_idx]
            if not valid_candidates:
                continue

            trial_rng = random.Random(200 + inst_i)
            w, c, l, mask = extractor.prepare_input_tensors(tokens)
            with torch.no_grad():
                input_emb = torch.cat([
                    model.word_embedding(w),
                    model.char_cnn(c) if model.use_char_cnn else torch.empty(1, seq_len, 0, device=device),
                    model.lang_embedding(l, seq_len)
                ], dim=-1)
                target_tag_id = int(model.fc_shared(model.attention(model.encoder(model.concat_dropout(input_emb), mask=mask, return_hidden=True), mask=mask)[0])[0, q_idx].argmax(dim=-1).item())

            orig_conf = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal")

            for b_name, b_fn in baseline_configs.items():
                ref_emb = b_fn(input_emb)
                # Compute IG with m=10
                total_grad = torch.zeros_like(input_emb)
                for step in range(1, 11):
                    alpha = float(step) / 10
                    interp = (ref_emb + alpha * (input_emb - ref_emb)).requires_grad_(True)
                    model.zero_grad()
                    emiss_s = model.fc_shared(model.attention(model.encoder(model.concat_dropout(interp), mask=mask, return_hidden=True), mask=mask)[0])
                    emiss_s[0, q_idx, target_tag_id].backward()
                    total_grad = total_grad + interp.grad

                ig_sc = ((input_emb[0] - ref_emb[0]) * (total_grad[0] / 10)).sum(dim=-1).abs().tolist()
                ranked = sorted(valid_candidates, key=lambda idx: ig_sc[idx], reverse=True)
                top_1 = [ranked[0]]
                non_top = [i for i in valid_candidates if i not in top_1]
                rand_1 = trial_rng.sample(non_top, 1) if non_top else valid_candidates[:1]

                conf_top = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1)
                conf_rand = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1)
                base_gaps[b_name].append((max(0.0, orig_conf - conf_top) - max(0.0, orig_conf - conf_rand)))

        validation_results["baseline_sensitivity"][lang] = {
            b: round(sum(base_gaps[b]) / len(base_gaps[b]), 4)
            for b in baseline_configs
        }

        # -----------------------------------------------------------------
        # TEST 4: Multi-Seed Reproducibility Sweep
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Running Multi-Seed Reproducibility Sweep...")
        seeds = [42, 123, 2026]
        seed_results: dict[str, dict[int, float]] = {"grad_x_attn_relu": {}, "grad_x_input": {}}

        for s in seeds:
            s_rng = random.Random(s)
            s_sample = s_rng.sample(all_instances, min(sample_size, len(all_instances)))
            gaps_relu = []
            gaps_in = []

            for inst_i, inst in enumerate(s_sample):
                tokens = inst["tokens"]
                q_idx = inst["token_idx"]
                seq_len = len(tokens)
                valid_candidates = [i for i in range(seq_len) if i != q_idx]
                if not valid_candidates:
                    continue

                trial_rng = random.Random(s + inst_i)

                res_relu = attributor.attribute(tokens, q_idx, method="grad_x_attn", aggregation="relu")
                res_in = attributor.attribute(tokens, q_idx, method="grad_x_input", aggregation="abs")
                target_tag_id = res_relu["target_tag_id"]

                orig_conf = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal")

                # (a) Relu
                r_scores = res_relu["attribution_scores"].tolist()
                r_ranked = sorted(valid_candidates, key=lambda idx: r_scores[idx], reverse=True)
                top_1_r = [r_ranked[0]]
                non_top_r = [i for i in valid_candidates if i not in top_1_r]
                rand_1_r = trial_rng.sample(non_top_r, 1) if non_top_r else valid_candidates[:1]
                ct_r = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1_r)
                cr_r = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1_r)
                gaps_relu.append(max(0.0, orig_conf - ct_r) - max(0.0, orig_conf - cr_r))

                # (b) Input
                in_scores = res_in["attribution_scores"].tolist()
                in_ranked = sorted(valid_candidates, key=lambda idx: in_scores[idx], reverse=True)
                top_1_in = [in_ranked[0]]
                non_top_in = [i for i in valid_candidates if i not in top_1_in]
                rand_1_in = trial_rng.sample(non_top_in, 1) if non_top_in else valid_candidates[:1]
                ct_in = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1_in)
                cr_in = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1_in)
                gaps_in.append(max(0.0, orig_conf - ct_in) - max(0.0, orig_conf - cr_in))

            seed_results["grad_x_attn_relu"][s] = round(sum(gaps_relu) / len(gaps_relu), 4)
            seed_results["grad_x_input"][s] = round(sum(gaps_in) / len(gaps_in), 4)

        validation_results["multi_seed_reproducibility"][lang] = seed_results

        # -----------------------------------------------------------------
        # TEST 5: Permutation / Randomization Significance Testing
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Computing Paired Permutation Test p-values...")
        p_raw = compute_paired_permutation_pvalue(sanity_gaps["grad_x_input"], sanity_gaps["raw_attention"], num_permutations=1000)
        p_rand = compute_paired_permutation_pvalue(sanity_gaps["grad_x_input"], sanity_gaps["random_control"], num_permutations=1000)
        p_relu_raw = compute_paired_permutation_pvalue(sanity_gaps["grad_x_attn_relu"], sanity_gaps["raw_attention"], num_permutations=1000)
        p_relu_rand = compute_paired_permutation_pvalue(sanity_gaps["grad_x_attn_relu"], sanity_gaps["random_control"], num_permutations=1000)

        validation_results["permutation_tests"][lang] = {
            "p_val_grad_input_vs_raw": p_raw,
            "p_val_grad_input_vs_random": p_rand,
            "p_val_grad_relu_vs_raw": p_relu_raw,
            "p_val_grad_relu_vs_random": p_relu_rand,
        }

        # -----------------------------------------------------------------
        # TEST 6: Individual Head-Wise Faithfulness
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Evaluating Individual Head-Wise Faithfulness...")
        head_faith_gaps: dict[int, list[float]] = {h: [] for h in range(4)}
        for inst_i, inst in enumerate(selected_instances[:40]):
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            seq_len = len(tokens)
            valid_candidates = [i for i in range(seq_len) if i != q_idx]
            if not valid_candidates:
                continue

            trial_rng = random.Random(300 + inst_i)
            raw_res = extractor.extract_attention(tokens)
            attn_w = raw_res["attention_weights"]  # (4, seq_len, seq_len)
            target_tag_id = raw_res["predicted_tag_ids"][q_idx]
            orig_conf = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal")

            for h in range(4):
                h_row = attn_w[h, q_idx, :].tolist()
                ranked = sorted(valid_candidates, key=lambda idx: h_row[idx], reverse=True)
                top_1 = [ranked[0]]
                non_top = [i for i in valid_candidates if i not in top_1]
                rand_1 = trial_rng.sample(non_top, 1) if non_top else valid_candidates[:1]

                conf_top = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=top_1)
                conf_rand = evaluate_confidence(model, extractor, tokens, q_idx, target_tag_id, confidence_mode="crf_marginal", zero_embed_indices=rand_1)
                head_faith_gaps[h].append((max(0.0, orig_conf - conf_top) - max(0.0, orig_conf - conf_rand)))

        validation_results["head_wise_faithfulness"][lang] = {
            f"head_{h}": round(sum(head_faith_gaps[h]) / len(head_faith_gaps[h]), 4)
            for h in range(4)
        }

        # -----------------------------------------------------------------
        # TEST 7: Detailed Context Distance Comparison (Attention vs Input)
        # -----------------------------------------------------------------
        print(f"[{lang.upper()}] Profiling Distance Attribution (Attention vs Input)...")
        dist_bins = ["dist_0_self", "dist_1", "dist_2", "dist_3", "dist_4_5", "dist_gt_5"]
        attn_mass_accum = {d: 0.0 for d in dist_bins}
        input_mass_accum = {d: 0.0 for d in dist_bins}
        total_eval = 0

        for inst in selected_instances:
            tokens = inst["tokens"]
            q_idx = inst["token_idx"]
            seq_len = len(tokens)
            if seq_len <= 1:
                continue

            raw_attr = attributor.attribute(tokens, q_idx, method="raw_attention")["attribution_scores"]
            in_attr = attributor.attribute(tokens, q_idx, method="grad_x_input", aggregation="abs")["attribution_scores"]

            for j in range(seq_len):
                d = abs(j - q_idx)
                a_val = float(raw_attr[j].item())
                i_val = float(in_attr[j].item())

                if d == 0:
                    b = "dist_0_self"
                elif d == 1:
                    b = "dist_1"
                elif d == 2:
                    b = "dist_2"
                elif d == 3:
                    b = "dist_3"
                elif 4 <= d <= 5:
                    b = "dist_4_5"
                else:
                    b = "dist_gt_5"

                attn_mass_accum[b] += a_val
                input_mass_accum[b] += i_val
            total_eval += 1

        validation_results["distance_attribution_comparison"][lang] = {
            "attention": {k: round(v / total_eval * 100.0, 2) for k, v in attn_mass_accum.items()},
            "input_attribution": {k: round(v / total_eval * 100.0, 2) for k, v in input_mass_accum.items()},
        }

    # Macro Summaries
    print("\nComputing Macro Averages for Validation Suite...")
    macro_sanity: dict[str, dict[str, float]] = {}
    for m in sanity_methods:
        macro_gap = sum(validation_results["sanity_checks"][l][m]["mean_gap"] for l in languages) / len(languages)
        macro_win = sum(validation_results["sanity_checks"][l][m]["win_rate"] for l in languages) / len(languages)
        macro_sanity[m] = {
            "macro_gap": round(macro_gap, 4),
            "macro_win_rate": round(macro_win, 2),
        }
    validation_results["macro_sanity"] = macro_sanity

    # Save to disk
    out_file = output_dir / "validation_suite_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(validation_results, fp, indent=2)
    print(f"\n[SUCCESS] Validation results successfully saved to {out_file}")

    return validation_results


if __name__ == "__main__":
    run_validation_suite()
