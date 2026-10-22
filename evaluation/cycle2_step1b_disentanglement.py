"""
Reliability Cycle 2, Step 1b: Assamese Emission-Bias vs. Attention-Degeneracy Causal Disentanglement.

This diagnostic autonomously performs:
1. Full-corpus label-consistency audit on data/gold/assamese/train.conll (for all tokens with >=5 occurrences).
2. Emission-bias-vs-consistency correlation analysis on the 270-sentence Assamese manifest (4-quadrant table).
3. Order-of-onset test: token-by-token and prefix-truncation evaluation on L3/L4 collapsed sentences.
4. Independent-cause test: counts of attention degeneracy without emission O-bias across the manifest.
5. Emits numerical results and structured JSON for reporting.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from evaluation.cycle2_root_cause_diagnostic import load_model
from evaluation.phase1_error_taxonomy_study import extract_entities
from evaluation.phase5_targeted_interventions import (
    ID_TO_TAG,
    TAG_TO_ID,
    load_conll_data,
    load_or_create_multilingual_vocabs,
    set_global_seed,
)
from evaluation.phase6_combined_retraining import load_diagnostic_manifest_sents
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import LANG_TO_ID

set_global_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "improved_assamese_v2.pt"


def run_full_corpus_consistency_audit(train_conll_path: Path, min_freq: int = 5):
    """
    Computes label consistency for all token types in train_conll_path with occurrences >= min_freq.
    Consistency rate = count(most common tag) / total occurrences.
    """
    lines = train_conll_path.read_text(encoding="utf-8").splitlines()
    token_tag_counts = defaultdict(Counter)
    total_tokens = 0
    total_sentences = 0

    for line in lines:
        line = line.strip()
        if not line:
            total_sentences += 1
            continue
        parts = line.split()
        if len(parts) >= 2:
            tok = parts[0]
            tag = parts[-1]
            token_tag_counts[tok][tag] += 1
            total_tokens += 1

    total_distinct_types = len(token_tag_counts)
    qualifying_tokens = {tok: cnts for tok, cnts in token_tag_counts.items() if sum(cnts.values()) >= min_freq}

    consistency_records = []
    for tok, cnts in qualifying_tokens.items():
        total_cnt = sum(cnts.values())
        top_tag, top_cnt = cnts.most_common(1)[0]
        rate = top_cnt / total_cnt
        o_cnt = cnts.get("O", 0)
        ent_cnt = total_cnt - o_cnt
        o_rate = o_cnt / total_cnt
        consistency_records.append({
            "token": tok,
            "total_count": total_cnt,
            "top_tag": top_tag,
            "top_tag_count": top_cnt,
            "consistency_rate": round(rate, 4),
            "o_count": o_cnt,
            "entity_count": ent_cnt,
            "o_rate": round(o_rate, 4),
            "tag_distribution": dict(cnts),
        })

    # Sort least-consistent to most-consistent, then by total_count descending
    consistency_records.sort(key=lambda r: (r["consistency_rate"], -r["total_count"]))

    rates = [r["consistency_rate"] for r in consistency_records]
    stats = {
        "total_tokens_in_train": total_tokens,
        "total_sentences_in_train": total_sentences,
        "total_distinct_types": total_distinct_types,
        "qualifying_types_ge_5": len(consistency_records),
        "min": float(np.min(rates)),
        "p10": float(np.percentile(rates, 10)),
        "p25": float(np.percentile(rates, 25)),
        "median": float(np.median(rates)),
        "p75": float(np.percentile(rates, 75)),
        "p90": float(np.percentile(rates, 90)),
        "max": float(np.max(rates)),
        "count_perfect_1_00": sum(1 for r in rates if r == 1.0),
        "count_inconsistent_lt_1_00": sum(1 for r in rates if r < 1.0),
        "count_lt_0_90": sum(1 for r in rates if r < 0.90),
        "count_lt_0_85": sum(1 for r in rates if r < 0.85),
        "count_lt_0_80": sum(1 for r in rates if r < 0.80),
        "count_lt_0_70": sum(1 for r in rates if r < 0.70),
    }

    return consistency_records, stats, token_tag_counts


def run_emission_and_attention_manifest_extraction(
    model: nn.Module,
    manifest: list[tuple[list[str], list[str], str]],
    w_map: dict[str, int],
    c_map: dict[str, int],
):
    """
    Extracts token-level emission logits, emission O-bias gaps, and multi-head attention entropy ratios
    for all tokens across all sentences in the Assamese diagnostic manifest.
    """
    tokenizer = ScriptTokenizer("assamese")
    manifest_token_records = []
    sentence_records = []

    o_tag_id = TAG_TO_ID["O"]
    entity_tag_ids = [v for k, v in TAG_TO_ID.items() if k != "O"]

    with torch.no_grad():
        for sent_idx, (toks, gold, src) in enumerate(manifest):
            seq_len = len(toks)
            if seq_len == 0:
                continue

            tok_gs = [tokenizer.tokenize_characters(w) for w in toks]
            max_c = max(len(gs) for gs in tok_gs) if tok_gs else 1

            w_b = torch.zeros((1, seq_len), dtype=torch.long)
            c_b = torch.zeros((1, seq_len, max_c), dtype=torch.long)
            l_b = torch.tensor([LANG_TO_ID["assamese"]], dtype=torch.long)
            m_b = torch.ones((1, seq_len), dtype=torch.bool)

            for j, (w, gs) in enumerate(zip(toks, tok_gs)):
                w_b[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
                for k, g in enumerate(gs):
                    c_b[0, j, k] = c_map.get(g, c_map.get("<UNK>", 1))

            w_emb = model.word_embedding(w_b.to(DEVICE))
            c_emb = model.char_cnn(c_b.to(DEVICE))
            l_emb = model.lang_embedding(l_b.to(DEVICE), seq_len)
            x_in = torch.cat([w_emb, c_emb, l_emb], dim=-1)
            x_in = model.concat_dropout(x_in)

            lstm_hidden = model.encoder(x_in, mask=m_b.to(DEVICE), return_hidden=True)
            attended_states, attn_weights_blocks = model.attention(lstm_hidden, mask=m_b.to(DEVICE))
            attn_weights = attn_weights_blocks[0] if isinstance(attn_weights_blocks, list) else attn_weights_blocks

            if model.crf_config == "shared":
                emissions = model.fc_shared(attended_states)
            else:
                emissions = model.fc_heads[l_b[0].item()](attended_states)

            # emissions: [1, seq_len, num_tags]
            # attn_weights: [1, num_heads, seq_len, seq_len]
            num_heads = attn_weights.shape[1]
            p = attn_weights[0]  # [num_heads, seq_len, seq_len]

            # Token-level query entropy across keys:
            # - sum_j p_{h, i, j} * log(p_{h, i, j})
            # Average across heads
            token_entropy_per_head = -(p * torch.log(p + 1e-12)).sum(dim=-1)  # [num_heads, seq_len]
            token_mean_entropy = token_entropy_per_head.mean(dim=0).cpu().numpy()  # [seq_len]

            max_possible_ent = math.log(seq_len) if seq_len > 1 else 1.0
            token_ent_ratio = token_mean_entropy / max_possible_ent if seq_len > 1 else np.zeros(seq_len)

            o_logits = emissions[0, :, o_tag_id].cpu().numpy()
            ent_logits = emissions[0, :, entity_tag_ids]
            max_ent_logits, max_ent_arg = ent_logits.max(dim=-1)
            max_ent_logits = max_ent_logits.cpu().numpy()
            max_ent_arg = max_ent_arg.cpu().numpy()

            o_bias_gaps = o_logits - max_ent_logits

            sent_record = {
                "sent_idx": sent_idx,
                "length": seq_len,
                "source": src,
                "mean_ent_ratio": float(np.mean(token_ent_ratio)),
                "mean_o_gap": float(np.mean(o_bias_gaps)),
                "gold_ents": extract_entities(gold),
            }
            sentence_records.append(sent_record)

            for pos in range(seq_len):
                best_ent_tag_id = entity_tag_ids[max_ent_arg[pos]]
                best_ent_tag = ID_TO_TAG[best_ent_tag_id]
                manifest_token_records.append({
                    "sent_idx": sent_idx,
                    "pos": pos,
                    "token": toks[pos],
                    "gold_tag": gold[pos],
                    "is_gold_entity": gold[pos] != "O",
                    "o_logit": float(round(o_logits[pos], 3)),
                    "max_ent_logit": float(round(max_ent_logits[pos], 3)),
                    "best_ent_tag": best_ent_tag,
                    "o_bias_gap": float(round(o_bias_gaps[pos], 3)),
                    "entropy": float(round(token_mean_entropy[pos], 3)),
                    "entropy_ratio": float(round(token_ent_ratio[pos], 4)),
                    "seq_len": seq_len,
                })

    return manifest_token_records, sentence_records


def run_order_of_onset_analysis(
    model: nn.Module,
    w_map: dict[str, int],
    c_map: dict[str, int],
    collapsed_cases: list[dict],
):
    """
    Traces token-by-token and prefix-by-prefix how emission O-bias and attention entropy evolve.
    Tests prefix lengths: 1, 2, 3, 5, 8, 12, 18, 25, ... up to full sentence length.
    """
    tokenizer = ScriptTokenizer("assamese")
    results = []
    o_tag_id = TAG_TO_ID["O"]
    entity_tag_ids = [v for k, v in TAG_TO_ID.items() if k != "O"]

    prefix_lengths = [1, 2, 3, 5, 8, 12, 18, 25, 34, 42, 53, 66]

    with torch.no_grad():
        for case in collapsed_cases:
            s_idx = case["sent_idx"]
            toks = case["tokens"]
            gold = case["gold"]
            full_len = len(toks)
            gold_ents = extract_entities(gold)

            # 1. Full-sentence token-by-token trace
            tok_gs = [tokenizer.tokenize_characters(w) for w in toks]
            max_c = max(len(gs) for gs in tok_gs) if tok_gs else 1

            w_b = torch.zeros((1, full_len), dtype=torch.long)
            c_b = torch.zeros((1, full_len, max_c), dtype=torch.long)
            l_b = torch.tensor([LANG_TO_ID["assamese"]], dtype=torch.long)
            m_b = torch.ones((1, full_len), dtype=torch.bool)

            for j, (w, gs) in enumerate(zip(toks, tok_gs)):
                w_b[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
                for k, g in enumerate(gs):
                    c_b[0, j, k] = c_map.get(g, c_map.get("<UNK>", 1))

            w_emb = model.word_embedding(w_b.to(DEVICE))
            c_emb = model.char_cnn(c_b.to(DEVICE))
            l_emb = model.lang_embedding(l_b.to(DEVICE), full_len)
            x_in = torch.cat([w_emb, c_emb, l_emb], dim=-1)
            x_in = model.concat_dropout(x_in)

            lstm_hidden = model.encoder(x_in, mask=m_b.to(DEVICE), return_hidden=True)
            attended_states, attn_weights_blocks = model.attention(lstm_hidden, mask=m_b.to(DEVICE))
            attn_weights = attn_weights_blocks[0] if isinstance(attn_weights_blocks, list) else attn_weights_blocks

            if model.crf_config == "shared":
                emissions = model.fc_shared(attended_states)
            else:
                emissions = model.fc_heads[l_b[0].item()](attended_states)

            p = attn_weights[0]  # [H, L, L]
            token_entropy_per_head = -(p * torch.log(p + 1e-12)).sum(dim=-1)
            token_mean_entropy = token_entropy_per_head.mean(dim=0).cpu().numpy()
            max_possible_ent = math.log(full_len)
            ent_ratio = token_mean_entropy / max_possible_ent

            o_logits = emissions[0, :, o_tag_id].cpu().numpy()
            ent_logits = emissions[0, :, entity_tag_ids]
            max_ent_logits, _ = ent_logits.max(dim=-1)
            max_ent_logits = max_ent_logits.cpu().numpy()
            o_bias_gaps = o_logits - max_ent_logits

            token_trace = []
            for j in range(full_len):
                token_trace.append({
                    "pos": j,
                    "token": toks[j],
                    "gold": gold[j],
                    "is_ent": gold[j] != "O",
                    "o_bias_gap": float(round(o_bias_gaps[j], 3)),
                    "entropy_ratio": float(round(ent_ratio[j], 4)),
                })

            # 2. Sub-span / Prefix Truncation Sweep
            prefix_sweep = []
            valid_prefixes = [k for k in prefix_lengths if k <= full_len]
            if full_len not in valid_prefixes:
                valid_prefixes.append(full_len)

            for pref_k in valid_prefixes:
                sub_toks = toks[:pref_k]
                sub_gold = gold[:pref_k]
                sub_gs = [tokenizer.tokenize_characters(w) for w in sub_toks]
                sub_max_c = max(len(gs) for gs in sub_gs) if sub_gs else 1

                sub_wb = torch.zeros((1, pref_k), dtype=torch.long)
                sub_cb = torch.zeros((1, pref_k, sub_max_c), dtype=torch.long)
                sub_mb = torch.ones((1, pref_k), dtype=torch.bool)

                for j, (w, gs) in enumerate(zip(sub_toks, sub_gs)):
                    sub_wb[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
                    for k_idx, g in enumerate(gs):
                        sub_cb[0, j, k_idx] = c_map.get(g, c_map.get("<UNK>", 1))

                sw_emb = model.word_embedding(sub_wb.to(DEVICE))
                sc_emb = model.char_cnn(sub_cb.to(DEVICE))
                sl_emb = model.lang_embedding(l_b.to(DEVICE), pref_k)
                sx_in = torch.cat([sw_emb, sc_emb, sl_emb], dim=-1)
                sx_in = model.concat_dropout(sx_in)

                slstm_hidden = model.encoder(sx_in, mask=sub_mb.to(DEVICE), return_hidden=True)
                sattended, sattn_blocks = model.attention(slstm_hidden, mask=sub_mb.to(DEVICE))
                s_attn = sattn_blocks[0] if isinstance(sattn_blocks, list) else sattn_blocks

                if model.crf_config == "shared":
                    semissions = model.fc_shared(sattended)
                else:
                    semissions = model.fc_heads[l_b[0].item()](sattended)

                so_logits = semissions[0, :, o_tag_id].cpu().numpy()
                sent_logits = semissions[0, :, entity_tag_ids]
                smax_ent, _ = sent_logits.max(dim=-1)
                smax_ent = smax_ent.cpu().numpy()
                so_gap = so_logits - smax_ent

                if pref_k > 1:
                    sp = s_attn[0]
                    sent_per_head = -(sp * torch.log(sp + 1e-12)).sum(dim=-1)
                    smean_ent = sent_per_head.mean(dim=0).cpu().numpy()
                    s_max_ent = math.log(pref_k)
                    s_ent_ratio = smean_ent / s_max_ent
                    mean_attn_ent_ratio = float(np.mean(s_ent_ratio))
                    token0_ent_ratio = float(s_ent_ratio[0])
                else:
                    mean_attn_ent_ratio = 0.0
                    token0_ent_ratio = 0.0

                prefix_sweep.append({
                    "prefix_length": pref_k,
                    "token0": sub_toks[0],
                    "token0_gold": sub_gold[0],
                    "token0_o_bias_gap": float(round(so_gap[0], 3)),
                    "token0_ent_ratio": float(round(token0_ent_ratio, 4)),
                    "mean_sentence_o_gap": float(round(float(np.mean(so_gap)), 3)),
                    "mean_attn_entropy_ratio": float(round(mean_attn_ent_ratio, 4)),
                })

            results.append({
                "sent_idx": s_idx,
                "full_length": full_len,
                "first_token": toks[0],
                "first_token_gold": gold[0],
                "gold_entities": gold_ents,
                "token_trace": token_trace,
                "prefix_sweep": prefix_sweep,
            })

    return results


def main():
    print("=" * 80)
    print("RELIABILITY CYCLE 2, STEP 1b: CAUSAL DISENTANGLEMENT DIAGNOSTIC")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Full-Corpus Label Consistency Audit
    train_conll = DATA_DIR / "gold" / "assamese" / "train.conll"
    consistency_records, train_stats, token_tag_counts = run_full_corpus_consistency_audit(train_conll, min_freq=5)
    print(f"\n[Task 1: Full-Corpus Consistency Audit Complete]")
    print(f"  Total train tokens: {train_stats['total_tokens_in_train']}")
    print(f"  Total distinct types: {train_stats['total_distinct_types']}")
    print(f"  Types with >= 5 occurrences: {train_stats['qualifying_types_ge_5']}")
    print(f"  Distribution: Min={train_stats['min']:.3f}, P10={train_stats['p10']:.3f}, P25={train_stats['p25']:.3f}, Median={train_stats['median']:.3f}, P75={train_stats['p75']:.3f}, Max={train_stats['max']:.3f}")
    print(f"  Perfect (1.00): {train_stats['count_perfect_1_00']} ({train_stats['count_perfect_1_00']/train_stats['qualifying_types_ge_5']*100:.1f}%)")
    print(f"  Inconsistent (< 1.00): {train_stats['count_inconsistent_lt_1_00']} ({train_stats['count_inconsistent_lt_1_00']/train_stats['qualifying_types_ge_5']*100:.1f}%)")
    print(f"  Ambiguous (< 0.85):    {train_stats['count_lt_0_85']} ({train_stats['count_lt_0_85']/train_stats['qualifying_types_ge_5']*100:.1f}%)")
    print(f"  Severe Ambiguity (< 0.80): {train_stats['count_lt_0_80']} ({train_stats['count_lt_0_80']/train_stats['qualifying_types_ge_5']*100:.1f}%)")
    print(f"  Extreme Ambiguity (< 0.70): {train_stats['count_lt_0_70']} ({train_stats['count_lt_0_70']/train_stats['qualifying_types_ge_5']*100:.1f}%)")

    # Build consistency lookup map
    consistency_map = {r["token"]: r for r in consistency_records}

    # 2. Model Loading & Manifest Inference
    w_map, c_map = load_or_create_multilingual_vocabs(DATA_DIR / "processed")
    as_model = load_model(CHECKPOINT_PATH, len(w_map), len(c_map))
    manifest = load_diagnostic_manifest_sents("assamese")
    print(f"\n[Assamese Manifest Loaded: {len(manifest)} sentences]")

    manifest_tokens, manifest_sents = run_emission_and_attention_manifest_extraction(as_model, manifest, w_map, c_map)
    print(f"Extracted {len(manifest_tokens)} tokens across {len(manifest_sents)} sentences.")

    # 3. Task 2: Emission-Bias-vs-Consistency Correlation (4-Quadrant Analysis)
    # Define thresholds:
    # High-bias: O-bias gap >= +5.0 (Step 1 criterion)
    # Low-consistency: consistency_rate < 0.85 (derived empirically: separates the top 11.8% ambiguous types)
    # Also evaluate alternative cutoffs < 0.80 and < 1.00 for thoroughness
    CUTOFF_BIAS = 5.0
    CUTOFF_CONSISTENCY = 0.85

    # Filter to tokens that have consistency data (seen in train with >= 5 occurrences)
    tokens_with_consistency = []
    gold_entity_tokens_with_consistency = []

    for t in manifest_tokens:
        tok_str = t["token"]
        if tok_str in consistency_map:
            c_info = consistency_map[tok_str]
            merged = {**t, "consistency_rate": c_info["consistency_rate"], "train_total": c_info["total_count"], "train_o_rate": c_info["o_rate"], "top_tag": c_info["top_tag"]}
            tokens_with_consistency.append(merged)
            if t["is_gold_entity"]:
                gold_entity_tokens_with_consistency.append(merged)

    print(f"\nManifest tokens mapped to train consistency (>=5 occurrences): {len(tokens_with_consistency)} (of which {len(gold_entity_tokens_with_consistency)} are gold entities)")

    def compute_quadrants(token_list, bias_th=5.0, cons_th=0.85):
        q1_high_bias_low_cons = []
        q2_high_bias_high_cons = []
        q3_low_bias_low_cons = []
        q4_low_bias_high_cons = []

        for item in token_list:
            is_high_bias = item["o_bias_gap"] >= bias_th
            is_low_cons = item["consistency_rate"] < cons_th
            if is_high_bias and is_low_cons:
                q1_high_bias_low_cons.append(item)
            elif is_high_bias and not is_low_cons:
                q2_high_bias_high_cons.append(item)
            elif not is_high_bias and is_low_cons:
                q3_low_bias_low_cons.append(item)
            else:
                q4_low_bias_high_cons.append(item)

        return {
            "Q1_high_bias_low_cons": len(q1_high_bias_low_cons),
            "Q2_high_bias_high_cons": len(q2_high_bias_high_cons),
            "Q3_low_bias_low_cons": len(q3_low_bias_low_cons),
            "Q4_low_bias_high_cons": len(q4_low_bias_high_cons),
            "total": len(token_list),
            "samples_Q1": q1_high_bias_low_cons[:5],
            "samples_Q2": q2_high_bias_high_cons[:5],
            "samples_Q3": q3_low_bias_low_cons[:5],
            "samples_Q4": q4_low_bias_high_cons[:5],
        }

    # Evaluate for ALL tokens in manifest that have train consistency data
    all_quadrants_085 = compute_quadrants(tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=0.85)
    all_quadrants_080 = compute_quadrants(tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=0.80)
    all_quadrants_100 = compute_quadrants(tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=1.00)

    # Evaluate specifically for GOLD ENTITY tokens in manifest
    entity_quadrants_085 = compute_quadrants(gold_entity_tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=0.85)
    entity_quadrants_080 = compute_quadrants(gold_entity_tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=0.80)
    entity_quadrants_100 = compute_quadrants(gold_entity_tokens_with_consistency, bias_th=CUTOFF_BIAS, cons_th=1.00)

    # Distinct types analysis
    unique_types_map = {}
    for item in tokens_with_consistency:
        tok = item["token"]
        if tok not in unique_types_map:
            unique_types_map[tok] = {
                "token": tok,
                "consistency_rate": item["consistency_rate"],
                "o_bias_gaps": [],
                "is_gold_entity_any": False,
            }
        unique_types_map[tok]["o_bias_gaps"].append(item["o_bias_gap"])
        if item["is_gold_entity"]:
            unique_types_map[tok]["is_gold_entity_any"] = True

    unique_types_list = []
    for tok, d in unique_types_map.items():
        unique_types_list.append({
            "token": tok,
            "consistency_rate": d["consistency_rate"],
            "o_bias_gap": float(np.mean(d["o_bias_gaps"])),
            "is_gold_entity": d["is_gold_entity_any"],
        })
    unique_types_quadrants = compute_quadrants(unique_types_list, bias_th=CUTOFF_BIAS, cons_th=0.85)

    print("\n[Task 2: Correlation Quadrants Results (Cutoff: Gap >= +5.0, Cons < 0.85)]")
    print("  ALL Tokens in Manifest:")
    print(f"    Q1 (High Bias & Low Cons):  {all_quadrants_085['Q1_high_bias_low_cons']:4d} ({all_quadrants_085['Q1_high_bias_low_cons']/len(tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q2 (High Bias & High Cons): {all_quadrants_085['Q2_high_bias_high_cons']:4d} ({all_quadrants_085['Q2_high_bias_high_cons']/len(tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q3 (Low Bias & Low Cons):   {all_quadrants_085['Q3_low_bias_low_cons']:4d} ({all_quadrants_085['Q3_low_bias_low_cons']/len(tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q4 (Low Bias & High Cons):  {all_quadrants_085['Q4_low_bias_high_cons']:4d} ({all_quadrants_085['Q4_low_bias_high_cons']/len(tokens_with_consistency)*100:5.1f}%)")

    print("\n  GOLD ENTITY Tokens in Manifest:")
    print(f"    Q1 (High Bias & Low Cons):  {entity_quadrants_085['Q1_high_bias_low_cons']:4d} ({entity_quadrants_085['Q1_high_bias_low_cons']/len(gold_entity_tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q2 (High Bias & High Cons): {entity_quadrants_085['Q2_high_bias_high_cons']:4d} ({entity_quadrants_085['Q2_high_bias_high_cons']/len(gold_entity_tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q3 (Low Bias & Low Cons):   {entity_quadrants_085['Q3_low_bias_low_cons']:4d} ({entity_quadrants_085['Q3_low_bias_low_cons']/len(gold_entity_tokens_with_consistency)*100:5.1f}%)")
    print(f"    Q4 (Low Bias & High Cons):  {entity_quadrants_085['Q4_low_bias_high_cons']:4d} ({entity_quadrants_085['Q4_low_bias_high_cons']/len(gold_entity_tokens_with_consistency)*100:5.1f}%)")

    # 4. Task 3: Order-of-Onset Test on L3/L4 Collapsed Sentences
    # Select canonical long collapsed sentences from Step 1
    # Sentence 200 (Len 42), Sentence 230 (Len 66), Sentence 220 (Len 53), Sentence 232 (Len 47), Sentence 223 (Len 43)
    stress_sents = load_conll_data(DATA_DIR / "gold" / "assamese" / "long_multientity_eval_v2.conll")
    s200_toks, s200_gold = stress_sents[0]

    collapsed_test_cases = [
        {"sent_idx": 200, "tokens": s200_toks, "gold": s200_gold},
    ]

    # Find additional long collapsed sentences from manifest
    for s_idx, (toks, gold, src) in enumerate(manifest):
        if len(toks) >= 40 and s_idx in [230, 220, 232, 223, 225]:
            collapsed_test_cases.append({"sent_idx": s_idx, "tokens": toks, "gold": gold})

    onset_results = run_order_of_onset_analysis(as_model, w_map, c_map, collapsed_test_cases)
    print(f"\n[Task 3: Order-of-Onset Test Complete on {len(onset_results)} collapsed sentences]")
    for res in onset_results[:3]:
        print(f"\n  Sentence #{res['sent_idx']} (Len={res['full_length']}, Opening Token='{res['first_token']}'):")
        print("    Prefix Length Sweep:")
        for psw in res["prefix_sweep"]:
            print(f"      Len {psw['prefix_length']:2d}: Token0 '{psw['token0']}' O-Bias Gap = {psw['token0_o_bias_gap']:+6.2f} (EntRatio: {psw['token0_ent_ratio']:.3f}), Mean Sent O-Gap = {psw['mean_sentence_o_gap']:+6.2f}, Mean Attn EntRatio = {psw['mean_attn_entropy_ratio']:.3f}")

    # 5. Task 4: Independent-Cause Test
    # Test across all manifest tokens:
    # "identify whether any tokens in the manifest show attention degeneracy WITHOUT a corresponding strong emission O-bias"
    # Degeneracy threshold: entropy ratio >= 0.90 (or >= 0.95)
    # Absence of strong emission O-bias: O-bias gap < +5.0 (or <= 0.0)
    attn_degen_th = 0.90
    attn_degen_th_strict = 0.95
    no_bias_th = 5.0
    entity_dominant_th = 0.0

    degen_without_bias_count = 0
    degen_with_entity_dominant_count = 0
    degen_with_bias_count = 0
    sharp_with_bias_count = 0
    sharp_without_bias_count = 0

    degen_without_bias_examples = []
    sharp_with_bias_examples = []

    for t in manifest_tokens:
        is_degen = t["entropy_ratio"] >= attn_degen_th
        is_high_bias = t["o_bias_gap"] >= no_bias_th

        if is_degen and not is_high_bias:
            degen_without_bias_count += 1
            if len(degen_without_bias_examples) < 10:
                degen_without_bias_examples.append(t)
            if t["o_bias_gap"] <= entity_dominant_th:
                degen_with_entity_dominant_count += 1
        elif is_degen and is_high_bias:
            degen_with_bias_count += 1
        elif not is_degen and is_high_bias:
            sharp_with_bias_count += 1
            if len(sharp_with_bias_examples) < 10:
                sharp_with_bias_examples.append(t)
        else:
            sharp_without_bias_count += 1

    total_manifest_tokens = len(manifest_tokens)
    print("\n[Task 4: Independent-Cause Test]")
    print(f"  Total Manifest Tokens Evaluated: {total_manifest_tokens}")
    print(f"  Attention Degeneracy (Ratio >= 0.90) WITHOUT Strong Emission Bias (Gap < +5.0): {degen_without_bias_count} ({degen_without_bias_count/total_manifest_tokens*100:.2f}%)")
    print(f"  Attention Degeneracy (Ratio >= 0.90) WITH Entity Dominant (Gap <= 0.0):        {degen_with_entity_dominant_count} ({degen_with_entity_dominant_count/total_manifest_tokens*100:.2f}%)")
    print(f"  Attention Degeneracy (Ratio >= 0.90) WITH Strong Emission Bias (Gap >= +5.0):    {degen_with_bias_count} ({degen_with_bias_count/total_manifest_tokens*100:.2f}%)")
    print(f"  Sharp Attention (Ratio < 0.90) WITH Strong Emission Bias (Gap >= +5.0):         {sharp_with_bias_count} ({sharp_with_bias_count/total_manifest_tokens*100:.2f}%)")
    print(f"  Sharp Attention (Ratio < 0.90) WITHOUT Strong Emission Bias (Gap < +5.0):      {sharp_without_bias_count} ({sharp_without_bias_count/total_manifest_tokens*100:.2f}%)")

    # Strict check (Ratio >= 0.95)
    strict_degen_without_bias = sum(1 for t in manifest_tokens if t["entropy_ratio"] >= attn_degen_th_strict and t["o_bias_gap"] < no_bias_th)
    strict_degen_with_bias = sum(1 for t in manifest_tokens if t["entropy_ratio"] >= attn_degen_th_strict and t["o_bias_gap"] >= no_bias_th)
    print(f"\n  Strict Degeneracy (Ratio >= 0.95) WITHOUT Strong Emission Bias: {strict_degen_without_bias} ({strict_degen_without_bias/total_manifest_tokens*100:.2f}%)")
    print(f"  Strict Degeneracy (Ratio >= 0.95) WITH Strong Emission Bias:    {strict_degen_with_bias} ({strict_degen_with_bias/total_manifest_tokens*100:.2f}%)")

    # Check specifically on GOLD ENTITY tokens
    gold_ent_tokens = [t for t in manifest_tokens if t["is_gold_entity"]]
    ent_degen_without_bias = sum(1 for t in gold_ent_tokens if t["entropy_ratio"] >= attn_degen_th and t["o_bias_gap"] < no_bias_th)
    ent_degen_with_bias = sum(1 for t in gold_ent_tokens if t["entropy_ratio"] >= attn_degen_th and t["o_bias_gap"] >= no_bias_th)
    ent_sharp_with_bias = sum(1 for t in gold_ent_tokens if t["entropy_ratio"] < attn_degen_th and t["o_bias_gap"] >= no_bias_th)
    ent_sharp_without_bias = sum(1 for t in gold_ent_tokens if t["entropy_ratio"] < attn_degen_th and t["o_bias_gap"] < no_bias_th)
    print(f"\n  Among Gold Entity Tokens (N={len(gold_ent_tokens)}):")
    print(f"    Degenerate Attention (>=0.90) WITHOUT Strong O-Bias (<+5.0): {ent_degen_without_bias} ({ent_degen_without_bias/len(gold_ent_tokens)*100:.1f}%)")
    print(f"    Degenerate Attention (>=0.90) WITH Strong O-Bias (>=+5.0):    {ent_degen_with_bias} ({ent_degen_with_bias/len(gold_ent_tokens)*100:.1f}%)")
    print(f"    Sharp Attention (<0.90) WITH Strong O-Bias (>=+5.0):         {ent_sharp_with_bias} ({ent_sharp_with_bias/len(gold_ent_tokens)*100:.1f}%)")
    print(f"    Sharp Attention (<0.90) WITHOUT Strong O-Bias (<+5.0):      {ent_sharp_without_bias} ({ent_sharp_without_bias/len(gold_ent_tokens)*100:.1f}%)")

    # Save full diagnostic results to JSON
    output_json = {
        "train_consistency_stats": train_stats,
        "train_consistency_records_all": consistency_records,
        "train_consistency_records_inconsistent": [r for r in consistency_records if r["consistency_rate"] < 1.0],
        "manifest_token_count": total_manifest_tokens,
        "tokens_with_consistency_count": len(tokens_with_consistency),
        "gold_entity_tokens_with_consistency_count": len(gold_entity_tokens_with_consistency),
        "correlation_quadrants": {
            "all_tokens_cutoff_085": all_quadrants_085,
            "all_tokens_cutoff_080": all_quadrants_080,
            "all_tokens_cutoff_100": all_quadrants_100,
            "entity_tokens_cutoff_085": entity_quadrants_085,
            "entity_tokens_cutoff_080": entity_quadrants_080,
            "entity_tokens_cutoff_100": entity_quadrants_100,
            "unique_types_cutoff_085": unique_types_quadrants,
        },
        "order_of_onset_results": onset_results,
        "independent_cause_test": {
            "total_tokens": total_manifest_tokens,
            "attn_degen_th": attn_degen_th,
            "no_bias_th": no_bias_th,
            "degen_without_bias_count": degen_without_bias_count,
            "degen_with_entity_dominant_count": degen_with_entity_dominant_count,
            "degen_with_bias_count": degen_with_bias_count,
            "sharp_with_bias_count": sharp_with_bias_count,
            "sharp_without_bias_count": sharp_without_bias_count,
            "strict_degen_without_bias_count": strict_degen_without_bias,
            "strict_degen_with_bias_count": strict_degen_with_bias,
            "gold_entity_breakdown": {
                "total_entities": len(gold_ent_tokens),
                "degen_without_bias": ent_degen_without_bias,
                "degen_with_bias": ent_degen_with_bias,
                "sharp_with_bias": ent_sharp_with_bias,
                "sharp_without_bias": ent_sharp_without_bias,
            },
            "sample_degen_without_bias": degen_without_bias_examples[:5],
            "sample_sharp_with_bias": sharp_with_bias_examples[:5],
        },
    }

    def json_serialize(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    out_file = SCRATCH_DIR / "cycle2_step1b_disentanglement.json"
    out_file.write_text(json.dumps(output_json, indent=2, ensure_ascii=False, default=json_serialize), encoding="utf-8")
    print(f"\n[Saved Diagnostic Data] -> {out_file}")



if __name__ == "__main__":
    main()
