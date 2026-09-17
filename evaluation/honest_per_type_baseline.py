"""
Honest Per-Entity-Type Baseline Evaluation on Locked Gold-Dev.

Evaluates candidate checkpoints per language exclusively on:
  data/gold/{lang}/dev.conll

Never touches test.conll or long_multientity_eval.conll.
Produces full breakdown of PER, LOC, ORG, MISC (precision, recall, F1),
plus overall entity F1 and token accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG

LANGUAGES = ["konkani", "assamese", "sinhala"]

CHECKPOINTS_TO_EVALUATE: dict[str, list[tuple[str, str, Path]]] = {
    "konkani": [
        ("Phase 9 Fine-Tuned", "finetuned_konkani.pt", PROJECT_ROOT / "checkpoints" / "finetuned_konkani.pt"),
        ("Phase 10 Self-Trained (Baseline)", "selftrained_konkani_best.pt", PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt"),
        ("Phase 10/11 Patch v1 (Rejected)", "patched_konkani.pt", PROJECT_ROOT / "checkpoints" / "patched_konkani.pt"),
        ("Patch v2 (Rejected)", "patched_v2_konkani.pt", PROJECT_ROOT / "checkpoints" / "patched_v2_konkani.pt"),
        ("Currently Deployed (final_konkani)", "final_konkani.pt", PROJECT_ROOT / "checkpoints" / "final_konkani.pt"),
    ],
    "assamese": [
        ("Phase 9 Fine-Tuned", "finetuned_assamese.pt", PROJECT_ROOT / "checkpoints" / "finetuned_assamese.pt"),
        ("Phase 10 Self-Trained", "selftrained_assamese_best.pt", PROJECT_ROOT / "checkpoints" / "selftrained_assamese_best.pt"),
        ("Phase 11 Augmented (Baseline)", "augmented_assamese.pt", PROJECT_ROOT / "checkpoints" / "augmented_assamese.pt"),
        ("Phase 10/11 Patch v1 (Rejected)", "patched_assamese.pt", PROJECT_ROOT / "checkpoints" / "patched_assamese.pt"),
        ("Patch v2 (Candidate)", "patched_v2_assamese.pt", PROJECT_ROOT / "checkpoints" / "patched_v2_assamese.pt"),
        ("Currently Deployed (final_assamese)", "final_assamese.pt", PROJECT_ROOT / "checkpoints" / "final_assamese.pt"),
    ],
    "sinhala": [
        ("Phase 9 Fine-Tuned", "finetuned_sinhala.pt", PROJECT_ROOT / "checkpoints" / "finetuned_sinhala.pt"),
        ("Phase 10 Self-Trained", "selftrained_sinhala_best.pt", PROJECT_ROOT / "checkpoints" / "selftrained_sinhala_best.pt"),
        ("Phase 11 Augmented (Baseline)", "augmented_sinhala.pt", PROJECT_ROOT / "checkpoints" / "augmented_sinhala.pt"),
        ("Phase 10/11 Patch v1 (Rejected)", "patched_sinhala.pt", PROJECT_ROOT / "checkpoints" / "patched_sinhala.pt"),
        ("Patch v2 (Rejected)", "patched_v2_sinhala.pt", PROJECT_ROOT / "checkpoints" / "patched_v2_sinhala.pt"),
        ("Currently Deployed (final_sinhala)", "final_sinhala.pt", PROJECT_ROOT / "checkpoints" / "final_sinhala.pt"),
    ],
}


def load_model(ckpt_path: Path, device: torch.device, w_to_id: dict, c_to_id: dict) -> NERMultilingualModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = NERMultilingualModel(len(w_to_id), len(c_to_id), 9, 3, hidden_dim=128).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def evaluate_conll(
    model: NERMultilingualModel,
    conll_path: Path,
    lang: str,
    device: torch.device,
    w_to_id: dict,
    c_to_id: dict,
) -> dict[str, Any]:
    ds = MultilingualNERDataset({lang: [conll_path]}, w_to_id, c_to_id)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for b in dl:
            paths = model(
                b["word_ids"].to(device),
                b["char_ids"].to(device),
                lang_ids=b["lang_ids"].to(device),
                mask=b["mask"].to(device),
            )
            for i in range(len(b["seq_lengths"])):
                slen = b["seq_lengths"][i]
                all_golds.append([ID_TO_TAG[t.item()] for t in b["tags"][i, :slen]])
                all_preds.append([ID_TO_TAG[t] for t in paths[i][:slen]])

    return compute_seqeval_metrics(all_golds, all_preds)


def run_honest_baseline_evaluation() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Honest Baseline Evaluation on device: {device}")

    proc_dir = PROJECT_ROOT / "data" / "processed"
    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for lang in LANGUAGES:
        print(f"\n==================== EVALUATING {lang.upper()} ON GOLD-DEV ====================")
        dev_file = PROJECT_ROOT / "data" / "gold" / lang / "dev.conll"
        assert dev_file.is_file(), f"Missing dev set: {dev_file}"

        all_results[lang] = []
        for desc, fname, cpath in CHECKPOINTS_TO_EVALUATE[lang]:
            if not cpath.is_file():
                print(f"Skipping {fname} (not found)")
                continue

            model = load_model(cpath, device, w_to_id, c_to_id)
            metrics = evaluate_conll(model, dev_file, lang, device, w_to_id, c_to_id)

            record = {
                "description": desc,
                "filename": fname,
                "path": str(cpath.relative_to(PROJECT_ROOT)),
                "overall_f1": metrics["overall"]["entity_f1"] * 100.0,
                "overall_prec": metrics["overall"]["entity_precision"] * 100.0,
                "overall_rec": metrics["overall"]["entity_recall"] * 100.0,
                "token_acc": metrics["token_accuracy"] * 100.0,
                "by_type": {},
            }

            for etype in ["PER", "LOC", "ORG", "MISC"]:
                tinfo = metrics["by_type"].get(etype, {})
                record["by_type"][etype] = {
                    "precision": tinfo.get("precision", 0.0) * 100.0,
                    "recall": tinfo.get("recall", 0.0) * 100.0,
                    "f1": tinfo.get("f1", 0.0) * 100.0,
                    "tp": tinfo.get("tp", 0),
                    "fp": tinfo.get("fp", 0),
                    "fn": tinfo.get("fn", 0),
                }

            all_results[lang].append(record)
            print(
                f"[{lang.upper()}] {fname:<28} | Overall F1: {record['overall_f1']:.2f}% | "
                f"PER F1: {record['by_type']['PER']['f1']:.2f}% | "
                f"LOC F1: {record['by_type']['LOC']['f1']:.2f}% | "
                f"ORG F1: {record['by_type']['ORG']['f1']:.2f}% | "
                f"MISC F1: {record['by_type']['MISC']['f1']:.2f}%"
            )

    return all_results


def generate_honest_baseline_markdown(results: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Honest Per-Entity-Type Baseline Report",
        "",
        "**Date**: 2026-09-17  ",
        "**Evaluation Set**: Locked, untouched `data/gold/{lang}/dev.conll` (200 sentences per language).  ",
        "**Rule Enforced**: No test set peeking, no contaminated `long_multientity_eval.conll`.  ",
        "",
        "---",
    ]

    for lang in LANGUAGES:
        records = results.get(lang, [])
        lines.extend([
            f"",
            f"## {lang.capitalize()} Master Per-Type Baseline",
            f"",
            f"| Checkpoint | Origin Phase | Overall F1 | Token Acc | PER F1 (P / R) | LOC F1 (P / R) | ORG F1 (P / R) | MISC F1 (P / R) |",
            f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for r in records:
            p = r["by_type"]["PER"]
            l = r["by_type"]["LOC"]
            o = r["by_type"]["ORG"]
            m = r["by_type"]["MISC"]

            per_str = f"**{p['f1']:.2f}%** ({p['precision']:.1f}/{p['recall']:.1f})"
            loc_str = f"**{l['f1']:.2f}%** ({l['precision']:.1f}/{l['recall']:.1f})"
            org_str = f"**{o['f1']:.2f}%** ({o['precision']:.1f}/{o['recall']:.1f})"
            misc_str = f"**{m['f1']:.2f}%** ({m['precision']:.1f}/{m['recall']:.1f})"

            lines.append(
                f"| `{r['filename']}` | {r['description']} | **{r['overall_f1']:.2f}%** | {r['token_acc']:.2f}% | "
                f"{per_str} | {loc_str} | {org_str} | {misc_str} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved Honest Baseline Report to: {out_path}")


if __name__ == "__main__":
    res = run_honest_baseline_evaluation()
    report_file = PROJECT_ROOT / "reports" / "honest_per_type_baseline_table.md"
    generate_honest_baseline_markdown(res, report_file)
