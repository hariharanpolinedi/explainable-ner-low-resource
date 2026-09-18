"""
Evaluate saved no_attention_ablation_v3_{lang}.pt checkpoints on gold-dev
and produce reports/experiment_logs/no_attention_matched_v3_results.json.
"""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from evaluation.compare_joint_vs_monolingual import evaluate_joint_model_on_language
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import MultilingualNERDataset, load_or_create_multilingual_vocabs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gold_dir = PROJECT_ROOT / "data" / "gold"
proc_dir = PROJECT_ROOT / "data" / "processed"
checkpoints_dir = PROJECT_ROOT / "checkpoints"
languages = ("konkani", "assamese", "sinhala")

word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

results = {}
for lang in languages:
    ckpt_path = checkpoints_dir / f"no_attention_ablation_v3_{lang}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
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
        use_attention=False,
        use_crf=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    dev_ds = MultilingualNERDataset({lang: [gold_dir / lang / "dev.conll"]}, word_to_id, char_to_id)
    m = evaluate_joint_model_on_language(model, dev_ds, device, batch_size=32)
    
    results[lang] = {
        "entity_f1": round(m["entity_f1"] * 100, 2),
        "entity_precision": round(m["entity_precision"] * 100, 2),
        "entity_recall": round(m["entity_recall"] * 100, 2),
        "token_accuracy": round(m["token_accuracy"] * 100, 2),
    }
    print(f"[{lang.capitalize()}] Dev F1: {results[lang]['entity_f1']}% | Precision: {results[lang]['entity_precision']}% | Recall: {results[lang]['entity_recall']}% | Token Acc: {results[lang]['token_accuracy']}%")

macro_f1 = round(sum(results[l]["entity_f1"] for l in languages) / 3, 2)
results["macro"] = {"entity_f1": macro_f1}
print(f"Macro Dev F1: {macro_f1}%")

out_file = PROJECT_ROOT / "reports" / "experiment_logs" / "no_attention_matched_v3_results.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Wrote results to {out_file}")
