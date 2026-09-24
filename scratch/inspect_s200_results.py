import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "reports" / "cycle2_assamese_strengthening_sweep_data.json"

with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

for model_name, info in data.items():
    s200 = info["s200_results"]
    dm = info["dev_metrics"]
    sm = info["stress_metrics"]
    print("=" * 70)
    print(f"Model: {model_name} | Checkpoint: {info['checkpoint']}")
    print(f"  Dev F1: {dm['entity_f1']:.2f}% (P: {dm['entity_precision']:.2f}%, R: {dm['entity_recall']:.2f}%)")
    print(f"  Stress F1: {sm['entity_f1']:.2f}% | Collapse Rate: {sm['collapse_rate']:.2f}% ({sm['collapse_count']}/{sm['sents_with_entities']})")
    print(f"  S200 Mean Gap: {s200['mean_gap']:+.2f} | Token0 Gap: {s200['token0_gap']:+.2f}")
    print(f"  S200 Pred Entity Count: {s200['pred_entity_tokens']} / {s200['gold_entity_tokens']} (Hits: {s200['hits']})")
    print(f"  S200 Pred Spans: {s200['pred_spans']}")
    print(f"  S200 Gold Spans: {s200['gold_spans']}")
    print("  Token Breakdown (Gold != O or Pred != O):")
    for d in s200["token_details"]:
        if d["gold"] != "O" or d["pred"] != "O":
            print(f"    [{d['idx']:2d}] {d['token']:15s} Gold: {d['gold']:6s} | Pred: {d['pred']:6s} | O-Logit: {d['o_logit']:+6.2f} | MaxEnt: {d['max_ent_logit']:+6.2f} ({d['best_entity_tag']}) | Gap: {d['o_bias_gap']:+6.2f}")
