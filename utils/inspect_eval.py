import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('reports/20_multilingual_eval_results.json', encoding='utf-8') as f:
    items = json.load(f)

for item in items:
    if item['language'] != 'sinhala':
        continue
    print(f"[{item['id']}] {item['title']}: {item['matched_count']}/{item['expected_count']} ({item['recall']}%)")
    if item['missed']:
        print("  Missed expected:")
        for m in item['missed']:
            print(f"    - {m[0]} ({m[1]})")
    preds = [p['text'] + f" ({p['type']})" for p in item['predicted_entities']]
    print(f"  Model predictions ({len(preds)}):")
    print("    " + ", ".join(preds))
    print()

