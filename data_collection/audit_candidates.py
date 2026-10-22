"""
Audit candidate provenance file.
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / 'data/processed/kokborok_org_candidates_provenance.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in provenance: {len(rows)}")
for i, r in enumerate(rows):
    ent = r['entity']
    cat = r['entity_type']
    sent = r['sentence']
    print(f"[{i+1:2d}] {ent:30s} | {cat:12s} | {sent[:75]}")
