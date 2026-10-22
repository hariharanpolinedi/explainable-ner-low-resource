"""
Keyword scan script for all public sentences.
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / 'data_collection/unified_public_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

search_words = [
    'police', 'Police', 'Sorkar', 'sorkar', 'TTAADC', 'TSR', 'NIT', 'Gramin Bank',
    'High Court', 'BJP', 'CPIM', 'IPFT', 'Motha', 'FIFA', 'BCCI', 'ICC', 'BBC', 'CNN',
    'Mission', 'Sabha', 'Sangha', 'Samiti', 'Council', 'Assembly', 'Department',
    'department', 'Hukumu', 'Columbia', 'Harvard', 'Oxford', 'TIT', 'TSECL', 'THDC', 'TIDC'
]

for kw in search_words:
    pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
    matches = []
    for item in corpus:
        if pattern.search(item['kok']):
            matches.append((item['doc_id'], item['kok'], item.get('en', '')))
    print(f"\nKeyword '{kw}': {len(matches)} matches")
    for m in matches[:3]:
        print(f"  [{m[0]}] {m[1]}")
        if m[2]:
            print(f"      EN: {m[2]}")
