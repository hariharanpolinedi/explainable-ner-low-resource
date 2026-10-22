"""
Investigate public sources for Kokborok text discovery:
1. SpeeD-TB (speed-tb/kokborok)
2. sdmy/kokborok
3. agnivamaiti/kokborok-qa
4. KokLai (koklai.in)
5. Additional HF datasets & Wikimedia
"""

import json
import urllib.request
import re
from pathlib import Path

# 1. Search Hugging Face
search_queries = ['kokborok', 'tripuri', 'trp', 'kokborok-qa']
found_datasets = {}

for q in search_queries:
    api_url = f"https://huggingface.co/api/datasets?search={q}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "KokborokCorpusBot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for d in data:
                did = d.get("id", "")
                if did not in found_datasets:
                    found_datasets[did] = d
    except Exception as e:
        print(f"Error searching {q}: {e}")

print("=== HUGGING FACE SEARCH RESULTS ===")
print(f"Found {len(found_datasets)} datasets:")
for did, meta in sorted(found_datasets.items()):
    tags = meta.get("tags", [])[:4]
    downloads = meta.get("downloads", 0)
    print(f"  - {did:<35} | Downloads: {downloads:5d} | Tags: {tags}")

# 2. Inspect agnivamaiti/kokborok-qa
print("\n=== INSPECTING agnivamaiti/kokborok-qa ===")
api_url = "https://huggingface.co/api/datasets/agnivamaiti/kokborok-qa"
try:
    req = urllib.request.Request(api_url, headers={"User-Agent": "KokborokCorpusBot/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
        print("agnivamaiti/kokborok-qa metadata:")
        print("  Tags:", meta.get("tags", []))
        print("  Siblings:", [s.get("rfilename") for s in meta.get("siblings", [])])
except Exception as e:
    print(f"agnivamaiti/kokborok-qa API error: {e}")

# 3. Inspect KokLai (koklai.in)
print("\n=== INSPECTING KokLai (koklai.in) ===")
try:
    req = urllib.request.Request("https://koklai.in", headers={"User-Agent": "KokborokCorpusBot/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        print(f"KokLai status: {resp.status}, length: {len(html)} chars")
        # Check for downloadable files/links
        links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
        data_links = [l for l in links if any(ext in l.lower() for ext in ['.csv', '.json', '.txt', '.zip', '.tar', 'download', 'api'])]
        print(f"  Downloadable data links found: {data_links}")
except Exception as e:
    print(f"KokLai web request: {e}")
