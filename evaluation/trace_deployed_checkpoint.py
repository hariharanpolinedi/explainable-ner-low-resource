"""
Checkpoint Provenance Trace:
Identifies exact file path, SHA-256 hash, git history, and phase of origin
for the currently deployed model in deployment/bundles/{lang}/v1/model.pt.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_history_for_file(rel_path: str) -> list[dict[str, str]]:
    """Retrieve git log entries for a given file."""
    try:
        res = subprocess.run(
            ["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso", "--", rel_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commits = []
        for line in res.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("|", 3)
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "subject": parts[3] if len(parts) > 3 else "",
            })
        return commits
    except Exception as e:
        return [{"error": str(e)}]


def trace_all_deployed_checkpoints() -> dict[str, Any]:
    languages = ["konkani", "assamese", "sinhala"]
    trace_results: dict[str, Any] = {}

    # Checkpoint hashes in checkpoints/
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    all_ckpt_hashes: dict[str, list[str]] = {}
    for p in ckpt_dir.glob("*.pt"):
        chash = compute_sha256(p)
        all_ckpt_hashes.setdefault(chash, []).append(p.name)

    for lang in languages:
        bundle_meta_path = PROJECT_ROOT / "deployment" / "bundles" / lang / "v1" / "bundle_metadata.json"
        deployed_model_path = PROJECT_ROOT / "deployment" / "bundles" / lang / "v1" / "model.pt"

        if not bundle_meta_path.is_file() or not deployed_model_path.is_file():
            trace_results[lang] = {"error": f"Missing bundle for {lang}"}
            continue

        with open(bundle_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        deployed_hash = compute_sha256(deployed_model_path)
        matching_ckpts = all_ckpt_hashes.get(deployed_hash, [])

        # Trace git commit of bundle metadata (since *.pt is gitignored)
        bundle_git = get_git_history_for_file(f"deployment/bundles/{lang}/v1/bundle_metadata.json")

        # Map to phase
        origin_phase = "UNKNOWN"
        is_rejected_patch = False
        rejected_patch_match = None

        if "selftrained_konkani_best.pt" in matching_ckpts:
            origin_phase = "Phase 10 (Self-Training Baseline)"
        elif "augmented_assamese.pt" in matching_ckpts:
            origin_phase = "Phase 11 (Data Augmentation Baseline)"
        elif "augmented_sinhala.pt" in matching_ckpts:
            origin_phase = "Phase 11 (Data Augmentation Baseline)"
        elif any("patched_v2" in m for m in matching_ckpts):
            origin_phase = "Patch v2 (Rejected)"
            is_rejected_patch = True
            rejected_patch_match = "Patch v2"
        elif any("patched_" in m for m in matching_ckpts):
            origin_phase = "Phase 10/11 Patch v1 (Rejected)"
            is_rejected_patch = True
            rejected_patch_match = "Patch v1"

        trace_results[lang] = {
            "deployed_file": str(deployed_model_path.relative_to(PROJECT_ROOT)),
            "deployed_sha256": deployed_hash,
            "bundle_metadata_source": meta["artifacts"]["model"]["original_source"],
            "matching_checkpoints_in_checkpoints_dir": matching_ckpts,
            "origin_phase": origin_phase,
            "is_rejected_patch": is_rejected_patch,
            "rejected_patch_version": rejected_patch_match,
            "bundle_creation_commit": bundle_git[0] if bundle_git else None,
        }

    return trace_results


if __name__ == "__main__":
    results = trace_all_deployed_checkpoints()
    print("=== DEPLOYED CHECKPOINT PROVENANCE TRACE ===")
    for lang, res in results.items():
        print(f"\n--- {lang.upper()} ---")
        if "error" in res:
            print(f"Error: {res['error']}")
            continue
        print(f"Deployed Bundle: {res['deployed_file']}")
        print(f"SHA256: {res['deployed_sha256']}")
        print(f"Metadata Source: {res['bundle_metadata_source']}")
        print(f"Identical Checkpoints in checkpoints/: {res['matching_checkpoints_in_checkpoints_dir']}")
        print(f"Phase of Origin: {res['origin_phase']}")
        print(f"Is Rejected Patch?: {res['is_rejected_patch']}")
        if res['bundle_creation_commit']:
            print(f"Bundle Commit: {res['bundle_creation_commit']['hash'][:8]} - {res['bundle_creation_commit']['subject']} ({res['bundle_creation_commit']['date']})")
