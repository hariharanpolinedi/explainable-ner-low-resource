"""
Append-only experiment logging utility for tracking reproducibility,
configurations, dataset versions, and evaluation metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_git_commit_hash() -> str:
    """
    Retrieve the current git commit hash via 'git rev-parse HEAD'.
    Gracefully handles missing git repository or git executable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "git_commit_unavailable"


def compute_file_hash(filepath: str | Path) -> str:
    """Compute SHA256 content hash of a file."""
    path = Path(filepath)
    if not path.is_file():
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ExperimentLogger:
    """
    Append-only logger writing experiment metadata to CSV or JSON-lines.
    """

    LOG_FIELDS = [
        "timestamp",
        "git_commit_hash",
        "config_path",
        "config_hash",
        "dataset_version",
        "metrics",
    ]

    def __init__(
        self,
        log_dir: str | Path = "reports/experiment_logs",
        backend: str = "csv",
        filename: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend.lower().strip()

        if self.backend not in {"csv", "jsonl"}:
            raise ValueError(f"Unsupported logging backend '{backend}'. Must be 'csv' or 'jsonl'.")

        if filename:
            self.filepath = self.log_dir / filename
        else:
            ext = "csv" if self.backend == "csv" else "jsonl"
            self.filepath = self.log_dir / f"experiments.{ext}"

    def log_run(
        self,
        config_path: str | Path,
        metrics: dict[str, Any],
        dataset_version: str = "v0.0-unversioned",
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record a single experiment run.

        Args:
            config_path: Path to configuration file used for the run.
            metrics: Dictionary of evaluated metrics (e.g. {'entity_f1': 0.91, 'precision': 0.92}).
            dataset_version: Identifier / placeholder for the dataset version.
            extra_metadata: Optional dictionary of additional metadata.

        Returns:
            The recorded log entry dictionary.
        """
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit_hash": get_git_commit_hash(),
            "config_path": str(config_path),
            "config_hash": compute_file_hash(config_path),
            "dataset_version": str(dataset_version),
            "metrics": metrics,
        }

        if extra_metadata:
            entry["extra_metadata"] = extra_metadata

        if self.backend == "csv":
            self._write_csv(entry)
        else:
            self._write_jsonl(entry)

        return entry

    def _write_csv(self, entry: dict[str, Any]) -> None:
        file_exists = self.filepath.is_file() and self.filepath.stat().st_size > 0
        csv_row = {
            "timestamp": entry["timestamp"],
            "git_commit_hash": entry["git_commit_hash"],
            "config_path": entry["config_path"],
            "config_hash": entry["config_hash"],
            "dataset_version": entry["dataset_version"],
            "metrics": json.dumps(entry["metrics"], sort_keys=True),
        }
        if "extra_metadata" in entry:
            csv_row["extra_metadata"] = json.dumps(entry["extra_metadata"], sort_keys=True)

        fieldnames = list(csv_row.keys())

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

    def _write_jsonl(self, entry: dict[str, Any]) -> None:
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_logs(self) -> list[dict[str, Any]]:
        """Read all logged experiment records from the active log file."""
        if not self.filepath.is_file():
            return []

        records: list[dict[str, Any]] = []
        if self.backend == "csv":
            with open(self.filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed_row = dict(row)
                    if "metrics" in parsed_row:
                        try:
                            parsed_row["metrics"] = json.loads(parsed_row["metrics"])
                        except Exception:
                            pass
                    if "extra_metadata" in parsed_row:
                        try:
                            parsed_row["extra_metadata"] = json.loads(parsed_row["extra_metadata"])
                        except Exception:
                            pass
                    records.append(parsed_row)
        else:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records


if __name__ == "__main__":
    logger = ExperimentLogger(log_dir="reports/experiment_logs", backend="csv")
    dummy_metrics = {"entity_f1": 0.923, "precision": 0.931, "recall": 0.915}
    entry = logger.log_run(
        config_path="configs/base_config.yaml",
        metrics=dummy_metrics,
        dataset_version="v0.0-smoke-test",
    )
    print("Logged test run successfully:")
    print(entry)
