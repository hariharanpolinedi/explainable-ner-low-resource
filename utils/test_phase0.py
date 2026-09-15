"""
Verification test suite for Phase 0 components:
1. Config-loader test (equality & error raising on unknown key)
2. Seeding-determinism test (Python random sequence reproducibility)
3. Logger smoke test (log dummy run, read back, assert schema & values)
"""

import os
import random
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed


def test_config_loader() -> tuple[bool, str]:
    """Test config loading equality and strict error raising on unknown keys."""
    try:
        cfg1 = load_config("configs/base_config.yaml")
        cfg2 = load_config("configs/base_config.yaml")
        assert cfg1 == cfg2, "Repeated load of base_config.yaml did not yield equal dataclasses"
        assert isinstance(cfg1, BaseConfig), "Loaded config is not an instance of BaseConfig"

        # Test CLI override functionality
        cfg_override = load_config("configs/base_config.yaml", overrides=["seed=999"])
        assert cfg_override.seed == 999, f"Override failed: expected 999, got {cfg_override.seed}"

        # Test unknown key rejection
        raised = False
        try:
            load_config("configs/base_config.yaml", overrides=["invalid_key=foo"])
        except ValueError:
            raised = True
        assert raised, "Config loader failed to raise ValueError on unknown top-level key"

        raised_nested = False
        try:
            load_config("configs/base_config.yaml", overrides=["logging.invalid_field=bar"])
        except ValueError:
            raised_nested = True
        assert raised_nested, "Config loader failed to raise ValueError on unknown nested key"

        return True, "Passed: Equality confirmed, overrides functional, unknown keys correctly raise ValueError."
    except Exception as exc:
        return False, f"Failed: {exc}"


def test_seeding_determinism() -> tuple[bool, str]:
    """Test deterministic sequence generation across global seed resets."""
    try:
        seed = 42
        set_global_seed(seed)
        seq1 = [random.random() for _ in range(20)]

        # Mutate state
        _ = [random.random() for _ in range(50)]

        # Reset seed and re-generate
        set_global_seed(seed)
        seq2 = [random.random() for _ in range(20)]

        assert seq1 == seq2, f"Sequences diverge: {seq1[:3]} != {seq2[:3]}"
        return True, f"Passed: Sequences identical across seed reset ({len(seq1)} floats verified)."
    except Exception as exc:
        return False, f"Failed: {exc}"


def test_logger_smoke() -> tuple[bool, str]:
    """Test experiment logging to disk and reading back verified schema."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test CSV Logger
            logger_csv = ExperimentLogger(log_dir=tmpdir, backend="csv", filename="test_log.csv")
            fake_metrics = {"entity_f1": 0.935, "precision": 0.941, "recall": 0.929}
            logged_entry = logger_csv.log_run(
                config_path="configs/base_config.yaml",
                metrics=fake_metrics,
                dataset_version="v0.0-synthetic-test",
            )

            records = logger_csv.read_logs()
            assert len(records) == 1, f"Expected 1 record, found {len(records)}"
            record = records[0]

            for expected_field in [
                "timestamp",
                "git_commit_hash",
                "config_path",
                "config_hash",
                "dataset_version",
                "metrics",
            ]:
                assert expected_field in record, f"Missing field '{expected_field}' in logged record"

            assert record["metrics"]["entity_f1"] == 0.935, "Metric value mismatch in parsed log"
            assert record["dataset_version"] == "v0.0-synthetic-test"

            # 2. Test JSONL Logger
            logger_jsonl = ExperimentLogger(log_dir=tmpdir, backend="jsonl", filename="test_log.jsonl")
            logger_jsonl.log_run(
                config_path="configs/base_config.yaml",
                metrics=fake_metrics,
                dataset_version="v0.0-synthetic-test",
            )
            records_jsonl = logger_jsonl.read_logs()
            assert len(records_jsonl) == 1, f"Expected 1 record in JSONL, found {len(records_jsonl)}"
            assert records_jsonl[0]["metrics"]["entity_f1"] == 0.935

        return True, "Passed: CSV and JSONL loggers accurately wrote and deserialized all expected schema fields."
    except Exception as exc:
        return False, f"Failed: {exc}"


def main() -> int:
    tests = [
        ("Config-loader test", test_config_loader),
        ("Seeding-determinism test", test_seeding_determinism),
        ("Logger smoke test", test_logger_smoke),
    ]

    all_passed = True
    print("=" * 60)
    print("PHASE 0 TEST SUITE RUN")
    print("=" * 60)

    for name, test_fn in tests:
        passed, msg = test_fn()
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name}: {msg}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("OVERALL RESULT: ALL TESTS PASSED (GO)")
        return 0
    else:
        print("OVERALL RESULT: SOME TESTS FAILED (NO-GO)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
