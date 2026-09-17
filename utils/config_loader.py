"""
Strict, typed configuration loader for YAML configuration files.

Enforces dataclass-based typing and fails loudly on unknown, missing,
or malformed configuration keys. Supports CLI overrides in 'key=value' format.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    languages: list[str]


@dataclass(frozen=True)
class LoggingConfig:
    backend: str
    log_dir: str


@dataclass(frozen=True)
class LanguageSourceConfig:
    wiki_code: str
    script_name: str
    script_range: list[list[int]]
    news_feeds: list[str]
    other_sources: list[str]


@dataclass(frozen=True)
class DataCollectionConfig:
    min_sentence_length: int
    script_purity_threshold: float
    dedup_method: str
    max_articles_per_wiki: int
    sources: dict[str, LanguageSourceConfig]


@dataclass(frozen=True)
class PreprocessingConfig:
    unicode_normalization: str
    min_word_freq: int
    max_vocab_size: int
    char_min_freq: int
    abbreviations: dict[str, list[str]]


@dataclass(frozen=True)
class SplitRatiosConfig:
    train: float
    dev: float
    test: float


@dataclass(frozen=True)
class AnnotationConfig:
    target_sentence_count: int
    split_ratios: SplitRatiosConfig
    kappa_threshold: float
    overlap_sample_size: int
    tagset: list[str]


@dataclass(frozen=True)
class WeakSupervisionConfig:
    gazetteer_match_mode: str
    aggregation_method: str
    min_precision_threshold: float
    target_volume_multiplier: int
    stoplist: dict[str, list[str]]


@dataclass(frozen=True)
class ModelConfig:
    word_embedding_dim: int
    bilstm_hidden_dim: int
    bilstm_layers: int
    dropout: float
    use_crf: bool


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    use_silver_data: bool
    device: str
    use_class_weights: bool


@dataclass(frozen=True)
class CharCNNConfig:
    char_embedding_dim: int
    filter_widths: list[int]
    filters_per_width: int
    char_dropout: float
    concat_dropout: float


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int
    num_attention_blocks: int
    attention_dropout: float
    warmup_steps: int


@dataclass(frozen=True)
class MultilingualConfig:
    lang_embedding_dim: int
    crf_config: str
    sampling_strategy: str
    num_languages: int


@dataclass(frozen=True)
class FinetuneConfig:
    learning_rate: float
    unfreeze_epoch: int
    schedule: str


@dataclass(frozen=True)
class SelfTrainingConfig:
    initial_threshold: float
    threshold_step: float
    max_rounds: int
    volume_cap_multiplier: float
    learning_rate: float
    batch_size: int = 32
    max_epochs_per_round: int = 15
    patience: int = 4


@dataclass(frozen=True)
class AugmentationConfig:
    multiplier: int
    check_agreement: bool
    loss_variant: str
    focal_gamma: float
    learning_rate: float
    batch_size: int = 32
    max_epochs: int = 10
    patience: int = 3


@dataclass(frozen=True)
class SearchConfig:
    budget: int
    hidden_dims: list[int]
    num_heads: list[int]
    num_attention_blocks: list[int]
    dropouts: list[float]
    learning_rates: list[float]
    lang_embedding_dims: list[int]
    batch_sizes: list[int]
    target_stage: str = "multilingual_joint"


@dataclass(frozen=True)
class AblationConfig:
    components: list[str]


@dataclass(frozen=True)
class BaseConfig:
    project: ProjectConfig
    seed: int
    logging: LoggingConfig
    data_collection: DataCollectionConfig | None = None
    preprocessing: PreprocessingConfig | None = None
    annotation: AnnotationConfig | None = None
    weak_supervision: WeakSupervisionConfig | None = None
    model: ModelConfig | None = None
    char_cnn: CharCNNConfig | None = None
    attention: AttentionConfig | None = None
    multilingual: MultilingualConfig | None = None
    finetune: FinetuneConfig | None = None
    self_training: SelfTrainingConfig | None = None
    augmentation: AugmentationConfig | None = None
    training: TrainingConfig | None = None
    search: SearchConfig | None = None
    ablation: AblationConfig | None = None





def _apply_override(data: dict[str, Any], key_path: str, raw_value: str) -> None:
    """Apply a single key_path=raw_value override into a nested dict."""
    keys = key_path.strip().split(".")
    target = data
    for k in keys[:-1]:
        if not isinstance(target, dict):
            raise ValueError(f"Cannot traverse non-dict element when applying override: '{key_path}'")
        if k not in target:
            target[k] = {}
        target = target[k]

    # Parse value with yaml to preserve int, float, bool, list etc.
    try:
        parsed_value = yaml.safe_load(raw_value)
    except Exception:
        parsed_value = raw_value

    target[keys[-1]] = parsed_value


def apply_overrides(data: dict[str, Any], overrides: list[str] | dict[str, Any] | None) -> dict[str, Any]:
    """Apply overrides to configuration dictionary."""
    if not overrides:
        return data

    updated = dict(data)
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            _apply_override(updated, k, str(v) if not isinstance(v, (int, float, bool, list, dict)) else yaml.dump(v).strip())
    elif isinstance(overrides, list):
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override format '{item}'. Expected 'key=value' or 'section.key=value'.")
            k, v = item.split("=", 1)
            _apply_override(updated, k.strip(), v.strip())
    else:
        raise TypeError(f"Overrides must be a list of 'key=value' strings or a dict, got {type(overrides).__name__}")

    return updated


def _validate_and_instantiate(cls: type[T], data: Any, path: str = "") -> T:
    """Strictly convert dict into a dataclass, rejecting any unknown keys."""
    if not is_dataclass(cls):
        return data

    if not isinstance(data, dict):
        location = f" at '{path}'" if path else ""
        raise ValueError(f"Expected mapping/dict for dataclass {cls.__name__}{location}, got {type(data).__name__}")

    cls_fields = {f.name: f for f in fields(cls)}
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(cls)}

    provided_keys = set(data.keys())
    allowed_keys = set(cls_fields.keys())

    unknown_keys = provided_keys - allowed_keys
    if unknown_keys:
        location = f" in '{path}'" if path else " at root level"
        unknown_str = ", ".join(sorted(f"'{k}'" for k in unknown_keys))
        allowed_str = ", ".join(sorted(f"'{k}'" for k in allowed_keys))
        raise ValueError(
            f"Unknown configuration key(s) {unknown_str}{location}. Allowed keys are: {allowed_str}"
        )

    init_kwargs: dict[str, Any] = {}
    for name, fld in cls_fields.items():
        subpath = f"{path}.{name}" if path else name
        if name not in data:
            from dataclasses import MISSING
            if fld.default is not MISSING:
                init_kwargs[name] = fld.default
                continue
            elif fld.default_factory is not MISSING:
                init_kwargs[name] = fld.default_factory()
                continue
            else:
                raise ValueError(f"Missing required configuration key '{subpath}' for {cls.__name__}")

        raw_val = data[name]
        fld_type = type_hints.get(name, fld.type)

        # Unwrap Union / Optional
        origin = get_origin(fld_type)
        if origin is not None:
            import types
            args = get_args(fld_type)
            if (origin is types.UnionType or str(origin) == "typing.Union") and type(None) in args:
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    fld_type = non_none_args[0]
                    origin = get_origin(fld_type)

        if is_dataclass(fld_type):
            init_kwargs[name] = _validate_and_instantiate(fld_type, raw_val, subpath)
        else:
            if origin is list and isinstance(raw_val, list):
                item_types = get_args(fld_type)
                if item_types and is_dataclass(item_types[0]):
                    init_kwargs[name] = [
                        _validate_and_instantiate(item_types[0], item, f"{subpath}[{i}]")
                        for i, item in enumerate(raw_val)
                    ]
                else:
                    init_kwargs[name] = raw_val
            elif origin is dict and isinstance(raw_val, dict):
                item_types = get_args(fld_type)
                if len(item_types) == 2 and is_dataclass(item_types[1]):
                    val_cls = item_types[1]
                    init_kwargs[name] = {
                        k: _validate_and_instantiate(val_cls, v, f"{subpath}.{k}")
                        for k, v in raw_val.items()
                    }
                else:
                    init_kwargs[name] = raw_val
            else:
                init_kwargs[name] = raw_val

    return cls(**init_kwargs)


def load_config(
    config_path: str | Path = "configs/base_config.yaml",
    overrides: list[str] | dict[str, Any] | None = None,
    schema_cls: type[T] = BaseConfig,
) -> T:
    """
    Load and parse a YAML configuration file into a typed dataclass schema.

    Fails loudly with ValueError if unknown or missing keys are encountered.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if content is None:
        content = {}
    elif not isinstance(content, dict):
        raise ValueError(f"Configuration root in {config_path} must be a YAML mapping, got {type(content).__name__}")

    if overrides:
        content = apply_overrides(content, overrides)

    return _validate_and_instantiate(schema_cls, content)


def parse_cli_args(args: list[str] | None = None) -> tuple[str, list[str]]:
    """Parse CLI arguments for config path and overrides."""
    parser = argparse.ArgumentParser(
        description="Load configuration with strict schema and optional CLI overrides."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Config override in format 'key=value' or 'section.key=value' (can be specified multiple times)",
    )
    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise ValueError(f"Unrecognized CLI arguments: {unknown}")
    return parsed.config, parsed.override


if __name__ == "__main__":
    cfg_file, cli_overrides = parse_cli_args(sys.argv[1:])
    try:
        config = load_config(cfg_file, overrides=cli_overrides)
        print(f"Successfully loaded configuration from {cfg_file} (Project: {config.project.name}, Seed: {config.seed})")
    except Exception as exc:
        print(f"Configuration Error: {exc}", file=sys.stderr)
        sys.exit(1)
