# No-Pretrained-Model Compliance Checklist & Running Ledger

## Project-Wide Absolute Constraint
**NO pretrained models, pretrained embeddings, pretrained tokenizers/vocabularies, or externally trained checkpoints may ever be introduced into this project.** Everything must be built and trained from scratch. Target performance is >90% entity-level F1 across all target low-resource languages (Konkani, Assamese, Sinhala).

Every project phase must append a dated compliance entry to this ledger verifying zero inclusion of external pretrained artifacts.

---

## Phase 0: Reproducible Engineering Foundation & Environment Setup
- **Date**: 2026-09-15
- **Status**: COMPLIANT
- **Components Introduced**:
  - Top-level directory skeleton and data directory hierarchy for Konkani, Assamese, and Sinhala.
  - Pinned Python virtual environment with minimal dependencies (`PyYAML==6.0.3`).
  - Base configuration specification (`configs/base_config.yaml`).
  - Typed, dataclass-enforced YAML configuration loader with CLI override support (`utils/config_loader.py`).
  - Deterministic global seeding utility with guarded framework imports (`utils/seed.py`).
  - Append-only experiment logging utility with git commit and config hash tracking (`utils/logger.py`).
  - Comprehensive project documentation and setup instructions (`README.md`).
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained weights, tokenizers, word embeddings, or external checkpoints downloaded or stored.
  - Verified dependencies: Only `PyYAML==6.0.3` installed in the dedicated project virtual environment. No ML modeling libraries or models present.
  - Preemptive `.gitignore` configured to reject model checkpoints, binary weight files (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.onnx`, `*.h5`), and large datasets.

---

## Phase 1: Corpus Acquisition & Raw Data Consolidation
- **Date**: 2026-09-15
- **Status**: COMPLIANT
- **Components Introduced**:
  - Multilingual raw text acquisition scripts: `data_collection/download_wikipedia.py`, `data_collection/scrape_news.py`, `data_collection/collect_other.py`.
  - Gazetteers builder: `data_collection/build_gazetteers.py` creating person, place, and organization registries for Konkani, Assamese, and Sinhala.
  - Quality filtering and deduplication pipeline: `data_collection/dedup_and_filter.py` implementing exact SHA-256 sentence deduplication, whitespace length filtering, and Unicode script-purity filtering (>= 90%).
  - Corpus statistics generator: `data_collection/corpus_stats.py` calculating sentences, tokens, and unique surface vocabulary counts.
  - Comprehensive provenance audit ledger: `reports/corpus_provenance.md`.
  - Detailed corpus statistics report: `reports/corpus_stats.md`.
  - Verification test suite: `utils/test_phase1.py` with results in `reports/phase1_test_results.md`.
  - Extended configuration schema: `data_collection:` section in `configs/base_config.yaml`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Raw text was gathered strictly via direct Wikimedia XML/columnar streams, public news RSS feeds, and public domain archives.
  - Text filtering was performed strictly using deterministic character-level Unicode block checks and hash algorithms—zero pretrained tokenizers, neural models, embeddings, or external NLP checkpoints were used.
  - All gazetteer entities were compiled from open encyclopedic infoboxes and administrative geographic lists without invoking pretrained NER or geocoding tools.
  - Pinned dependency `pyarrow==21.0.0` added solely for reading open Wikimedia parquet tables (pure data format library; contains no machine learning models or weights).
