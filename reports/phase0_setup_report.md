# Phase 0 Setup & Environment Report

**Date**: 2026-09-15  
**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Author**: ML/NLP & MLOps Engineering Team  
**Phase Status**: COMPLETED / GO  

---

## 1. Initial Workspace Inspection

Prior to any file creation or modification, an exhaustive inspection of the workspace was performed:
- **Git Status**: Workspace was not a git repository (`fatal: not a git repository`). No prior commits, branches, or tracking existed.
- **Directory Contents**: Working directory `C:\Users\harih\Desktop\NLP-MARK-3` was completely empty.
- **Pre-existing Artifacts**: None (`requirements.txt`, `environment.yml`, `configs/`, `utils/`, `README.md`, `NO_PRETRAINED_MODEL_CHECKLIST.md` were all absent).
- **Python Runtime**: Python 3.13.7 (`C:\Users\harih\AppData\Local\Programs\Python\Python313\python.exe`). No virtual environment was active at inception.

---

## 2. Hardware Detection & Acceleration Baseline

Hardware inspection was executed via `nvidia-smi` and runtime checks:

| Attribute | Detected Value | Notes |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | 1 discrete physical GPU available |
| **VRAM Capacity** | 4096 MiB (4 GB) GDDR6 | Power cap: 80W, Idle temp: 58°C |
| **Driver Version** | 595.79 | WDDM 3.2 |
| **CUDA Driver Version** | 13.2 | Compatible with modern PyTorch CUDA wheels |
| **CUDA Execution Check** | `CUDA available: True` | Verified via host runtime probe (`torch.cuda.is_available()`) |
| **Phase 0 Execution Mode** | CPU-only | Intentionally minimal; no ML frameworks installed in Phase 0 |

### Future Hardware Strategy (Phases 5–16)
The presence of the 4GB RTX 3050 GPU is recorded for training phases. Due to the 4GB VRAM ceiling, model architectures developed from scratch in Phase 5 must maintain compact parameter footprints (e.g. lightweight Transformer / BiLSTM-CRF architectures with hidden dimensions ~128–256 and small batch sizes 16–32 with gradient accumulation).

---

## 3. Environment & Dependency Architecture

A clean, isolated virtual environment was initialized at `venv/`:
- **Python Version**: 3.13.7
- **Installed Packages**:
  - `PyYAML==6.0.3`
  - `pip==26.2.1`
- **Zero Pretrained Model Guarantee**:
  `PyYAML` is a pure parser library with no bundled neural weights, tokenizers, or external network dependencies. No deep learning libraries (`torch`, `tensorflow`, `transformers`, `huggingface_hub`) were installed.
- **Requirements File**: Pinned to exact version in `requirements.txt`.

---

## 4. Engineering Utilities Implemented

1. **Folder Skeleton**:
   - 39 directory paths created covering all project lifecycle stages, from `data_collection/` to `deployment/` and `paper/`.
   - Complete multilingual hierarchy initialized for `konkani`, `assamese`, and `sinhala` across all 7 data tiers (`raw`, `gazetteers`, `processed`, `gold`, `silver`, `pseudo`, `augmented`).
   - Every directory contains a `.gitkeep` placeholder tracked in version control.

2. **Configuration Loader (`utils/config_loader.py`)**:
   - Strictly enforces typed schemas via Python `dataclasses` (`ProjectConfig`, `LoggingConfig`, `BaseConfig`).
   - Fails loudly with `ValueError` if any unexpected or unregistered key is encountered.
   - Fully supports nested dot-notation CLI overrides (`--override logging.backend=jsonl`).

3. **Global Seeding Utility (`utils/seed.py`)**:
   - Seeds `random`, `PYTHONHASHSEED`, and guards `numpy`, `torch`, and `tensorflow`.
   - Handles missing optional frameworks gracefully without crashing.
   - Guarantees exact floating-point sequence reproducibility.

4. **Experiment Logger (`utils/logger.py`)**:
   - Append-only architecture supporting both CSV and JSONL formats.
   - Captures UTC timestamp, git commit hash (`git rev-parse HEAD`), configuration file path, configuration SHA256 content hash, dataset version tag, and metrics payload.

5. **Compliance Ledger (`NO_PRETRAINED_MODEL_CHECKLIST.md`)**:
   - Running ledger initialized with Phase 0 audit showing zero pretrained models, weights, or tokenizers.

6. **Documentation (`README.md`)**:
   - Features the verbatim zero-pretrained-model constraint, target languages, target metric (>90% entity-level F1), architecture overview, and verified setup steps.

---

## 5. Next Steps
Phase 0 foundation is complete and validated. Phase 1 (Corpus Acquisition & Raw Data Consolidation) may proceed upon instruction.
