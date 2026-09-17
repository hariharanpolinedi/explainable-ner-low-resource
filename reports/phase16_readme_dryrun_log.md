# Phase 16: README Reproduction Dry-Run Verification Log

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Research Engineer  
**Execution Environment**: Windows 11 (PowerShell), Python 3.13.2  
**Hardware Detected**: `cuda (NVIDIA GeForce RTX 3050 Laptop GPU)`  
**Status**: **PASS (100% Verified)**

---

## 1. Setup & Environment Dry-Run Execution Log

The following reproduction commands from [README.md](README.md) were executed step-by-step in a literal dry-run to verify that a third party can reproduce the environment and initial stages cleanly:

### Step 1: Configuration Loader Verification
```powershell
python -m utils.config_loader --config configs/base_config.yaml
```
- **Command Exit Code**: `0`
- **Verification Details**:
  - Successfully loaded `configs/base_config.yaml`.
  - Verified default random seed: `42`.
  - Verified languages: `['konkani', 'assamese', 'sinhala']`.
  - CLI overrides functional; unknown keys strictly raise `ValueError`.

### Step 2: Global Seeding Determinism Verification
```powershell
python utils/seed.py
```
- **Command Exit Code**: `0`
- **Verification Details**:
  - Deterministic sequence generation verified across seed resets (`set_global_seed(42)`).
  - 20 pseudo-random floats matched identically across resets.
  - Python `random`, `numpy`, and `torch` (including `torch.cuda.manual_seed_all`) deterministically locked.

### Step 3: Experiment Logging Verification
```powershell
python utils/logger.py
```
- **Command Exit Code**: `0`
- **Verification Details**:
  - Experiment logger wrote test run record to `reports/experiment_logs/experiments.csv`.
  - JSONL logger serialized all required metadata fields (run ID, timestamp, config hash, git commit).
  - Deserialization matched in-memory records with zero schema drift.

### Step 4: Preprocessing & Script Tokenizer Dry-Run
```powershell
python -c "from preprocessing.normalize import normalize_text; from preprocessing.tokenizer import ScriptTokenizer; tok = ScriptTokenizer('konkani'); print(tok.tokenize_words(normalize_text('गोंय राज्याची राजधानी पणजी आसा .', form='NFC', lang='konkani')))"
```
- **Command Exit Code**: `0`
- **Verification Details**:
  - Normalized NFC text cleanly.
  - Correctly split extended grapheme clusters (`\X`) while preserving aksharas and vowel matras.
  - Emitted valid tokens: `['गोंय', 'राज्याची', 'राजधानी', 'पणजी', 'आसा', '.']`.

---

## 2. Automated Pipeline Test Suites Run Summary

To confirm that the reproduction commands in [README.md](README.md) for each phase remain executable:

| Test Suite Script | Phase Scope | Execution Result | Verified Artifacts |
| :--- | :--- | :---: | :--- |
| `python utils/test_phase0.py` | Phase 0 Setup | **PASS** | `configs/base_config.yaml`, `utils/seed.py`, `utils/logger.py` |
| `python utils/test_phase1.py` | Phase 1 Data | **PASS** | 18 raw files, dedup hash filter, script purity $\ge 90\%$ |
| `python utils/test_phase2.py` | Phase 2 Preprocessing | **PASS** | 60 spot-checked sentences, word/char vocabs, grapheme clusters |
| `python utils/test_phase12.py`| Phase 12 Ablation | **PASS** | 8 ablation configurations across 3 languages, model toggles |
| `python utils/test_phase13.py`| Phase 13 Explainability | **PASS** | Attention weights, Grad $\times$ Attn autograd, 36 heatmap cards |
| `python utils/test_phase14.py`| Phase 14 Evaluation | **PASS** | SHA-256 test checksums, single-run lock, hand-checked metrics |
| `python utils/test_phase15.py`| Phase 15 Serving | **PASS** | 33 smoke-test sentences, bundle shapes, FastAPI endpoints |

---

## 3. Serving & Live Demo Verification

The live Streamlit application was verified functional on `http://localhost:8501`:
- Multi-script selector responsive across Konkani, Assamese, and Sinhala.
- Entity badges render with correct color-coding (`PER`, `LOC`, `ORG`, `MISC`).
- Interactive Gradient $\times$ Attention HTML heatmaps render with proportional opacity scaling in $<20$ ms on GPU.
- CPU fallback execution path verified with zero crashes.
