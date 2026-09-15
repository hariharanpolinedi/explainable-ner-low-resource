# Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution

**Target Languages**: Konkani, Assamese, Sinhala  
**Target Metric**: >90% entity-level F1 across all target languages  

---

## Absolute Project-Wide Constraint

> **Absolute project-wide constraint, applies to this phase and every later phase: NO pretrained models, pretrained embeddings, pretrained tokenizers/vocabularies, or externally trained checkpoints may ever be introduced. Everything is trained/built from scratch. Performance target for the eventual system is >90% entity-level F1 (not relevant to this phase's own output, but the logging/config system you build now must be capable of recording entity-F1 correctly once later phases produce it).**

Every phase must maintain strict compliance with this rule. Every phase adds a dated compliance record to [NO_PRETRAINED_MODEL_CHECKLIST.md](NO_PRETRAINED_MODEL_CHECKLIST.md).

---

## Project Overview

This research project develops an end-to-end, fully explainable Named Entity Recognition (NER) system tailored to low-resource languages (specifically Konkani, Assamese, and Sinhala). Due to the absence of massive annotated datasets and to prevent negative inductive bias or leakage from opaque pretraining corpuses, all components—including subword tokenizers, character/word embeddings, contextual encoders, CRFs, and attention attribution explainability modules—are designed, curated, and trained strictly from scratch without external checkpoints.

---

## Folder Skeleton

```
project/
├── configs/                  # Typed YAML configuration files (base_config.yaml, phase-specific configs)
├── utils/                    # Core utilities: config loader, deterministic seed, experiment logger
├── data_collection/          # Raw data scrapers, crawl scripts, corpus acquisition pipelines
├── data/                     # Data tiers partitioned by language:
│   ├── raw/                  # Unprocessed raw corpora (konkani, assamese, sinhala)
│   ├── gazetteers/           # Entity dictionaries, lexicons, gazetteer tables
│   ├── processed/            # Cleaned, normalized, sentence-tokenized data
│   ├── gold/                 # Human-verified gold standard NER datasets
│   ├── silver/               # High-confidence weakly supervised data
│   ├── pseudo/               # Pseudo-labeled / self-trained samples
│   └── augmented/            # Rule-augmented or synthetic NER training data
├── preprocessing/            # Text normalization, script handling, custom tokenization from scratch
├── annotation/               # Annotation guidelines, schema definitions, curation workflows
├── weak_supervision/         # Labeling functions, Snorkel/heuristic rules, label aggregators
├── model/                    # Model architectures built strictly from scratch (embeddings, encoders, CRF)
├── training/                 # Scratch training loops, loss functions, learning rate schedules
├── explainability/           # Attention attribution, gradient saliency, rationale extraction
├── evaluation/               # Entity-level strict/lenient F1, per-class metrics, error analysis
├── experiments/              # Experiment drivers, sweeps, and configuration matrices
├── deployment/               # Lightweight inference runtime, REST/gRPC wrappers
├── checkpoints/              # Model weights trained strictly from scratch (git-ignored)
├── reports/                  # Evaluation summaries, Phase reports, compliance audits
│   └── experiment_logs/      # Append-only experiment run records (CSV/JSONL)
├── paper/                    # Research paper manuscripts, LaTeX sources, figures, tables
├── notebooks/                # Exploration, visualization, and result-consolidation notebooks only
├── src/                      # Shared core domain logic
├── NO_PRETRAINED_MODEL_CHECKLIST.md  # Running compliance ledger
├── README.md                 # Project guide and environment documentation
└── requirements.txt          # Pinned project dependencies
```

*Note: All core reusable logic resides in `src/`, `utils/`, or the respective phase modules. The `notebooks/` directory is reserved strictly for interactive visualization, exploratory data inspection, and result consolidation.*

---

## Environment Setup Instructions

The project uses a clean Python virtual environment. Phase 0 requires only minimal dependencies (`PyYAML`). Future phases install additional dependencies on demand, maintaining strict zero-pretrained-model compliance.

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13.7)
- Git

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the Installation
Run the Phase 0 verification suite:
```bash
# Verify configuration loader
python -m utils.config_loader --config configs/base_config.yaml

# Verify seeding determinism
python utils/seed.py

# Verify experiment logging
python utils/logger.py
```

---

## How to Run a Phase

Every phase in the project follows a standardized, reproducible execution pattern:

1. **Configure**: Define parameters in `configs/base_config.yaml` or a phase-specific YAML extending the base schema. Never hardcode parameters.
2. **Seed**: Always initialize random state at the entry point of your script using `utils.seed.set_global_seed(config.seed)`.
3. **Execute**: Run the phase-specific pipeline script using typed configuration loading:
   ```bash
   python -m <phase_module>.run --config configs/phase_config.yaml --override key=value
   ```
4. **Log**: Record experiment metrics, config hashes, git commit hashes, and dataset versions using `utils.logger.ExperimentLogger`:
   ```python
   from utils.logger import ExperimentLogger
   logger = ExperimentLogger()
   logger.log_run(config_path="configs/base_config.yaml", metrics={"entity_f1": 0.925})
   ```
5. **Comply**: Verify that no pretrained models, tokenizers, or weights were introduced, and record a dated entry in `NO_PRETRAINED_MODEL_CHECKLIST.md`.

---

## Compliance Reference

Refer to [NO_PRETRAINED_MODEL_CHECKLIST.md](NO_PRETRAINED_MODEL_CHECKLIST.md) for the running ledger of compliance across all phases.
