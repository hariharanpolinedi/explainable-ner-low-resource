# Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution

[![No Pretrained Models](https://img.shields.io/badge/Constraint-100%25%20From--Scratch-success.svg)](#absolute-project-wide-constraint)
[![Test Set Integrity](https://img.shields.io/badge/Gold--Test-Cryptographically%20Locked-blue.svg)](reports/gold_test_lockfile.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](requirements.txt)
[![Hardware](https://img.shields.io/badge/Hardware-GPU%20(CUDA)%20+%20CPU%20Fallback-orange.svg)](#environment-setup-instructions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A research-grade Named Entity Recognition (NER) and attention attribution explainability system engineered strictly from scratch for low-resource South Asian languages.**

An end-to-end neural sequence labeling pipeline built without any pretrained models, embeddings, or tokenizers—demonstrating that high-performance NER (>96% F1) is achievable for truly low-resource languages using carefully engineered architectures, weak supervision, and explainable attention mechanisms.

### Supported Languages
- **Konkani** (कोंकणी — Devanagari script)
- **Assamese** (অসমীয়া — Bengali-Assamese script)  
- **Sinhala** (සිංහල — Sinhala script)
- **Kokborok** (Roman / Latin — exploratory)

---

> [!IMPORTANT]
> **IMMUTABLE BASELINE V1 SNAPSHOT**  
> `baseline_v1/` is an immutable historical snapshot. Nothing under this directory is ever modified, retrained, or overwritten by any later phase. All v2 development happens outside this directory.  
> See [`baseline_v1/BASELINE_MANIFEST.md`](baseline_v1/BASELINE_MANIFEST.md) and [`reports/baseline_v1_reference_card.md`](reports/baseline_v1_reference_card.md).

---

## Absolute Project-Wide Constraint

> [!IMPORTANT]
> **STRICT ZERO-PRETRAINED ARTIFACTS RULE**  
> **NO pretrained models (e.g. BERT, RoBERTa, XLM-R), NO pretrained embeddings (Word2Vec, FastText, GloVe), NO pretrained tokenizers (SentencePiece, HuggingFace tokenizers, BPE libraries), and NO externally trained checkpoints were ever introduced into this project.**  
> Every component—from raw Wikimedia text filtering and custom Unicode extended grapheme cluster (`\X`) tokenizers, to Char-CNN word representations, multi-head self-attention mechanisms, linear-chain CRF decoders, and Gradient × Attention saliency attribution—was designed, engineered, and trained purely from scratch using local corpora and gold annotations.
>
> Full compliance audit trail: [NO_PRETRAINED_MODEL_CHECKLIST.md](NO_PRETRAINED_MODEL_CHECKLIST.md).

---

## Executive Results Summary

Evaluated on the cryptographically locked gold-test splits (`data/gold/{lang}/test.conll`, 200 sentences per language) touched **exactly once** in Phase 14:

| Language | Script Family | Official Checkpoint | Precision | Recall | **Entity F1 (Strict)** | **Token Accuracy** | Target (>90%) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | Devanagari | `checkpoints/final_konkani.pt` | 99.15% | 93.55% | **96.27%** | 99.47% | >90.0% | **MET (+6.27 pp)** |
| **Assamese** | Bengali-Assamese | `checkpoints/final_assamese.pt` | 95.83% | 100.00% | **97.87%** | 99.64% | >90.0% | **MET (+7.87 pp)** |
| **Sinhala** | Sinhala | `checkpoints/final_sinhala.pt` | 96.32% | 99.24% | **97.76%** | 99.70% | >90.0% | **MET (+7.76 pp)** |
| **Kokborok** | Roman / Latin | `checkpoints/kokborok_exp4_best.pt` | 77.78% | 87.50% | **82.35%** | 99.55% | Exploratory | **FINAL BEST** |
| **Macro Avg (4-Lang)**| — | — | **92.27%** | **95.07%** | **93.56%** | **99.59%** | — | — |

- **Out-of-Vocabulary Generalization**: In-Vocab F1 = **98.41%**, Out-of-Vocab F1 = **91.03%** (Char-CNN successfully captures unseen inflections).
- **Explainability Method**: **Gradient × Attention** ($\nabla_{\alpha} \text{score}$) selected over raw attention for causal emission grounding.
- **Detailed Paper Tables**: See [paper/results_tables.md](paper/results_tables.md).
- **Methodology Chapter**: See [paper/methodology.md](paper/methodology.md).
- **Scientific Limitations**: See [paper/limitations.md](paper/limitations.md).

---

## Architectural Overview

```
                        Raw Input Sentence
                                │
                                ▼
            Unicode NFC + Script-Specific Tokenization
            (Extended Grapheme Clusters '\X' via Regex)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 100d Word Embedding   90d Char-CNN Encoder    16d Lang Embedding
 (Random Uniform 0)    (Filters: 3, 4, 5)      (Dense Conditioning)
        └───────────────────────┬───────────────────────┘
                                │ (Concat -> 206d)
                                ▼
                   Bidirectional LSTM Encoder
                 (Hidden Dim: 128, Output: 256)
                                │
                                ▼
                 Multi-Head Self-Attention Layer  ◄─── Attribution Hook
              (H=4 heads, d_k=64, Residual + LN)       (Grad x Attn)
                                │
                                ▼
                Linear-Chain CRF Sequence Decoder
                  (Exact Viterbi Global Decoding)
                                │
                                ▼
                 Predicted Named Entity Spans
                 [PER, LOC, ORG, MISC] + Heatmaps
```

---

## Repository Structure

```
NLP-MARK-3/
├── configs/                  # Typed YAML configurations (base_config.yaml, final_{lang}.yaml)
├── data/
│   ├── raw/                  # Corpora, Kokborok sources, and newspaper source material
│   ├── gazetteers/           # Curated entity registries per language (PER, LOC, ORG)
│   ├── processed/            # Word and extended grapheme cluster vocabularies
│   ├── gold/                 # Gold annotated splits (train.conll, dev.conll, test.conll [locked])
│   ├── silver/               # High-precision weakly supervised datasets (7,200 sents/lang)
│   └── pseudo/               # Confidence-filtered self-trained datasets
├── preprocessing/            # Script-aware tokenizers, normalizers, grapheme segmentation
├── weak_supervision/         # Gazetteers, toponymic rules, distant supervision aggregators
├── model/                    # Neural architectures from scratch (Char-CNN, BiLSTM, Attention, CRF)
├── training/                 # Multilingual joint training, layer-unfreezing fine-tuning, augmentation
├── explainability/           # Attention extraction, Gradient x Attention, erasure tests, heatmaps
├── evaluation/               # Seqeval strict span metrics, confusion matrices, error categorization
├── experiments/              # Bounded hyperparameter search, ablation runner, consolidator
├── deployment/               # Production FastAPI REST service, Streamlit UI demo, versioned bundles
│   └── bundles/              # Self-contained packages (model.pt, vocabs, config, metadata)
├── checkpoints/              # Model weights trained from scratch (git-ignored)
├── reports/                  # Comprehensive research reports across all phases 0–16
├── paper/                    # Research paper source, assets, tables, and figures
├── utils/                    # Pinned config loaders, deterministic seeds, test suites
├── requirements.txt          # Minimal pinned dependencies
├── NO_PRETRAINED_MODEL_CHECKLIST.md # Running compliance ledger (Phases 0–16 audited)
└── README.md                 # Project guide & reproduction manual
```

---

## Environment Setup Instructions

### 1. Prerequisites
- Python 3.10 to 3.13 (Verified on Python 3.13.2)
- Git
- NVIDIA GPU with CUDA drivers (Optional; CPU fallback is fully verified)

### 2. Clone and Setup Environment
```bash
git clone <repo-url>
cd NLP-MARK-3

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Clean Installation (Dry Run)
```bash
# Verify configuration loader
python -m utils.config_loader --config configs/base_config.yaml

# Verify deterministic seeding
python utils/seed.py

# Verify append-only experiment logging
python utils/logger.py
```

---

## Step-by-Step Pipeline Reproduction

Every phase of the project can be independently audited and verified via automated test suites:

### Data & Preprocessing (Phases 1–3)
```bash
# Phase 1: Corpus acquisition & deduplication verification
python utils/test_phase1.py

# Phase 2: Unicode normalization & extended grapheme cluster tokenization
python utils/test_phase2.py

# Phase 3: Gold seed annotation & cryptographic test lockfile check
python utils/test_phase3.py
```

### Weak Supervision & Initial Modeling (Phases 4–7)
```bash
# Phase 4: Gazetteer matching & distant supervision silver assembly
python utils/test_phase4.py

# Phase 5: Word-level BiLSTM-CRF baseline evaluation
python utils/test_phase5.py

# Phase 6: Character-level CNN encoder integration & OOV analysis
python utils/test_phase6.py

# Phase 7: Multi-head self-attention layer & weight exposure sanity
python utils/test_phase7.py
```

### Multilingual Training & Optimization (Phases 8–12)
```bash
# Phase 8: Multilingual joint warm-start & language conditioning
python utils/test_phase8.py

# Phase 9: Layer-unfreezing specialization fine-tuning
python utils/test_phase9.py

# Phase 10: Confidence-filtered iterative self-training
python utils/test_phase10.py

# Phase 11: Targeted entity-substitution augmentation
python utils/test_phase11.py

# Phase 12: Bounded hyperparameter search & full master ablation suite
python utils/test_phase12.py
```

### Explainability & Official Evaluation (Phases 13–16)
```bash
# Phase 13: Gradient x Attention attribution & token-erasure faithfulness test
python utils/test_phase13.py

# Phase 14: Final locked gold-test evaluation & error categorization
python utils/test_phase14.py

# Phase 15: Serving verification & deployment bundle integrity
python utils/test_phase15.py

# Phase 16: Documentation packaging, cross-phase consistency & final audit
python utils/test_phase16.py
```

---

## Interactive Serving & Live Demo

The deployment package provides a high-performance REST API and an interactive Streamlit explanation dashboard:

### 1. Launch Interactive Streamlit UI
```bash
streamlit run deployment/app.py --server.port 8501
```
Navigate to **[http://localhost:8501](http://localhost:8501)** in your browser:
- **Language Selector**: Choose between Konkani, Assamese, and Sinhala.
- **Color-Coded Badges**: Visualizes `PER` (blue), `LOC` (green), `ORG` (orange), and `MISC` (purple) entities.
- **Attribution Heatmaps**: Renders per-token **Gradient × Attention** HTML overlays showing context words that causally drove each classification.

### 2. Launch FastAPI REST Service
```bash
uvicorn deployment.api:app --host 127.0.0.1 --port 8000
```
Test prediction via `curl`:
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sentence": "गोंय राज्याची राजधानी पणजी ही मांडवी न्हंयचे देगेर वसिल्ली आसा .", "language": "konkani"}'
```

---

## Research Reports Index

| Report Document | Topic & Scope |
| :--- | :--- |
| [reports/phase14_final_results.md](reports/phase14_final_results.md) | **Official Thesis Results**: Test F1, per-class metrics, trajectory |
| [reports/phase14_error_analysis.md](reports/phase14_error_analysis.md) | Qualitative & quantitative failure mode breakdown |
| [reports/phase12_ablation_table.md](reports/phase12_ablation_table.md) | Master 7-condition ablation matrix isolating each module |
| [reports/phase13_faithfulness_results.md](reports/phase13_faithfulness_results.md) | Token-erasure causal faithfulness test & method selection |
| [reports/phase13_head_wise_analysis.md](reports/phase13_head_wise_analysis.md) | Attention head functional specialization & entropy metrics |
| [reports/phase13_heatmap_gallery/README.md](reports/phase13_heatmap_gallery/README.md) | 36 HTML visual heatmap cards across all categories |
| [reports/gold_test_lockfile.md](reports/gold_test_lockfile.md) | Cryptographic SHA-256 lockfile for test split integrity |
| [reports/phase15_demo_smoketest.md](reports/phase15_demo_smoketest.md) | 33-sentence real-world deployment smoke test |

---

## Model Checkpoints

Model checkpoints are intentionally excluded from Git because the local collection is several gigabytes. Place the required weights in `checkpoints/` before inference; they are distributed separately rather than through the source repository.

| Language | Checkpoint | Entity F1 |
| :--- | :--- | :---: |
| Konkani | `final_konkani.pt` | 96.27% |
| Assamese | `final_assamese.pt` | 97.87% |
| Sinhala | `final_sinhala.pt` | 97.76% |
| Kokborok | `kokborok_exp4_best.pt` | 82.35% |

> **Note:** Checkpoints are excluded from Git because the local collection is several gigabytes. Place separately obtained checkpoints in `checkpoints/` before running inference.

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{polinedi2026explainable,
  title={Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution},
  author={Polinedi, Hariharan},
  year={2026},
  publisher={GitHub},
  url={https://github.com/hariharanpolinedi/explainable-ner-low-resource}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Wikimedia Foundation for providing corpora through Wikipedia dumps
- Native speakers who contributed to gold annotation and validation
- Research community for feedback on faithfulness metrics and attribution methods

---

**Built with ❤️ for low-resource language preservation and NLP accessibility**
