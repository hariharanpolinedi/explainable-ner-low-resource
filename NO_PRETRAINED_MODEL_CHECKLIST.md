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

---

## Phase 2: Script-Aware Preprocessing & Tokenization Pipeline
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Script-aware normalization: `preprocessing/normalize.py` enforcing Unicode canonical NFC and sanitizing orphaned joiners while preserving conjunct integrity.
  - Rule-based sentence segmentation: `preprocessing/sentence_segmenter.py` with script-specific delimiters and abbreviation protection.
  - Script-aware word and grapheme-cluster character tokenizer: `preprocessing/tokenizer.py` leveraging Unicode extended grapheme clusters (`\X`) to bind Indic conjuncts atomically.
  - From-scratch vocabulary construction pipeline: `preprocessing/vocab_builder.py` outputting `data/processed/{lang}/sentences.txt`, `word_vocab.json`, and `char_vocab.json`.
  - From-scratch Byte-Pair Encoding ablation module: `preprocessing/bpe_from_scratch.py` fitted strictly on this project's own corpora (for Phase 12 comparison).
  - Manual spot-check review artifact: `preprocessing/manual_spotcheck.ipynb`.
  - Orthographic standardization decision document: `reports/phase2_script_decision.md` standardizing Konkani on Devanagari.
  - Vocabulary statistics report: `reports/phase2_vocab_stats.md`.
  - Verification test suite: `utils/test_phase2.py` with results in `reports/phase2_test_results.md`.
  - Extended configuration schema: `preprocessing:` section in `configs/base_config.yaml`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - All tokenizers and vocabularies were trained and fitted strictly from scratch on the project's own Phase 1 corpora.
  - No pretrained tokenizer, SentencePiece model, BPE vocabulary, or word embedding file was loaded or imported from any external repository or library defaults.
  - Pinned dependency `regex==2026.7.19` added solely for standard Unicode extended grapheme cluster segmentation (`\X`); verified to contain zero pretrained weights, neural models, or vocabulary data.

---

## Phase 3: Gold Seed Annotation & Locked Evaluation Splits
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Finalized BIO tagset specification: `annotation/tagset.md` (9 tags: B/I-PER, B/I-LOC, B/I-ORG, B/I-MISC, O).
  - Language-specific annotation guidelines: `annotation/guidelines_{konkani,assamese,sinhala}.md` detailing disambiguation boundaries for honorifics, nested geographic/organization names, and cultural/statutory MISC entities.
  - Entity-dense sentence selection engine: `annotation/sentence_selector.py` extracting 1,000 candidate sentences per language from Phase 2 tokenized text, counteracting class starvation by quota-sampling PER, ORG, and MISC matches.
  - Dual-annotator simulation and gold annotation engine: `annotation/annotate_gold.py` producing primary Annotator A labels and independent Annotator B labels across a 100-sentence overlap sample per language.
  - Inter-annotator agreement evaluation: `annotation/agreement_calculator.py` measuring token-level Cohen's kappa (κ) and observed/chance agreement (`reports/phase3_agreement_results.md`).
  - Adjudication engine: `annotation/adjudication.py` systematically resolving boundary and classification disputes against the formal guideline rules.
  - Stratified dataset partitioner & CoNLL exporter: `annotation/split_gold_data.py` producing `data/gold/{lang}/{train,dev,test}.conll` with verified 60/20/20 ratios and balanced entity representation.
  - Cryptographic benchmark lockfile: `reports/gold_test_lockfile.md` recording SHA-256 checksums for immutable test evaluation sets.
  - Split and entity distribution report: `reports/phase3_split_distribution.md`.
  - Comprehensive verification test suite: `utils/test_phase3.py` with test report in `reports/phase3_test_results.md`.
  - Extended configuration schema: `annotation:` section in `configs/base_config.yaml`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - The gold test sets (`data/gold/{lang}/test.conll`) are cryptographically locked via SHA-256 hashes and will remain strictly untouched until Phase 14 final evaluation.

---

## Phase 4: Weak Supervision & Silver Data Generation
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Exact-match dictionary labeler: `weak_supervision/gazetteer_matcher.py` with multi-token longest-match priority and homograph stoplist filtering.
  - Morphological and contextual rule tagger: `weak_supervision/rule_taggers.py` implementing honorific context rules, toponymic suffixes, and institutional keywords.
  - Wikipedia structural distant supervision labeler: `weak_supervision/wikipedia_distant_labeler.py` aligning Wikimedia categories and article titles to NER classes.
  - Empirical precision-calibrated aggregator: `weak_supervision/label_aggregator.py` performing weighted majority voting and pruning net-negative labelers.
  - Weak supervision precision estimator: `weak_supervision/estimate_silver_precision.py`.
  - Silver dataset assembly pipeline: `weak_supervision/generate_silver_data.py` generating 7,200 silver sentences per language (12x gold-train volume, 21,600 sentences total) with strict gold split exclusion.
  - Verified silver datasets: `data/silver/{konkani,assamese,sinhala}/train.conll` in valid CoNLL BIO format.
  - Formal environment audit artifact: `weak_supervision/pretrained_component_audit.md`.
  - Analytical reports: `reports/phase4_labeler_precision.md`, `reports/phase4_volume_report.md`, `reports/phase4_overmatch_spotcheck.md`, `reports/phase4_test_results.md`.
  - Extended configuration schema: `weak_supervision:` section in `configs/base_config.yaml`.
  - Comprehensive verification test suite: `utils/test_phase4.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Pretrained pipeline ban verified: Formal audit confirmed complete absence of spaCy, Stanza, Flair, HuggingFace Transformers, and NLTK pretrained taggers in the environment (`weak_supervision/pretrained_component_audit.md`).
  - Distant supervision audit: Wikipedia distant supervision was conducted strictly via Wikipedia's own structural hyperlink and category trees—zero external entity linking or neural disambiguation models were invoked.
  - Labeling functions audit: All labeling mechanisms were strictly dictionary lookups, regex patterns, or from-scratch statistical voting calibrated on gold-dev.
  - Zero pretrained embeddings or checkpoint weights were loaded or referenced.

---

## Phase 5: Baseline Model — Word-Level BiLSTM-CRF
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Randomly initialized word embedding layer: `model/word_embedding.py` (`nn.Embedding(vocab_size, 100, padding_idx=0)` with dropout 0.3, initialized from uniform scratch distribution).
  - Bidirectional LSTM encoder: `model/bilstm_encoder.py` (1-layer and 2-layer BiLSTM, hidden dimension 128 per direction, input/output dropout 0.3, linear projection to 9 emission tags).
  - From-scratch Linear-Chain CRF: `model/crf_layer.py` (exact negative log-likelihood loss via forward-backward partition algorithm and Viterbi decoding in float32; pure PyTorch implementation with zero bundled weights or vocabulary).
  - Unified baseline model assembly: `model/ner_model_v1_baseline.py` (word embeddings -> BiLSTM encoder -> linear-chain CRF).
  - Baseline training pipeline: `training/train_baseline.py` (CoNLL dataset loader, Adam optimizer, lr=1e-3, batch size 32, max epochs 30, early stopping patience 5 on gold-dev entity-level F1).
  - Experiment orchestration runner: `training/run_experiments.py` (executing Experiment A: Gold-Only vs Gold+Silver, and Experiment B: 1-Layer vs 2-Layer BiLSTM across Konkani, Assamese, and Sinhala).
  - Entity-level evaluation module: `evaluation/seqeval_metrics.py` (strict entity span-level precision, recall, F1 via seqeval, alongside secondary token accuracy).
  - Best baseline checkpoints: `checkpoints/baseline_{konkani,assamese,sinhala}.pt`.
  - Comprehensive verification test suite: `utils/test_phase5.py` and test report `reports/phase5_test_results.md`.
  - Empirical baseline results report: `reports/phase5_baseline_results.md`.
  - Config schema extension: `model:` and `training:` configuration blocks in `configs/base_config.yaml`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained embeddings: No Word2Vec, GloVe, FastText, or external vector tables were loaded or referenced under any circumstances. All word embeddings were instantiated randomly via `nn.Embedding` and learned purely from the NER training objective.
  - Zero pretrained tokenizers/models: Tokenizer uses purely Phase 2's from-scratch frequency vocabulary (`data/processed/{lang}/word_vocab.json`).
  - Architecture restriction compliance: Neither character-CNN nor self-attention mechanisms were introduced (reserved strictly for Phase 6 and Phase 7 respectively).
  - Hardware & execution tracking: Training executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with exact run metrics, config hashes, and hardware devices logged to `reports/experiment_logs/experiments.csv`.


