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

---

## Phase 6: Character-Level CNN Encoder Integration & OOV Analysis
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - From-scratch character-level CNN word encoder: `model/char_cnn.py` (`CharCNNWordEncoder` with 50d randomly initialized character/grapheme embeddings over Phase 2 `char_vocab.json`, 1D convolutions with filter banks [3,4,5] or [2,3,4], max-over-time pooling, and concatenated 90d representation).
  - Integrated Word+Char-CNN BiLSTM-CRF model: `model/ner_model_v2_charcnn.py` (`NERCharCNNModel` concatenating 100d word embedding with 90d char-CNN vector -> 190d combined representation with 0.3 dropout -> BiLSTM -> LinearChainCRF).
  - Joint token and grapheme-character data loader: `training/train_charcnn.py` (`NERCharCoNLLDataset` and `collate_fn_charcnn` with script-aware grapheme clustering).
  - Comparative OOV vs In-Vocab evaluation module: `evaluation/oov_analysis.py` (splitting gold-dev entities and tokens into IV and OOV subsets to measure subword generalization signature).
  - Experiment orchestration runner: `training/run_phase6_experiments.py` (executing primary Char-CNN training and filter-width sensitivity experiments).
  - Best Char-CNN model checkpoints: `checkpoints/charcnn_{konkani,assamese,sinhala}.pt`.
  - Analytical reports: `reports/phase6_charcnn_results.md` (overall and OOV-split metrics vs Phase 5) and `reports/phase6_filter_width_experiment.md` (narrow filter bank analysis).
  - Comprehensive verification test suite: `utils/test_phase6.py` and test report `reports/phase6_test_results.md`.
  - Config schema extension: `char_cnn:` hyperparameter block in `configs/base_config.yaml` and `CharCNNConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained character/subword models: Character embeddings were randomly initialized (`nn.init.uniform_(-0.1, 0.1)`) with padding index zeroed, learned purely from the NER objective. No character language models (e.g. ELMo, CharacterBERT, FLair), no byte-pair models from outside, and no external vectors were loaded.
  - Zero pretrained tokenizers: Grapheme clusters segmented purely using Phase 2's script-aware Unicode regex engine (`\X`) and Phase 2's `char_vocab.json`.
  - Architecture restriction compliance: Self-attention and attention attribution mechanisms were strictly NOT implemented (reserved exclusively for Phase 7).
  - Hardware & execution tracking: All training and sensitivity runs executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with runs logged to `reports/experiment_logs/experiments.csv`.

---

## Phase 7: Multi-Head Self-Attention Integration with Weight Instrumentation
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - From-scratch Multi-Head Self-Attention module: `model/self_attention.py` (`MultiHeadSelfAttentionLayer` and `MultiHeadSelfAttentionBlock` implementing scaled dot-product attention via raw tensor operations with linear Q/K/V projections, numerical padding mask enforcement, residual connection, and LayerNorm).
  - Attention-instrumented NER model: `model/ner_model_v3_attention.py` (`NERAttentionModel` wiring word embedding + char-CNN -> BiLSTM -> self-attention block -> Linear-Chain CRF, persisting full per-head attention weight tensors on `model.last_attention_weights`).
  - Self-attention training pipeline: `training/train_attention_model.py` (with linear learning rate warmup and early stopping on gold-dev entity F1).
  - Experiment orchestration runner: `training/run_phase7_experiments.py` (executing primary attention runs, 4 vs 8 head ablation, 1 vs 2 block ablation, and span length stratification).
  - Span-length-stratified evaluation module: `evaluation/span_length_analysis.py` (evaluating single-token vs multi-token entity F1).
  - Attention non-degeneracy sanity prototype: `explainability/attention_extractor_prototype.py` (programmatically confirming attention weights are neither uniform nor self-only across sampled entities).
  - Best attention checkpoints: `checkpoints/attention_{konkani,assamese,sinhala}.pt`.
  - Analytical reports: `reports/phase7_attention_results.md`, `reports/phase7_span_length_analysis.md`, `reports/phase7_attention_sanity_examples.md`.
  - Comprehensive verification test suite: `utils/test_phase7.py` and test report `reports/phase7_test_results.md`.
  - Config schema extension: `attention:` hyperparameter block in `configs/base_config.yaml` and `AttentionConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained transformer models or weights: All attention projections ($W_Q, W_K, W_V, W_O$) and LayerNorm parameters were initialized completely from scratch via Xavier uniform and learned purely from the NER objective. No HuggingFace Transformers, PyTorch pretrained models, BERT, RoBERTa, or external attention checkpoints were loaded or instantiated.
  - Padding mask hygiene: Attention scores to padding tokens are strictly masked out numerically ($A_{i, \text{pad}} = 0.00000000$), preserving exact padding invariance (difference $< 10^{-6}$).
  - Architecture restriction compliance: Explainability attribution methods (Integrated Gradients, attention rollout, faithfulness tests) were strictly NOT implemented (reserved exclusively for Phase 13). Multilingual joint warm-start training was NOT implemented (reserved for Phase 8).
  - Hardware & execution tracking: All training, ablation, and sanity evaluations executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with runs logged to `reports/experiment_logs/experiments.csv`.

---

## Phase 8: Multilingual Joint Warm-Start Training & Ablation Matrix
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - From-scratch language conditioning module: `model/language_conditioning.py` (`LanguageEmbedding` mapping integer language IDs to 16-dimensional dense learned vectors, initialized from scratch via standard normal distribution).
  - Multilingual NER model assembly: `model/ner_model_v4_multilingual.py` (`NERMultilingualModel` concatenating word + char-CNN + language embedding -> BiLSTM -> Multi-Head Self-Attention -> configurable CRF head, supporting both shared and per-language CRF heads).
  - Multilingual data pooling & batch sampling: `training/multilingual_dataset.py` (`MultilingualNERDataset` combining 23,400 gold+silver sentences tagged by language ID, and `MultilingualBatchSampler` supporting uniform and proportional sampling strategies).
  - Multilingual training loop: `training/train_joint_warmstart.py` (trained from fresh random initialization, early stopping on macro gold-dev entity F1).
  - Multilingual comparative evaluation pipeline: `evaluation/compare_joint_vs_monolingual.py` (evaluating joint models on each language's own gold-dev separately against Phase-7 monolingual baselines).
  - 2x2 matrix experiment orchestrator: `training/run_phase8_experiments.py` (executing all 4 configurations: shared-uniform, shared-proportional, per_language-uniform, per_language-proportional).
  - Retained checkpoints: `checkpoints/joint_best.pt` (winning configuration: `shared_uniform`), plus all matrix checkpoints `checkpoints/joint_{shared,per_language}_{uniform,proportional}.pt`.
  - Comprehensive reports: `reports/phase8_joint_vs_monolingual.md` (full 2x2 matrix results, cross-lingual transfer analysis) and `reports/phase8_test_results.md`.
  - Verification test suite: `utils/test_phase8.py`.
  - Config extensions: `multilingual:` block in `configs/base_config.yaml` and `MultilingualConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained components: All model weights, language embeddings, linear projections, and CRF transitions were randomly initialized and trained purely from scratch on this project's own pooled datasets. No pretrained multilingual model (e.g. mBERT, XLM-R, IndicBERT) or external checkpoint was ever imported or used.
  - Multi-task learning from scratch: Data pooling and language conditioning represent standard from-scratch multi-task learning, strictly free of external pretraining.
---

## Phase 9: Per-Language Fine-Tuning with Layer-Unfreezing Schedule
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Layer-wise unfreezing scheduler: `training/layer_unfreeze_schedule.py` (`LayerUnfreezeScheduler` managing lower-layer parameter locking during early epochs and progressive unfreezing).
  - Per-language fine-tuning & control pipeline: `training/finetune_per_language.py` (`train_finetune_experiment` supporting fine-tuning from `checkpoints/joint_best.pt` and from-scratch control training).
  - Experiment orchestrator: `training/run_phase9_experiments.py` (executing fine-tuned vs control comparisons and schedule sensitivity across Konkani, Assamese, and Sinhala).
  - Fine-tuned best checkpoints: `checkpoints/finetuned_{konkani,assamese,sinhala}.pt`.
  - From-scratch control checkpoints: `checkpoints/fromscratch_control_{konkani,assamese,sinhala}.pt`.
  - Analytical reports: `reports/phase9_finetune_results.md` and `reports/phase9_test_results.md`.
  - Verification test suite: `utils/test_phase9.py`.
  - Config extensions: `finetune:` section in `configs/base_config.yaml` and `FinetuneConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE (Explicit Compliance Clarification)
- **Audit Details**:
  - Internal checkpoint warm-start compliance: Initializing per-language fine-tuning from `checkpoints/joint_best.pt` is strictly permitted and 100% compliant with the project-wide constraint. The Phase 8 joint model was built from scratch and trained exclusively on this project's own gold and silver data across Phases 1–8. It is not an external or third-party pretrained checkpoint.
  - Zero external models/weights: No external pretrained transformer, embedding, or tokenizer weights were loaded or referenced.
  - Freezing mechanics hygiene: Verified numerically that frozen lower-layer weights (`word_embedding`, `char_cnn`, `encoder`) underwent strictly zero updates during scheduled frozen epochs.
  - Architecture restriction compliance: Neither pseudo-labeling / self-training (reserved for Phase 10) nor rule-based/morphological data augmentation (reserved for Phase 11) nor attribution pipelines (reserved for Phase 13) were implemented in this phase.
  - Hardware & execution tracking: All fine-tuning, control, and sensitivity runs executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with runs logged to `reports/experiment_logs/experiments.csv`.

---

## Phase 10: Iterative Self-Training / Pseudo-Labeling with Confidence Tightening
- **Date**: 2026-09-16
- **Status**: COMPLIANT
- **Components Introduced**:
  - Test-set isolation guard: `evaluation/test_set_isolation_guard.py` (strictly asserting zero overlap between pseudo-labeling candidates and `data/gold/{lang}/test.conll` across all 3 languages, with immediate `CriticalLeakageError` abort on contamination).
  - Confidence-based inference pipeline: `training/confidence_filter.py` (calculating exact CRF normalized sequence confidence $\exp((s(x, y^*) - \log Z(x))/T)$ on candidate unlabeled sentences and filtering items $\ge \text{threshold}$).
  - Self-training multi-round loop: `training/self_training_loop.py` (executing iterative rounds with tightening thresholds 0.90 -> 0.92, retraining with AdamW $\text{lr}=10^{-4}$ on gold+silver+cumulative_pseudo, automated checkpoint rollback upon regression, and stopping rule evaluation on 2-round improvement $<0.5$ F1 points).
  - Self-training curve tracking & visualization: `evaluation/plot_selftraining_curve.py` (generating `reports/phase10_selftraining_curve.png` and `reports/phase10_selftraining_curve.md`).
  - Pseudo-labeled datasets: `data/pseudo/{konkani,assamese,sinhala}/round_{1,2}.conll`.
  - Checkpoints: `checkpoints/selftrained_{konkani,assamese,sinhala}_best.pt` and per-round checkpoints `checkpoints/selftrained_{lang}_round{n}.pt`.
  - Comprehensive reports: `reports/phase10_selftraining_curve.md`, `reports/phase10_test_set_isolation_confirmation.md`, and `reports/phase10_test_results.md`.
  - Verification test suite: `utils/test_phase10.py`.
  - Config extensions: `self_training:` block in `configs/base_config.yaml` and `SelfTrainingConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Internal pseudo-labeling compliance: Pseudo-labels were generated exclusively by this project's own Phase 9 fine-tuned models (`checkpoints/finetuned_{lang}.pt`) on this project's own unlabeled pool (`data/processed/{lang}/sentences.txt`). No third-party or external pretrained NER tagger, POS tagger, LLM, or teacher model was ever utilized or imported.
  - Zero test-set leakage: Formally and cryptographically verified that `data/gold/{lang}/test.conll` remained strictly isolated, unread, and 100% disjoint from all candidate and pseudo-labeled pools (`test_overlap_count = 0`).
  - Checkpoint integrity & rollback: Any round that regressed on dev-F1 was immediately rolled back to the previous best checkpoint; no regressed model was ever carried forward or promoted to `checkpoints/selftrained_{lang}_best.pt`.
  - Volume cap adherence: Cumulative pseudo-data volume was strictly capped below $3\times$ combined gold+silver volume ($< 23,400$ sentences).
  - Hardware & execution tracking: All confidence inference passes and retraining rounds executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with runs logged to `reports/experiment_logs/experiments.csv`.

---

## Phase 11: Data Augmentation & Class-Imbalance Handling
- **Date**: 2026-09-17
- **Status**: COMPLIANT
- **Components Introduced**:
  - Entity-substitution augmentation module: `training/entity_substitution.py` (`EntitySubstitutionAugmenter` swapping entity spans with same-category entries from Phase 1 gazetteers, applying language agreement heuristics, and maintaining strict BIO tag validity).
  - Synonym-swap augmentation module: `training/synonym_swap.py` (`SynonymSwapAugmenter` substituting non-entity 'O' tokens using hand-curated and rule-derived thesauri from Phase 1/2 corpus statistics).
  - Class-imbalance loss module: `training/class_weighted_loss.py` (`ClassWeightedCRFLoss` and `FocalCRFLoss` with modulating factor $(1-p_t)^\gamma$).
  - Augmentation orchestration pipeline: `training/augmentation.py` (generating `data/augmented/{lang}/` datasets, performing naturalness spot-checks on 20 sentences per language, and executing isolated experiments).
  - Synthetic datasets: `data/augmented/{konkani,assamese,sinhala}/{entity_sub,synonym_swap,combined_augmented}.conll`.
  - Checkpoints: `checkpoints/augmented_assamese.pt`, `checkpoints/augmented_sinhala.pt`, and explicit retention note for `checkpoints/selftrained_konkani_best.pt`.
  - Comprehensive reports: `reports/phase11_augmentation_results.md`, `reports/phase11_naturalness_spotcheck.md`, and `reports/phase11_test_results.md`.
  - Verification test suite: `utils/test_phase11.py`.
  - Config extensions: `augmentation:` block in `configs/base_config.yaml` and `AugmentationConfig` in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
---

## Phase 12: Hyperparameter Optimization, Full Ablation Suite, and Final Configuration Lock
- **Date**: 2026-09-17
- **Status**: COMPLIANT
- **Components Introduced**:
  - Bounded random hyperparameter search module: `experiments/hparam_search.py` (15 trials systematically sampling hidden size {128, 192, 256}, attention heads {4, 8}, attention blocks {1, 2}, dropout {0.3, 0.4, 0.5}, learning rate {5e-4, 1e-3, 2e-3}, language embedding dimension {8, 16, 32}, and batch size {16, 32}).
  - Full ablation suite runner: `experiments/ablation_runner.py` (isolating (a) char-CNN, (b) self-attention, (c) CRF, (d) multilingual joint warm-start, (e) self-training, (f) augmentation, and (g) from-scratch BPE tokenization across all three languages).
  - Master ablation consolidation & MUST VERIFY resolution: `experiments/consolidate_ablation_table.py`.
  - Architectural ablation toggles in `model/ner_model_v4_multilingual.py`: `use_char_cnn`, `use_attention`, and `use_crf` (supporting both shared and per-language softmax decoding).
  - Final locked configurations: `configs/final_{konkani,assamese,sinhala}.yaml`.
  - Final locked model checkpoints: `checkpoints/final_{konkani,assamese,sinhala}.pt`.
  - Comprehensive research reports: `reports/phase12_ablation_table.md`, `reports/phase12_hparam_search.md`, `reports/phase12_must_verify_resolution.md`, and `reports/phase12_test_results.md`.
  - Automated test suite: `utils/test_phase12.py` (verifying toggle correctness, search-space coverage, ablation completeness, and final reproducibility).
  - Config extensions: `search:` and `ablation:` blocks in `configs/base_config.yaml` and `SearchConfig`, `AblationConfig` dataclasses in `utils/config_loader.py`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained models or tokenizers: All models trained from fresh random initialization; the tokenization ablation employed pure Python Byte-Pair Encoding (`FromScratchBPETokenizer`) fitted strictly on local project corpora without any HuggingFace tokenizers, sentencepiece, or third-party tokenizers.
  - Zero test-set evaluation: All hyperparameter comparisons, ablation delta computations, and final model verifications were conducted exclusively on gold-dev (`data/gold/{lang}/dev.conll`). The gold test sets (`data/gold/{lang}/test.conll`) remained 100% untouched and unread throughout.
  - Scope constraint compliance: Explainability and attention attribution pipelines (reserved for Phase 13) were strictly NOT implemented in this phase.
  - Hardware & execution tracking: All search trials and ablation trainings executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with runs logged to `reports/experiment_logs/experiments.csv`.

---

## Phase 13: Explainability Pipeline (Gradient × Attention, Erasure Faithfulness Test) and Method Selection
- **Date**: 2026-09-17
- **Status**: COMPLIANT
- **Components Introduced**:
  - Attention extraction module: `explainability/attention_extractor.py` (retrieving per-head, per-token attention weight matrices from frozen models without modification).
  - Gradient × attention attribution pipeline: `explainability/gradient_attention_attribution.py` (computing backward gradients of entity emission scores w.r.t attention weights, calculating element-wise product Grad × Attn with magnitude normalization).
  - Attention rollout stub: `explainability/attention_rollout.py` (reproducible stub for multi-layer rollout, clearly marked not-invoked due to 1-block architecture resolution).
  - Quantitative erasure faithfulness test: `explainability/erasure_faithfulness_test.py` (measuring top-k vs random-k confidence drop across >=100 predictions per language on gold-dev).
  - Head-wise specialization analyzer: `explainability/head_wise_analysis.py` (analyzing functional differentiation across attention heads 0–3).
  - Attribution heatmap visualizer: `explainability/heatmap_visualizer.py` (generating 36 HTML visual cards and index gallery in `reports/phase13_heatmap_gallery/`).
  - Automated test suite: `utils/test_phase13.py` (5 automated validation checks passing).
  - Config additions: `explainability:` block in `configs/base_config.yaml` and `ExplainabilityConfig` dataclass in `utils/config_loader.py`.
  - Comprehensive research reports: `reports/phase13_faithfulness_results.md`, `reports/phase13_head_wise_analysis.md`, `reports/phase13_test_results.md`, and `reports/phase13_heatmap_gallery/README.md`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Zero pretrained interpretability packages: Attribution was computed strictly using native PyTorch autograd and tensor operations directly on this project's own model architecture. Zero imports of Captum, transformers-interpret, or external transformer interpretability toolkits.
---

## Phase 14: Final Locked Gold-Test Evaluation & Error Analysis
- **Date**: 2026-09-17
- **Status**: COMPLIANT
- **Components Introduced**:
  - Final test evaluation runner: `evaluation/final_test_eval.py` (cryptographic SHA-256 gold-test integrity verification, single-evaluation-run guard with official lockfile, strict seqeval entity-level P/R/F1, token-level accuracy, per-entity-type breakdown, and Phase 8 joint model direct comparison).
  - Confusion matrix generator: `evaluation/confusion_matrix.py` (9x9 BIO tag confusion matrices and span-level entity confusion matrices per language, rendered via matplotlib into `reports/phase14_confusion_matrices/`).
  - Deterministic error categorization: `evaluation/error_categorization.py` (rule-based categorization of all test discrepancies into boundary errors, type-confusion errors, missed entities / FN, and spurious entities / FP with contextual token spans and linguistic diagnoses).
  - Out-of-vocabulary evaluation: `evaluation/oov_analysis.py` (reused to evaluate in-vocab vs out-of-vocab entity F1 and token accuracy against training corpus vocabulary).
  - Automated validation suite: `utils/test_phase14.py` (4 automated verification checks passing: test-set integrity, single-run guard, metric correctness against hand-computed benchmark, confusion matrix and error categorization completeness).
  - Comprehensive research reports: `reports/phase14_final_results.md`, `reports/phase14_error_analysis.md`, `reports/phase14_test_results.md`, and `reports/experiment_logs/phase14_test_eval_lock.json`.
  - Visual artifacts: `reports/phase14_confusion_matrices/{konkani,assamese,sinhala}.png` and `reports/phase14_confusion_matrices/{konkani,assamese,sinhala}_entity_matrix.png`.
- **Pretrained Artifacts Used**: NONE
- **Audit Details**:
  - Pure from-scratch evaluation tooling: All metric computation, seqeval strict span extraction, OOV partitioning, and error categorization were implemented purely using standard Python, numpy, and matplotlib; zero third-party pretrained models, pretrained tokenizers, or external NLP classification APIs were introduced into the evaluation or error-analysis tooling.
  - Single-evaluation discipline strictly honored: Gold-test splits (`data/gold/{lang}/test.conll`) were touched and evaluated exactly once in this phase. Zero test-set peeking, best-of-N selection, hyperparameter adjustment, or retraining was performed.
  - Frozen model compliance: The Phase-12 final checkpoints (`checkpoints/final_{lang}.pt`) and Phase-8 joint checkpoint (`checkpoints/joint_best.pt`) were loaded as frozen inputs; inference-only execution with zero gradient backpropagation.
  - Cryptographic test-set integrity: Verified byte-for-byte that all three gold-test splits matched their locked SHA-256 checksums from Phase 3 prior to inference.
  - Hardware & execution tracking: All test inferences were executed on local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`) with the single official run logged to `reports/experiment_logs/experiments.csv`.
