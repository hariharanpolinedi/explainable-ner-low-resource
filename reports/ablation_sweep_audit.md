# Master Ablation Suite Scope Audit & Undertraining Exposure Analysis

**Document Purpose**: Comprehensive provenance audit of all seven ablation conditions in [`paper_assets/ablation_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/paper_assets/ablation_table.md) before final journal manuscript drafting.  
**Audit Script**: [`evaluation/audit_ablation_runner_scope.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/evaluation/audit_ablation_runner_scope.py)  
**Date of Audit**: `2026-09-18`  
**Underlying Incident**: Condition (b) "No Self-Attention" was previously found to be severely inflated (-37.41% drop vs. genuine -6.62% drop) due to undertraining in `experiments/ablation_runner.py` (data truncated to 1,000 sents/language, 4 epochs). This audit determines which of the remaining conditions were exposed to that same bug.

---

## 1. Provenance & Exposure Audit Table

| Condition | Code Path in `experiments/ablation_runner.py` | Training Data Volume | Epoch Count | Data-Matched (Y/N) | Epoch-Matched (Y/N) | Verdict | Rationale & Evidence |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **(a) No Char-CNN** | `train_ablation_model(use_char_cnn=False)` | 1,000 sents/lang (3,000 total) | 4 | **N** | **N** | **NEEDS RERUN** | Trained from scratch on truncated data for only 4 epochs; undertrained. |
| **(b) No Self-Attention** | `train_ablation_model(use_attention=False)` | 1,000 sents/lang (3,000 total) | 4 | **Y** | **Y** | **ALREADY CORRECTED** | Corrected in prior verification pass (v3 matched pool: 36,736 sents, 12 epochs). |
| **(c) No CRF** | `train_ablation_model(use_crf=False)` | 1,000 sents/lang (3,000 total) | 4 | **N** | **N** | **NEEDS RERUN** | Trained from scratch on truncated data for only 4 epochs with softmax; undertrained. |
| **(d) No Multilingual Joint Warm-Start** | `evaluate_checkpoint_on_gold_dev('fromscratch_control_{lang}.pt')` | Full Gold + Silver (8,050 sents/lang) | Up to 20 | **Y** | **Y** | **TRUST AS-IS** | Evaluates genuine Phase 9 monolingual control checkpoints; completely unaffected by runner script. |
| **(e) No Self-Training** | `evaluate_checkpoint_on_gold_dev('finetuned_{lang}.pt')` | Full Gold + Silver (8,050 sents/lang) | 9 + 4-6 | **Y** | **Y** | **TRUST AS-IS** | Evaluates genuine Phase 9 specialized checkpoints; unaffected by runner script. |
| **(f) No Augmentation** | `evaluate_checkpoint_on_gold_dev('selftrained_{lang}_best.pt')` | Full Gold + Silver + Pseudo (up to 12,050 sents/lang) | Phase 8+9+10 | **Y** | **Y** | **TRUST AS-IS** | Evaluates genuine Phase 10 best checkpoints prior to augmentation; unaffected by runner script. |
| **(g) Tokenization: Subword BPE vs Word+Char** | `train_bpe_tokenization_ablation()` | 1,000 sents/lang (3,000 total) | 4 | **N** | **N** | **NEEDS RERUN** | Trained from scratch on truncated data for only 4 epochs; undertrained. |

---

## 2. Key Findings

1. **Two Distinct Implementation Paradigms in Phase 12 Runner**:
   - **Pre-existing Checkpoint Evaluations (Conditions d, e, f)**:
     These conditions simply load checkpoints saved during Phase 9 (monolingual controls and fine-tuned models) and Phase 10 (self-trained best models) and evaluate them on Gold-Dev. These checkpoints received full training over dozens of epochs on the full corpus. They are **100% clean and uncompromised**.
   - **From-Scratch Retrained Models (Conditions a, b, c, g)**:
     These conditions re-instantiated models from scratch within `experiments/ablation_runner.py` and called `train_ablation_model()` with hardcoded arguments:
     ```python
     train_ds = MultilingualNERDataset(..., max_sentences_per_lang=1000)
     epochs = 4
     ```
     This truncation capped data to ~8-12% of the full dataset and ran only ~375 optimizer steps, leaving all four models severely undertrained.

2. **Impact on Reported Numbers**:
   - The reported ablation drops in `reports/phase12_ablation_table.md` for (a) -11.70%, (c) -17.21%, and (g) -13.26% conflate the true component removal with severe undertraining.
   - Exact parity retraining under the matched protocol (36,736 pooled sentences, 10-12 epochs, warmup + cosine decay) isolates the genuine marginal contribution of Char-CNN, CRF, and tokenization.
