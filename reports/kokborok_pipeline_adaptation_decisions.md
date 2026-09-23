# Phase 10 Authoritative Decision Document: Kokborok Pipeline Adaptation

**Phase**: 10 — Kokborok Pipeline Adaptation  
**Author**: Senior ML/NLP Research Engineer  
**Status**: COMPLETE (Authoritative Architectural & Pipeline Ledger)  
**Date**: 2026-09-23  

> [!IMPORTANT]
> **GOVERNING PRINCIPLE & EVIDENCE TRACEABILITY**  
> Every pipeline, normalization, and tokenization decision in this phase is grounded strictly in the empirical findings of Phase 9 ([`reports/kokborok_dataset_inspection.md`](kokborok_dataset_inspection.md)). Wherever Phase 8's provisional linguistic predictions conflicted with Phase 9's direct empirical observations, Phase 9 definitively governs.

---

## 1. Phase 8 vs. Phase 9 Reconciliation Matrix

| Dimension | Phase 8 Provisional Expectation | Phase 9 Empirical Finding | Phase 10 Architectural Resolution |
| :--- | :--- | :--- | :--- |
| **Script Orthography** | Dual-script uncertainty: Bengali script (`Beng`) vs. Roman script (`Latn`) | **100% Roman (Latin) script**. Exactly 0 Bengali characters found ($205,468$ Latin letters, $0$ Bengali codepoints). | **Discarded Brahmic pipeline**. Built clean Latin-alphabet normalizer and tokenizer. No virama, no conjunct repair, no danda normalization. |
| **Grapheme Clusters (`\X`)** | Predicted necessity of complex extended grapheme cluster regex for Brahmic conjuncts | Unnecessary for Roman script; single Latin letters with standard subword representations. | Simplified character tokenization to atomic Latin codepoints (safely handling accents if present). |
| **Agglutinative Case Suffixes** | Predicted bound suffixing morphology (`-no`, `-ni`, `-o`, `-bai`, `-rok`, `-song`) | Fully corroborated: over 3,800 tokens carry these exact case markers (`Tripurao`, `Station-o`, `Bangladeshirok`). | Tokenizer preserves hyphenated compounds (`Station-o`, `bank-o`) as unified tokens matching unhyphenated suffixes (`Tripurao`). |
| **File Formats & Schema** | Unknown | 6 Microsoft Excel `.xlsx` files: parallel pairs, sentiment data, code-switched LID text. | Dedicated ingestion and normalization parser pulling 4,020 valid sentences into `data/processed/kokborok/sentences.txt`. |
| **Existing NER Annotations** | Anticipated possibility of annotated entities | **ZERO NER annotations exist** in the raw files. | No external non-BIO annotation format conversion needed. Clean tokenized format prepared for Phase 11 annotation. |
| **Unicode Anomalies** | Anticipated possible diacritic corruption | Corrupted Windows-1252 apostrophe `’` into replacement char `\ufffd` (``) in 59 instances. | Built automated regex repair `(?<=\w)\ufffd(?=\w) -> '` in `preprocessing/kokborok_normalize.py`. |

---

## 2. Preprocessing Architecture & Component Specialization

### 2.1 Text Normalization (`preprocessing/kokborok_normalize.py`)
- **Canonical Unicode Normalization**: Applies standard Unicode `NFC` decomposition/recomposition.
- **Control Character Stripping**: Strips byte-order marks (`\ufeff`) and zero-width spaces (`\u200b`, `\u200e`, `\u200f`).
- **Mojibake Restoration**: Automatically repairs `\ufffd` between letters into standard apostrophe `'` (e.g. `That\ufffds` $\to$ `That's`, `today\ufffds` $\to$ `today's`).
- **Punctuation Standardization**: Maps curly quotes (`‘`, `’`, `“`, `”`) to standard ASCII quotes (`'`, `"`) and typographic dashes (`—`, `–`) to ASCII `-`.
- **LID Tag Stripping**: Strips inline code-switching tags (`/trp`, `/en`, `/tep`, etc.) while retaining lexical content.
- **Whitespace Consolidation**: Compresses consecutive whitespace runs into single spaces and strips margins.

### 2.2 Word & Character Tokenization (`preprocessing/kokborok_tokenizer.py`)
- **Word Token Regex**:
  $$\text{TOKEN\_REGEX} = \texttt{[\textbackslash p\{L\}\textbackslash p\{N\}]+(?:-[\textbackslash p\{L\}\textbackslash p\{N\}]+)* | [\textasciicircum\textbackslash s\textbackslash p\{L\}\textbackslash p\{N\}]}$$
  - Captures alphanumeric word stems and bound hyphenated case markers (`Station-o`, `bank-o`, `Angko-o`, `North-East`).
  - Completely separates sentence punctuation (`,`, `.`, `!`, `?`, `;`, `:`, `(`, `)`, `"`) into isolated tokens.
- **Subword Character Segmentation**: Decomposes words into individual character graphemes for the Char-CNN embedding layer.
- **Detokenizer**: Reconstructs fluent sentences with correct left-attaching punctuation (`,`, `.`, `!`, `?`) and right-attaching punctuation (`(`, `[`).

---

## 3. Label-Scheme & Format Conversion Analysis

1. **Pre-Existing Annotation Status**:
   - As established in Phase 9, the supplied dataset contains **0 Named Entity annotations**.
   - No span-offset JSON, BIO, or CoNLL tags exist in the source files.
2. **Target Label Scheme for Phase 11**:
   - Standard 9-tag BIO scheme: `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-MISC`, `I-MISC`.
   - 100% interoperable with the existing linear-chain CRF decoder used across Konkani, Assamese, and Sinhala.
3. **Format Conversion Validation**:
   - Since no pre-existing entity annotations exist, there was no third-party format to convert.
   - Text has been fully prepared and validated into segmented, normalized sentences ready for gold annotation and weak supervision in Phase 11.

---

## 4. Corpus Processing & Vocabulary Construction Results

All 4,020 sentences were extracted from the 6 read-only Excel files, normalized, and processed into standalone Kokborok artifacts under `data/processed/kokborok/`:

| Artifact | File Path | Scope / Description | Key Metric |
| :--- | :--- | :--- | :---: |
| **Normalized Sentences** | `data/processed/kokborok/sentences.txt` | Cleaned, normalized Kokborok sentences | **4,020 sentences** |
| **Total Word Tokens** | — | Total running tokens across corpus | **31,959 tokens** |
| **Word Vocabulary** | `data/processed/kokborok/word_vocab.json` | Case-sensitive word types + special tokens | **6,272 tokens** |
| **Character Vocabulary** | `data/processed/kokborok/char_vocab.json` | Unique Latin characters + special tokens | **81 characters** |

*Special Tokens Included*: `<PAD>` (ID 0), `<UNK>` (ID 1), `<BOS>` (ID 2), `<EOS>` (ID 3).

> [!NOTE]
> **ISOLATION PRINCIPLE CONFIRMED**  
> In strict compliance with the Phase 10 boundary rules, this vocabulary was constructed as an independent, standalone study. It has **NOT** been merged into the 3-language unified vocabulary tables (`checkpoints/`, `configs/`). That integration decision belongs exclusively to Phase 12.

---

## 5. Verification & Validation Summary

1. **Reconciliation-Completeness Test**: Every disagreement between Phase 8 and Phase 9 (Script, Brahmic vs. Latin, Conjuncts, Annotations) was resolved strictly in favor of Phase 9. (**PASS**)
2. **Manual Tokenization Spot-Check**: 20-sentence curated audit covering all sub-collections and edge cases achieved a **100% pass rate** (0 critical errors). (**PASS**)
3. **Label-Mapping Completeness**: The 9-tag BIO scheme is formally designated as the target ontology for Phase 11. (**PASS**)
4. **Read-Only Integrity Test**: Pre- and post-inspection SHA-256 checksums across all 6 files in `Kokborok_datasets/` confirmed 100% byte-identical immutability. (**PASS**)
