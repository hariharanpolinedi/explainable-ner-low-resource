# Phase 2 Script Decision: Konkani Orthographic Standardization

**Date**: 2026-09-16  
**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Focus Language**: Konkani (`gom` / `kok`)  
**Status**: APPROVED & CODIFIED  

---

## 1. Context & Sociolinguistic Background

Konkani is a Southern Indo-Aryan language spoken along the Konkan coast of Western India (primarily in Goa, coastal Karnataka, southern Maharashtra, and northern Kerala). Uniquely among Indian constitutional languages, Konkani has historically been written across five distinct orthographies:
1. **Devanagari** (the official script of Goa and standard for academic/literary publication)
2. **Romi / Roman** (widely used in the Goan Catholic community and Tiatr theatre)
3. **Kannada script** (predominantly used by Konkani speakers in Karwar, Mangalore, and coastal Karnataka)
4. **Malayalam script** (used by Konkani communities in Kochi and northern Kerala)
5. **Perso-Arabic** (historically used by Navayath Muslims in Bhatkal)

---

## 2. Empirical Audit of the Phase 1 Konkani Corpus

An automated script-character inspection of all raw text files collected in Phase 1 (`data/raw/konkani/*.txt`) yielded the following distribution:

| Script / Category | Unicode Range | Character Count | Percentage of Native Text |
| :--- | :---: | :---: | :---: |
| **Devanagari** | `U+0900 – U+097F` | **6,111,503** | **99.9999%** |
| **Kannada** | `U+0C80 – U+0CFF` | **4** | < 0.0001% (stray symbols) |
| **Latin** | `U+0041 – U+007A` | 20,078 | ~0.3% (acronyms, references, URLs) |

The raw corpus demonstrates near-total homogeneity in the Devanagari script, reflecting the official curation standards of the `gom.wikipedia.org` repository and Goan publications such as *Bhaangarbhuin*.

---

## 3. Official Project Decision

> **Decision**: The project **standardizes strictly on the Devanagari script** for all Konkani data, preprocessing, tokenization, gazetteers, annotation, and model architectures. Any isolated non-Devanagari characters (such as the 4 stray Kannada characters) are explicitly filtered out during preprocessing.

---

## 4. Technical and Linguistic Rationale

1. **Statutory & Legal Primacy**:
   Under the **Goa Official Language Act of 1987**, Konkani in the Devanagari script is the sole official language of the State of Goa.
2. **National Academy Standard**:
   India's National Academy of Letters (**Sahitya Akademi**) officially recognized Konkani as an independent literary language in 1975 under the Devanagari orthographic standard, and all national Sahitya Akademi awards for Konkani literature are conferred in Devanagari.
3. **Prevention of Vocabulary Inflation & Representation Sparsity**:
   If Kannada or Romi scripts were co-mingled in the same training corpus, the word and character vocabularies would immediately fracture across parallel orthographies without shared subwords. For a low-resource corpus (~31k sentences), splitting vocabulary space across multiple scripts would reduce token frequency by >50%, devastating embeddings and CRF transition matrices.
4. **Character-CNN & Attention Attribution Coherence**:
   In Phase 6 (Character-CNN encoder) and Phase 8 (Attention Attribution), grapheme embeddings rely on consistent spatial and phonological regularities (such as viramas, matras, and conjunct formation). Mixing Devanagari with Kannada or Latin scripts would destroy the spatial alignment required for explainable attention heatmaps over entity tokens.

---

## 5. Policy for Cross-Script Extensions

Should future research phases explore multi-script or transliteration scenarios (e.g., Devanagari-to-Romi transfer), such experiments will be handled strictly as distinct, isolated ablation benchmarks with explicit rule-based transliteration engines. The primary benchmark, gold seed annotations, and evaluation suites remain exclusively Devanagari Konkani.
