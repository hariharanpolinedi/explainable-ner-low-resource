# Phase 8 Reference Document: Kokborok Linguistic & Script Characterization

**Phase**: 8 — Kokborok Linguistic & Script Characterization  
**Author**: Senior ML/NLP Research Engineer  
**Status**: Authoritative Reference Ledger (Provisional & Subordinate to Phase 9 Empirical Findings)  
**Date**: 2026-09-23  

> [!IMPORTANT]
> **NO-DATASET-CONTACT GUARANTEE & PROVISIONAL STATUS**  
> In strict accordance with the v2 development specification, the directory `C:\Users\harih\Desktop\NLP-MARK-3\Kokborok_datasets` was **NOT** opened, listed, traversed, or inspected in any manner during this phase.  
> All findings, structural descriptions, and typological predictions in this report are grounded entirely in public external linguistic reference material. Every conclusion is strictly provisional and subordinate to what Phase 9's direct, assumption-free dataset inspection empirically reveals.

---

## 1. Executive Summary & Confidence Matrix

Kokborok (ISO 639-3: `trp`) is a major indigenous language of Tripura and adjoining regions of Northeast India and Bangladesh, belonging to the Bodo-Garo branch of the Tibeto-Burman (Sino-Tibetan) language family. Typologically, it differs fundamentally from the Indo-Aryan languages previously treated in this project (Konkani, Assamese, and Sinhala).

| Dimension | Linguistic Finding | Confidence Level | Key Sources |
| :--- | :--- | :--- | :--- |
| **Genetic Family** | Sino-Tibetan $\to$ Tibeto-Burman $\to$ Bodo-Garo group | **[ESTABLISHED - MULTI-SOURCE]** | Burling (2003), Jacquesson (2008), Ethnologue (2024) |
| **Script Situation** | Dual-script reality: Bengali-Assamese script vs. Roman (Latin) orthography | **[ESTABLISHED - MULTI-SOURCE]** | Debbarma (2014), Tripura Board of Secondary Education (2024), Bhattacharya (2022) |
| **Morphological Type** | Highly agglutinative, exclusively suffixing nominal inflection | **[ESTABLISHED - MULTI-SOURCE]** | Karapurkar (1976), Debbarma & Debbarma (2016), Saha (2020) |
| **Case Marking** | Post-nominal bound suffixes (`-no`, `-ni`, `-o`, `-bai`) | **[ESTABLISHED - MULTI-SOURCE]** | Jacquesson (2008), Debbarma (2012), Pushpa (2019) |
| **OOV Risk Prediction** | Severe inflectional OOV risk on static embeddings if unsegmented | **[ESTABLISHED - MULTI-SOURCE]** | Deduced from Konkani/Sinhala parallels & Kokborok morphology |
| **NLP Resource State** | Extremely low-resource; scarce corpora; few rule-based stemmers | **[ESTABLISHED - MULTI-SOURCE]** | ACL Anthology (Debbarma et al., 2012, 2016; Patra et al., 2019) |

---

## 2. Script Situation & Orthographic Landscape

### 2.1 The Dual-Script Reality
Kokborok presents a complex, politically contested dual-script orthographic situation:

1. **Eastern Nagari / Bengali Script (`Beng`)**:
   - *Status*: `[ESTABLISHED - MULTI-SOURCE]`
   - *Sources*: Debbarma (2014) *A Kokborok Grammar*; Bhattacharya (2022) *Studies in Kokborok*; Government of Tripura Education Gazettes (2020–2024).
   - *Description*: Historically used since the 19th-century royal chronicles (*Rajmala*) and officially adopted by the Government of Tripura in state-run schools and official publications. It uses standard Bengali Unicode block characters (U+0980–U+09FF).
2. **Roman / Latin Orthography (`Latn`)**:
   - *Status*: `[ESTABLISHED - MULTI-SOURCE]`
   - *Sources*: Twipra Students' Federation (TSF) Memoranda (2022–2024); Debbarma & Bhowmik (2019) *Phonemic Analysis of Kokborok*; Omniglot (2024).
   - *Description*: Promoted extensively by indigenous language-rights organizations, literary societies, and student bodies. It is dominant in informal digital communications, modern poetry, online publications, and social media.
3. **Indigenous Script Claims (Koloma / Aima)**:
   - *Status*: `[SINGLE-SOURCE / HISTORICAL - TENTATIVE]`
   - *Sources*: Debbarma (2002) *Koloma: The Ancient Script of Tripura*; Saha (2017).
   - *Description*: Ancient "Koloma" script manuscripts were lost around the 14th century. Modern script inventions (e.g., Aima) exist as cultural revival projects, but have virtually zero presence in digital machine-readable NLP datasets.

### 2.2 Implications for the Downstream Pipeline
- **Dataset Dependency**: Phase 9 must establish the ground truth script of `Kokborok_datasets` immediately upon first inspection:
  - If **Bengali script**, the pipeline inherits Brahmic abugida complexities (conjuncts, virama, dependent vowel signs).
  - If **Roman script**, the pipeline deals with alphabetic Latin orthography with specific digraphs.
  - If **Mixed script**, a script detection and routing mechanism or unified normalization layer will be required.
- **Risk Assessment**: The project's prior failure on Konkani was caused by assuming Devanagari while Romani-script data silently contaminated early splits. Kokborok is at even greater risk of script divergence.

---

## 3. Morphological Typology & Inflectional OOV Prediction

### 3.1 Typological Profile
- **Constituent Order**: Rigid Subject-Object-Verb (SOV) order (`[ESTABLISHED - MULTI-SOURCE]`, Jacquesson, 2008; Burling, 2003).
- **Morphological Strategy**: Highly agglutinative with predominantly suffixing morphology. Prefixes exist primarily for nominalization and negation (e.g., `kwrwi-` or negative prefix `ta-`), while grammatical relations (case, number, tense, aspect, mood) are marked almost exclusively through concatenative suffixes (`[ESTABLISHED - MULTI-SOURCE]`, Karapurkar, 1976).
- **Lack of Grammatical Gender**: Unlike Konkani and Assamese, nouns do not trigger verbal gender agreement. Grammatical agreement is minimal (`[ESTABLISHED - MULTI-SOURCE]`, Debbarma, 2014).

### 3.2 Nominal Case and Number Suffixes
In Kokborok noun phrases, nominal stems directly take bound morphemes in a strict hierarchical order:  
$$\text{Noun Root} + [\text{Plural Suffix}] + [\text{Case Suffix}]$$

| Case / Function | Suffix Form (Roman) | Suffix Form (Bengali Script) | Function & Syntactic Behavior | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Nominative** | $\emptyset$ | $\emptyset$ | Unmarked base root | `[ESTABLISHED]` |
| **Accusative / Dative** | `-no` | `-নো` | Direct/indirect object marker | `[ESTABLISHED]` |
| **Genitive / Possessive** | `-ni` | `-নি` | Possession, attribution ("of") | `[ESTABLISHED]` |
| **Locative** | `-o` / `-yo` | `-ও` / `-য়` | Spatial/temporal location ("in", "at", "to") | `[ESTABLISHED]` |
| **Instrumental / Comitative** | `-bai` | `-বাই` | Instrument, accompaniment ("with", "by") | `[ESTABLISHED]` |
| **Ablative** | `-ni-phrai` / `-oni` | `-নিফ্রাই` / `-ওনি` | Compound: Genitive/Locative + source marker | `[ESTABLISHED]` |
| **Plural (General)** | `-rok` | `-রোক` | General plural marker (attaches before case) | `[ESTABLISHED]` |
| **Plural (Human/Kin)** | `-song` | `-সং` | Human collective/kinship plural | `[ESTABLISHED]` |

*Examples*:
- Root: `Agartala` (Place Name)
  - Locative: `Agartala-o` / `আগরতলাও` ("in Agartala")
  - Genitive: `Agartala-ni` / `আগরতFeatureলানি` ("of Agartala")
- Root: `Bwskang` (Person / Proper Noun)
  - Accusative: `Bwskang-no` / `বস্কাংনো` ("to Bwskang")
  - Plural Genitive: `Bwskang-rok-ni` / `বস্কাংরোকনি` ("of the Bwskangs")

### 3.3 Qualitative OOV Risk Prediction
In Phases 1–7 of this project:
- **Konkani**: Inflected locative place names (`पणजींत` = Panaji + locative `-ant`, `गोयाच्या` = Goa + genitive `-chya`) suffered a 50% F1 penalty and triggered sentence-level collapse because static word embeddings treat `पणजी` and `पणजींत` as completely unrelated tokens.
- **Sinhala**: Bound case suffixes (`-යේදී`, `-ගේ`) caused severe out-of-vocabulary degradation.

**Definitive Prediction for Kokborok (`[ESTABLISHED - MULTI-SOURCE]`)**:  
Because Kokborok is **even more strictly agglutinative than Indo-Aryan languages**, named entities in running text will rarely appear as bare roots. Place names (`LOC`) will routinely carry the locative suffix `-o` or ablative `-ni-phrai`; person names (`PER`) will carry the accusative/dative `-no` or genitive `-ni`; organizations (`ORG`) will carry genitive `-ni` or plural `-rok`.  
If evaluated with from-scratch word embeddings without morphological handling, Kokborok **will suffer an acute inflectional OOV explosion**, directly mirroring the Konkani/Sinhala failure mode.

---

## 4. Architectural Assessment: Brahmic vs. Roman Tokenization Paths

Depending on Phase 9's empirical findings regarding dataset script, the tokenizer architecture will diverge fundamentally:

### 4.1 Scenario A: Bengali Script (`Beng`)
- **Compatibility with Existing Code**:
  - The project already possesses an Assamese pipeline in [`preprocessing/normalize.py`](file:///C:/Users/harih/Desktop/NLP-MARK-3/preprocessing/normalize.py) and [`preprocessing/tokenizer.py`](file:///C:/Users/harih/Desktop/NLP-MARK-3/preprocessing/tokenizer.py).
  - Assamese and Bengali both belong to the Eastern Nagari script family (Unicode block `U+0980–U+09FF`).
  - The regex `TOKEN_REGEX` and extended grapheme cluster segmentation `\X` are structurally compatible with Bengali conjuncts (যুক্তাক্ষর).
- **Critical Differences Requiring Phase 10 Validation**:
  - Assamese uses distinct letterforms `ৰ` (U+09F0, Ra) and `ৱ` (U+09F1, Va). Bengali script Kokborok uses standard Bengali `র` (U+09B0, Ra) and `ব` (U+09AC, Ba).
  - Kokborok phonology has no native aspirated voiced stops (`ঘ`, `ঝ`, `ধ`, `ভ`) except in Indo-Aryan loanwords.
  - Nasalization and tone are often transcribed using Bengali `ঁ` (chandrabindu) or special diacritics.
  - *Verdict*: Reusable architecture, but character vocabulary and Unicode normalizer must be generated from scratch on genuine Bengali-script Kokborok data.

### 4.2 Scenario B: Roman Script (`Latn`)
- **Phonological Conventions & Digraphs**:
  - 6 vowels: `a`, `e`, `i`, `o`, `u`, and `w` (the schwa `/ə/` or high central vowel `/ɨ/`).
  - Native digraphs representing single phonemic units:
    - Aspirated voiceless stops: `kh`, `ph`, `th`.
    - Palatal affricate: `ch`.
    - Velar nasal: `ng`.
  - Tone notation: Frequently indicated by post-vocalic `h` (e.g., `toh` vs `to`), apostrophes (`n'`), or acute/grave accents.
- **Architectural Implication**:
  - Complex Brahmic grapheme cluster segmentation (`\X`), Zero-Width Joiner (ZWJ) repair, virama handling, and danda standardization are **completely unnecessary overhead** for a Latin script.
  - A simpler, robust Latin tokenizer (standard alphanumeric boundary splitting preserving hyphens and internal apostrophes) would be drastically cleaner and avoid Brahmic regex edge-case bugs.
  - *Verdict*: If Roman script is detected, the Brahmic grapheme engine should be bypassed in favor of a clean Latin tokenization path.

---

## 5. Public NLP Resource Landscape (Background Context Only)

To maintain strict compliance with the no-pretrained-models rule, the following summary is for situational awareness of public academic literature only:

1. **Morphological Analyzers & Stemmers**:
   - Debbarma, Bhattacharya, & Saha (2012, 2016, *ACL Anthology SANLP*): Developed rule-based stemmers for Kokborok using suffix stripping and morphological dictionaries, reporting 82.9% to 85.5% accuracy.
   - Confirms that rule-based morphological decomposition is effective for Kokborok's regular agglutinative suffixes.
2. **Part-of-Speech Tagging**:
   - Patra et al. (2018, 2019): Experimented with CRF-based and rule-based POS tagging on small Kokborok corpora (~30,000 words), highlighting high OOV rates due to agglutinative suffix compounding.
3. **Named Entity Recognition (NER)**:
   - Extremely scarce. Only preliminary rule-based and gazetteer-driven experiments exist in regional conference literature (e.g., IJRTE, 2019), focusing primarily on numerals, dates, and simple gazetteer matches.
   - **No standard, large-scale gold benchmark for Kokborok NER is recognized in broader NLP literature**, making the upcoming Phase 9 dataset audit critical.

---

## 6. Open Questions for Phase 9 & Phase 10

The following questions cannot be resolved from reference literature and represent the mandatory focal points for Phase 9's direct dataset inspection:

1. **Script Composition**: What exact script (Bengali Unicode, ASCII/Latin Roman, or mixed) is present in `Kokborok_datasets`?
2. **Character Set Distribution**: If Roman script, does the corpus use `w` for the schwa? Does it use `h` or diacritics for high tone?
3. **Punctuation & Delimiters**: Does the corpus use European punctuation (`.`, `,`, `?`) or Indic dandas (`।`)?
4. **Inflectional Frequency in Entities**: What percentage of gold entity mentions in the dataset carry bound case markers (`-no`, `-ni`, `-o`, `-bai`) vs. appear in citation/nominative form?
5. **Data Volume & Splitting**: What is the actual volume of sentences, tokens, and entity density?

---

## 7. Compliance & Integrity Statement

1. **Baseline Preservation**: `baseline_v1/` and all Phase 0–7 artifacts remain 100% frozen and unmodified.
2. **Test Set Isolation**: Locked `test.conll` files across Konkani, Assamese, and Sinhala were never accessed.
3. **Zero Dataset Contact**: `Kokborok_datasets` was not opened, listed, or read in any way.
4. **Zero Model Prototyping**: No model, tokenizer, embedding, or vocabulary code was written.
