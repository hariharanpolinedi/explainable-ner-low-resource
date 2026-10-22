# Error Taxonomy Cross-Tabulation Report: Sinhala

- **Language**: Sinhala
- **Total Diagnostic Errors**: 55
- **Evaluation Base**: Adjudicated Gold Diagnostic Pool (`dev.conll` + `long_multientity_eval_v2.conll` + `supplementary_long_multientity.conll`)

> [!NOTE]
> In compliance with project statistical reliability rules, any cell computed from fewer than 10 instances
> is explicitly labeled as `[Statistically Fragile (N < 10)]`.

---

## 1. Cross-Tabulation: Error Category vs. Sentence Length (in Tokens)

| Error Category | < 10 Tokens | 10–20 Tokens | 21–30 Tokens | > 30 Tokens | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 0 | 0 | 15 | **15** |
| **False Positive** | 0 | 1 *(N<10)* | 0 | 11 | **12** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 0 | 0 | 1 *(N<10)* | 27 | **28** |

---

## 2. Cross-Tabulation: Error Category vs. Number of Gold Entities in Sentence

| Error Category | 0 Entities (FP) | 1 Entity | 2 Entities | 3+ Entities | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 0 | 4 *(N<10)* | 11 | **15** |
| **False Positive** | 2 *(N<10)* | 1 *(N<10)* | 7 *(N<10)* | 2 *(N<10)* | **12** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 0 | 10 | 12 | 6 *(N<10)* | **28** |

---

## 3. Cross-Tabulation: Error Category vs. Entity Type

| Error Category | PER | LOC | ORG | MISC | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 9 *(N<10)* | 1 *(N<10)* | 5 *(N<10)* | **15** |
| **False Positive** | 0 | 10 | 2 *(N<10)* | 0 | **12** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 2 *(N<10)* | 15 | 5 *(N<10)* | 6 *(N<10)* | **28** |

---

## 4. Cross-Tabulation: Error Category vs. Vocabulary Status (IV vs. OOV)

| Error Category | In-Vocabulary (IV) | Out-Of-Vocabulary (OOV) | Total | % OOV |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 10 | 5 *(N<10)* | **15** | 33.3% |
| **False Positive** | 9 *(N<10)* | 3 *(N<10)* | **12** | 25.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 0 | 0 | **0** | 0.0% |
| **Overextended Span** | 0 | 0 | **0** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 26 | 2 *(N<10)* | **28** | 7.1% |

---

## 5. Cross-Tabulation: Error Category vs. Morphological Inflection Status

| Error Category | Inflected Form | Uninflected / Lemma Form | Total | % Inflected |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 4 *(N<10)* | 11 | **15** | 26.7% |
| **False Positive** | 0 | 12 | **12** | 0.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 0 | 0 | **0** | 0.0% |
| **Overextended Span** | 0 | 0 | **0** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 17 | 11 | **28** | 60.7% |
