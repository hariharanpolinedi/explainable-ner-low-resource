# Error Taxonomy Cross-Tabulation Report: Konkani

- **Language**: Konkani
- **Total Diagnostic Errors**: 90
- **Evaluation Base**: Adjudicated Gold Diagnostic Pool (`dev.conll` + `long_multientity_eval_v2.conll` + `supplementary_long_multientity.conll`)

> [!NOTE]
> In compliance with project statistical reliability rules, any cell computed from fewer than 10 instances
> is explicitly labeled as `[Statistically Fragile (N < 10)]`.

---

## 1. Cross-Tabulation: Error Category vs. Sentence Length (in Tokens)

| Error Category | < 10 Tokens | 10–20 Tokens | 21–30 Tokens | > 30 Tokens | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 0 | 0 | 34 | **34** |
| **False Positive** | 0 | 0 | 0 | 1 *(N<10)* | **1** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 0 | 5 *(N<10)* | 2 *(N<10)* | 48 | **55** |

---

## 2. Cross-Tabulation: Error Category vs. Number of Gold Entities in Sentence

| Error Category | 0 Entities (FP) | 1 Entity | 2 Entities | 3+ Entities | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 0 | 5 *(N<10)* | 29 | **34** |
| **False Positive** | 0 | 0 | 1 *(N<10)* | 0 | **1** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 0 | 19 | 10 | 26 | **55** |

---

## 3. Cross-Tabulation: Error Category vs. Entity Type

| Error Category | PER | LOC | ORG | MISC | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 1 *(N<10)* | 30 | 0 | 3 *(N<10)* | **34** |
| **False Positive** | 0 | 1 *(N<10)* | 0 | 0 | **1** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 0 | **0** |
| **Overextended Span** | 0 | 0 | 0 | 0 | **0** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 6 *(N<10)* | 18 | 18 | 13 | **55** |

---

## 4. Cross-Tabulation: Error Category vs. Vocabulary Status (IV vs. OOV)

| Error Category | In-Vocabulary (IV) | Out-Of-Vocabulary (OOV) | Total | % OOV |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 18 | 16 | **34** | 47.1% |
| **False Positive** | 1 *(N<10)* | 0 | **1** | 0.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 0 | 0 | **0** | 0.0% |
| **Overextended Span** | 0 | 0 | **0** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 31 | 24 | **55** | 43.6% |

---

## 5. Cross-Tabulation: Error Category vs. Morphological Inflection Status

| Error Category | Inflected Form | Uninflected / Lemma Form | Total | % Inflected |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 5 *(N<10)* | 29 | **34** | 14.7% |
| **False Positive** | 0 | 1 *(N<10)* | **1** | 0.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 0 | 0 | **0** | 0.0% |
| **Overextended Span** | 0 | 0 | **0** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 16 | 39 | **55** | 29.1% |
