# Error Taxonomy Cross-Tabulation Report: Assamese

- **Language**: Assamese
- **Total Diagnostic Errors**: 86
- **Evaluation Base**: Adjudicated Gold Diagnostic Pool (`dev.conll` + `long_multientity_eval_v2.conll` + `supplementary_long_multientity.conll`)

> [!NOTE]
> In compliance with project statistical reliability rules, any cell computed from fewer than 10 instances
> is explicitly labeled as `[Statistically Fragile (N < 10)]`.

---

## 1. Cross-Tabulation: Error Category vs. Sentence Length (in Tokens)

| Error Category | < 10 Tokens | 10–20 Tokens | 21–30 Tokens | > 30 Tokens | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 0 | 1 *(N<10)* | 13 | **14** |
| **False Positive** | 0 | 1 *(N<10)* | 0 | 9 *(N<10)* | **10** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 4 *(N<10)* | **4** |
| **Overextended Span** | 0 | 0 | 0 | 2 *(N<10)* | **2** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 1 *(N<10)* | 1 *(N<10)* | 4 *(N<10)* | 50 | **56** |

---

## 2. Cross-Tabulation: Error Category vs. Number of Gold Entities in Sentence

| Error Category | 0 Entities (FP) | 1 Entity | 2 Entities | 3+ Entities | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 1 *(N<10)* | 2 *(N<10)* | 11 | **14** |
| **False Positive** | 0 | 2 *(N<10)* | 0 | 8 *(N<10)* | **10** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 0 | 0 | 4 *(N<10)* | **4** |
| **Overextended Span** | 0 | 0 | 0 | 2 *(N<10)* | **2** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 0 | 12 | 8 *(N<10)* | 36 | **56** |

---

## 3. Cross-Tabulation: Error Category vs. Entity Type

| Error Category | PER | LOC | ORG | MISC | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Missed Entity** | 0 | 6 *(N<10)* | 8 *(N<10)* | 0 | **14** |
| **False Positive** | 0 | 8 *(N<10)* | 0 | 2 *(N<10)* | **10** |
| **Wrong Entity Type** | 0 | 0 | 0 | 0 | **0** |
| **Incomplete Span** | 0 | 1 *(N<10)* | 3 *(N<10)* | 0 | **4** |
| **Overextended Span** | 0 | 2 *(N<10)* | 0 | 0 | **2** |
| **Bi Transition Error** | 0 | 0 | 0 | 0 | **0** |
| **O Inside Entity** | 0 | 0 | 0 | 0 | **0** |
| **Complete Sentence Collapse** | 7 *(N<10)* | 34 | 13 | 2 *(N<10)* | **56** |

---

## 4. Cross-Tabulation: Error Category vs. Vocabulary Status (IV vs. OOV)

| Error Category | In-Vocabulary (IV) | Out-Of-Vocabulary (OOV) | Total | % OOV |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 8 *(N<10)* | 6 *(N<10)* | **14** | 42.9% |
| **False Positive** | 4 *(N<10)* | 6 *(N<10)* | **10** | 60.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 2 *(N<10)* | 2 *(N<10)* | **4** | 50.0% |
| **Overextended Span** | 2 *(N<10)* | 0 | **2** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 35 | 21 | **56** | 37.5% |

---

## 5. Cross-Tabulation: Error Category vs. Morphological Inflection Status

| Error Category | Inflected Form | Uninflected / Lemma Form | Total | % Inflected |
| :--- | :---: | :---: | :---: | :---: |
| **Missed Entity** | 7 *(N<10)* | 7 *(N<10)* | **14** | 50.0% |
| **False Positive** | 1 *(N<10)* | 9 *(N<10)* | **10** | 10.0% |
| **Wrong Entity Type** | 0 | 0 | **0** | 0.0% |
| **Incomplete Span** | 3 *(N<10)* | 1 *(N<10)* | **4** | 75.0% |
| **Overextended Span** | 0 | 2 *(N<10)* | **2** | 0.0% |
| **Bi Transition Error** | 0 | 0 | **0** | 0.0% |
| **O Inside Entity** | 0 | 0 | **0** | 0.0% |
| **Complete Sentence Collapse** | 36 | 20 | **56** | 64.3% |
