# Kokborok Named Entity Recognition Tagset Specification (BIO Scheme)

**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Language**: Kokborok (Tripuri / `trp`)  
**Script**: Roman (Latin) Orthography  
**Standard**: CoNLL-2003 BIO (Begin, Inside, Outside) Scheme  
**Authority**: Single Source of Truth for Kokborok annotation, weak supervision, modeling, and evaluation.

---

## 1. Schema Overview

In strict alignment with the established project-wide 9-class BIO standard, every word token in Kokborok text is assigned exactly one label:

| Tag | Entity Class | Description | Canonical Kokborok Examples |
| :--- | :--- | :--- | :--- |
| **`B-PER`** | Person (Begin) | First token of a proper person name | `Mohan`, `Kanika`, `Bir`, `Radhakishore`, `Bwskang` |
| **`I-PER`** | Person (Inside) | Subsequent tokens of a multi-token person name | `Bikram` (in `Bir Bikram`), `Debbarma`, `Manikya` |
| **`B-LOC`** | Location (Begin) | First token of a geopolitical, topographical, or infrastructural location | `Agartala`, `Tripura`, `Kailashahar-o`, `Udaipur`, `Bilonia-o` |
| **`I-LOC`** | Location (Inside) | Subsequent tokens of a multi-token location | `Station-o` (in `Agartala Station-o`), `Hill`, `Twima` |
| **`B-ORG`** | Organization (Begin) | First token of an institution, company, governmental body, or union | `Tata`, `Tripura`, `Twipra`, `Kokborok`, `State` |
| **`I-ORG`** | Organization (Inside) | Subsequent tokens of a multi-token organization | `Technologies` (in `Tata Technologies`), `University`, `Mission` |
| **`B-MISC`** | Miscellaneous (Begin) | First token of named entities not PER, LOC, or ORG (festivals, treaties, languages) | `Kharchi`, `Garia`, `Ker`, `MoA`, `Kokborok` (as language entity) |
| **`I-MISC`** | Miscellaneous (Inside) | Subsequent tokens of a multi-token miscellaneous entity | `Puja` (in `Kharchi Puja`), `Sal` (in `Kokborok Sal`), `Day` |
| **`O`** | Outside | Non-entity tokens (common nouns, verbs, particles, punctuation, pronouns) | `kha`, `tong`, `ang`, `bai`, `kaham`, `.`, `,`, `!` |

---

## 2. Kokborok-Specific Morphological Invariants

Kokborok is an agglutinative language where nominal stems take bound case and plural suffixes:
1. **Attached Bound Suffixes**:
   - Locative `-o` / `-yo` (e.g., `Tripurao` = `Tripura` + in): The entire inflected token is tagged `B-LOC` because word tokenization treats it as a single word unit.
   - Accusative/Dative `-no` (e.g., `Mohan-no` or `Mohanno`): Tagged `B-PER`.
   - Plural `-rok` (e.g., `Bangladeshirok`): Tagged `B-MISC` (or nationality demonym).
2. **Hyphenated Bound Suffixes**:
   - In loanwords with hyphenated suffixes (e.g., `Agartala Station-o`):
     - `Agartala` $\to$ `B-LOC`
     - `Station-o` $\to$ `I-LOC`
3. **Punctuation Isolation**:
   - All punctuation marks (`,`, `.`, `!`, `?`, `"`, `(`, `)`) are strictly isolated tokens and must **always** be tagged `O`.

---

## 3. BIO Transition Invariants

A sequence of labels is structurally valid if and only if:
1. An `I-TYPE` tag is **always preceded** immediately by either `B-TYPE` or `I-TYPE` of the exact same entity class.
2. An `I-TYPE` tag must **never** directly follow `O` or an `I/B-` tag of a different entity class.
3. Every new entity mention must begin with a `B-TYPE` tag, even if it is a single-token mention.
