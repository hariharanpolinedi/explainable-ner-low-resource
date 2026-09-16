# Named Entity Recognition Tagset Specification (BIO Scheme)

**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Standard**: CoNLL-2003 BIO (Begin, Inside, Outside) Scheme  
**Scope**: Konkani, Assamese, Sinhala  
**Authority**: Single Source of Truth for all annotation, weak supervision, modeling, and evaluation phases.

---

## 1. Schema Overview

Every word token is assigned exactly one label from the following 9-class tagset:

| Tag | Entity Class | Description | Canonical Examples |
| :--- | :--- | :--- | :--- |
| **`B-PER`** | Person (Begin) | First token of a proper person name | Konkani: `दामोदर`, Assamese: `লক্ষ্মীনাথ`, Sinhala: `මාර්ටින්` |
| **`I-PER`** | Person (Inside) | Subsequent tokens of a multi-token person name | Konkani: `मावजो`, Assamese: `বেজবৰুৱা`, Sinhala: `වික්‍රමසිංහ` |
| **`B-LOC`** | Location (Begin) | First token of a geopolitical, topographical, or infrastructural location | Konkani: `पणजी`, Assamese: `গুৱাহাটী`, Sinhala: `කොළඹ` |
| **`I-LOC`** | Location (Inside) | Subsequent tokens of a multi-token location | Konkani: `गोंय` (in `दक्षिण गोंय`), Assamese: `উদ্যান` (in `কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যান`), Sinhala: `නගරය` |
| **`B-ORG`** | Organization (Begin) | First token of a named institution, company, governmental body, or political party | Konkani: `गोंय`, Assamese: `অসম`, Sinhala: `ශ්‍රී` |
| **`I-ORG`** | Organization (Inside) | Subsequent tokens of a multi-token organization | Konkani: `विद्यापीठ`, Assamese: `সাহিত্য সভা`, Sinhala: `ලංකා පාර්ලිමේන්තුව` |
| **`B-MISC`** | Miscellaneous (Begin) | First token of an entity with a proper name that is not PER, LOC, or ORG (festivals, treaties, historical eras, constitutional laws) | Konkani: `शिगमो`, Assamese: `বিহু`, Sinhala: `මහාවංශය` |
| **`I-MISC`** | Miscellaneous (Inside) | Subsequent tokens of a multi-token miscellaneous entity | Assamese: `ৰঙালী বিহু`, Konkani: `मुक्ती दीस`, Sinhala: `අලුත් අවුරුද්ද` |
| **`O`** | Outside | Non-entity tokens (common nouns, verbs, adjectives, punctuation, numbers) | `आसा`, `নহয়`, `සහ`, `.`, `।` |

---

## 2. BIO Transition Invariants

A sequence of labels is structurally valid if and only if:
1. An `I-TYPE` tag is **always preceded** immediately by either `B-TYPE` or `I-TYPE` of the exact same entity class.
2. An `I-TYPE` tag must **never** directly follow `O` or an `I/B-` tag of a different entity class.
3. Every new entity mention must begin with a `B-TYPE` tag, even if it is a single-token mention.

Any transition violating these invariants (e.g., `O -> I-PER` or `B-LOC -> I-ORG`) is invalid and will fail the automated validation suite.
