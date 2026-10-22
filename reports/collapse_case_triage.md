# Collapse-Case Root-Cause Triage Ledger

> **Phase 2 Audit Trail**  
> Documents every complete-sentence collapse case observed across the diagnostic benchmark pool,
> categorized by the 3-tier cascade: (1) OOV-driven, (2) Orthography-driven, (3) Attention-degeneracy, or (4) Unexplained.

---

Total Triaged Collapses Across All Languages: **72**

## Konkani Collapse Cases (N=30)

| Sentence ID | Source | Tokens | Gold Entities | Triage Category | Concrete Supporting Evidence |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `#28` | `dev` | 15 | 1 (PER) | **Attention-Degeneracy Collapse** | Uniform attention degeneracy (mean entropy ratio = 0.997 > 0.95) |
| `#35` | `dev` | 11 | 1 (PER) | **OOV-Driven Collapse** | Sentence OOV rate: 36.4% (4/11), Entity OOV rate: 0.0% (0/2) |
| `#41` | `dev` | 13 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 46.2% (6/13), Entity OOV rate: 100.0% (1/1) |
| `#49` | `dev` | 15 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 13.3% (2/15), Entity OOV rate: 100.0% (1/1) |
| `#56` | `dev` | 19 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 52.6% (10/19), Entity OOV rate: 0.0% (0/3) |
| `#200` | `long_v2` | 36 | 4 (ORG, ORG, LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 30.6% (11/36), Entity OOV rate: 25.0% (1/4) |
| `#201` | `long_v2` | 41 | 4 (PER, ORG, PER, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 58.5% (24/41), Entity OOV rate: 71.4% (5/7) |
| `#204` | `long_v2` | 33 | 1 (PER) | **OOV-Driven Collapse** | Sentence OOV rate: 42.4% (14/33), Entity OOV rate: 0.0% (0/1) |
| `#205` | `long_v2` | 46 | 1 (MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 41.3% (19/46), Entity OOV rate: 0.0% (0/1) |
| `#207` | `long_v2` | 31 | 1 (MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 35.5% (11/31), Entity OOV rate: 0.0% (0/1) |
| `#208` | `long_v2` | 43 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 39.5% (17/43), Entity OOV rate: 100.0% (1/1) |
| `#209` | `long_v2` | 40 | 2 (LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 40.0% (16/40), Entity OOV rate: 0.0% (0/2) |
| `#210` | `long_v2` | 37 | 2 (LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 45.9% (17/37), Entity OOV rate: 50.0% (1/2) |
| `#213` | `long_v2` | 37 | 3 (LOC, ORG, PER) | **OOV-Driven Collapse** | Sentence OOV rate: 40.5% (15/37), Entity OOV rate: 33.3% (1/3) |
| `#214` | `long_v2` | 50 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 66.0% (33/50), Entity OOV rate: 100.0% (1/1) |
| `#215` | `long_v2` | 34 | 4 (LOC, LOC, LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 44.1% (15/34), Entity OOV rate: 50.0% (2/4) |
| `#219` | `long_v2` | 40 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 47.5% (19/40), Entity OOV rate: 0.0% (0/1) |
| `#220` | `long_v2` | 38 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 50.0% (19/38), Entity OOV rate: 100.0% (1/1) |
| `#221` | `long_v2` | 30 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 43.3% (13/30), Entity OOV rate: 0.0% (0/1) |
| `#222` | `long_v2` | 46 | 1 (MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 37.0% (17/46), Entity OOV rate: 100.0% (1/1) |
| `#223` | `long_v2` | 33 | 5 (LOC, LOC, ORG, LOC, MISC) | **Attention-Degeneracy Collapse** | Uniform attention degeneracy (mean entropy ratio = 1.000 > 0.95) |
| `#224` | `long_v2` | 37 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 37.8% (14/37), Entity OOV rate: 100.0% (1/1) |
| `#226` | `long_v2` | 31 | 2 (ORG, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 48.4% (15/31), Entity OOV rate: 50.0% (1/2) |
| `#227` | `long_v2` | 41 | 6 (ORG, ORG, ORG, MISC, ORG, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 34.1% (14/41), Entity OOV rate: 16.7% (1/6) |
| `#229` | `long_v2` | 45 | 1 (MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 62.2% (28/45), Entity OOV rate: 100.0% (1/1) |
| `#230` | `long_v2` | 33 | 2 (MISC, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 45.5% (15/33), Entity OOV rate: 100.0% (2/2) |
| `#231` | `long_v2` | 32 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 53.1% (17/32), Entity OOV rate: 100.0% (1/1) |
| `#232` | `long_v2` | 30 | 1 (MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 56.7% (17/30), Entity OOV rate: 0.0% (0/1) |
| `#233` | `long_v2` | 68 | 2 (MISC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 54.4% (37/68), Entity OOV rate: 50.0% (1/2) |
| `#234` | `long_v2` | 36 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 41.7% (15/36), Entity OOV rate: 0.0% (0/1) |

---

## Assamese Collapse Cases (N=24)

| Sentence ID | Source | Tokens | Gold Entities | Triage Category | Concrete Supporting Evidence |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `#14` | `dev` | 11 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 36.4% (4/11), Entity OOV rate: 50.0% (1/2) |
| `#125` | `dev` | 9 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 44.4% (4/9), Entity OOV rate: 100.0% (1/1) |
| `#200` | `long_v2` | 42 | 8 (LOC, LOC, ORG, ORG, LOC, LOC, ORG, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 52.4% (22/42), Entity OOV rate: 37.5% (3/8) |
| `#201` | `long_v2` | 42 | 4 (ORG, LOC, ORG, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 28.6% (12/42), Entity OOV rate: 50.0% (2/4) |
| `#202` | `long_v2` | 40 | 5 (LOC, LOC, ORG, ORG, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 50.0% (20/40), Entity OOV rate: 66.7% (10/15) |
| `#203` | `long_v2` | 31 | 2 (LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 35.5% (11/31), Entity OOV rate: 50.0% (1/2) |
| `#206` | `long_v2` | 38 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 36.8% (14/38), Entity OOV rate: 0.0% (0/1) |
| `#207` | `long_v2` | 41 | 1 (LOC) | **Orthography-Driven Collapse** | Assamese Ra-phala ligature sequence in 'প্ৰচলিত' |
| `#209` | `long_v2` | 33 | 2 (LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 30.3% (10/33), Entity OOV rate: 0.0% (0/2) |
| `#211` | `long_v2` | 32 | 1 (ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 31.2% (10/32), Entity OOV rate: 100.0% (1/1) |
| `#212` | `long_v2` | 30 | 3 (ORG, ORG, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 26.7% (8/30), Entity OOV rate: 0.0% (0/3) |
| `#213` | `long_v2` | 31 | 2 (LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 25.8% (8/31), Entity OOV rate: 0.0% (0/2) |
| `#217` | `long_v2` | 39 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 30.8% (12/39), Entity OOV rate: 0.0% (0/1) |
| `#218` | `long_v2` | 36 | 1 (LOC) | **Orthography-Driven Collapse** | Assamese Ra-phala ligature sequence in 'প্ৰথম' |
| `#220` | `long_v2` | 53 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 35.8% (19/53), Entity OOV rate: 100.0% (1/1) |
| `#221` | `long_v2` | 31 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 38.7% (12/31), Entity OOV rate: 100.0% (1/1) |
| `#222` | `long_v2` | 41 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 53.7% (22/41), Entity OOV rate: 0.0% (0/1) |
| `#223` | `long_v2` | 43 | 3 (LOC, LOC, LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 37.2% (16/43), Entity OOV rate: 100.0% (3/3) |
| `#224` | `long_v2` | 34 | 6 (LOC, LOC, LOC, LOC, LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 50.0% (17/34), Entity OOV rate: 33.3% (2/6) |
| `#225` | `long_v2` | 43 | 2 (MISC, PER) | **Orthography-Driven Collapse** | Assamese Ra-phala ligature sequence in 'চলচ্চিত্ৰ' |
| `#227` | `long_v2` | 36 | 1 (PER) | **OOV-Driven Collapse** | Sentence OOV rate: 33.3% (12/36), Entity OOV rate: 0.0% (0/2) |
| `#228` | `long_v2` | 30 | 1 (PER) | **Orthography-Driven Collapse** | Assamese Ra-phala ligature sequence in 'চলচ্চিত্ৰ' |
| `#230` | `long_v2` | 66 | 4 (PER, PER, PER, PER) | **OOV-Driven Collapse** | Sentence OOV rate: 39.4% (26/66), Entity OOV rate: 0.0% (0/4) |
| `#232` | `long_v2` | 47 | 3 (LOC, LOC, LOC) | **Orthography-Driven Collapse** | Assamese Ra-phala ligature sequence in 'প্ৰতীক' |

---

## Sinhala Collapse Cases (N=18)

| Sentence ID | Source | Tokens | Gold Entities | Triage Category | Concrete Supporting Evidence |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `#200` | `long_v2` | 31 | 2 (MISC, LOC) | **Orthography-Driven Collapse** | Missing ZWJ in Sinhala conjunct: 'බෞද්ධ' contains \u0DCA followed immediately by consonant 'ධ' without \u200D |
| `#203` | `long_v2` | 31 | 2 (ORG, PER) | **Orthography-Driven Collapse** | Missing ZWJ in Sinhala conjunct: 'රාජපක්ෂ' contains \u0DCA followed immediately by consonant 'ෂ' without \u200D |
| `#207` | `long_v2` | 36 | 2 (LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 38.9% (14/36), Entity OOV rate: 0.0% (0/3) |
| `#208` | `long_v2` | 45 | 3 (MISC, LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 33.3% (15/45), Entity OOV rate: 33.3% (1/3) |
| `#209` | `long_v2` | 40 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 25.0% (10/40), Entity OOV rate: 0.0% (0/1) |
| `#210` | `long_v2` | 39 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 25.6% (10/39), Entity OOV rate: 0.0% (0/1) |
| `#216` | `long_v2` | 63 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 38.1% (24/63), Entity OOV rate: 0.0% (0/1) |
| `#217` | `long_v2` | 31 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 48.4% (15/31), Entity OOV rate: 0.0% (0/2) |
| `#218` | `long_v2` | 32 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 56.2% (18/32), Entity OOV rate: 0.0% (0/1) |
| `#219` | `long_v2` | 37 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 29.7% (11/37), Entity OOV rate: 0.0% (0/1) |
| `#222` | `long_v2` | 45 | 1 (LOC) | **Orthography-Driven Collapse** | Missing ZWJ in Sinhala conjunct: 'දුෂ්කර' contains \u0DCA followed immediately by consonant 'ක' without \u200D |
| `#223` | `long_v2` | 48 | 2 (ORG, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 41.7% (20/48), Entity OOV rate: 0.0% (0/2) |
| `#224` | `long_v2` | 39 | 2 (LOC, MISC) | **OOV-Driven Collapse** | Sentence OOV rate: 43.6% (17/39), Entity OOV rate: 0.0% (0/3) |
| `#225` | `long_v2` | 67 | 3 (MISC, PER, ORG) | **OOV-Driven Collapse** | Sentence OOV rate: 40.3% (27/67), Entity OOV rate: 33.3% (1/3) |
| `#228` | `long_v2` | 38 | 1 (ORG) | **Orthography-Driven Collapse** | Missing ZWJ in Sinhala conjunct: 'මානවශාස්ත්‍ර' contains \u0DCA followed immediately by consonant 'ත' without \u200D |
| `#229` | `long_v2` | 41 | 2 (LOC, LOC) | **Orthography-Driven Collapse** | Missing ZWJ in Sinhala conjunct: 'ආරම්භ' contains \u0DCA followed immediately by consonant 'භ' without \u200D |
| `#230` | `long_v2` | 30 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 43.3% (13/30), Entity OOV rate: 0.0% (0/2) |
| `#233` | `long_v2` | 69 | 1 (LOC) | **OOV-Driven Collapse** | Sentence OOV rate: 30.4% (21/69), Entity OOV rate: 0.0% (0/2) |

---
