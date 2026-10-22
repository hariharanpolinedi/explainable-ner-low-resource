# Phase 7: Original Failing Sentences Literal Re-Test

> **Phase 7 Deliverable — Literal Failure Mode Re-Test**  
> Direct, un-abstracted re-test of the exact real-world sentences that originally surfaced the failure modes during manual UI and edge-case testing.

---

## 1. Konkani Original Failing Sentence

**Sentence**: `"पणजींत जाल्ल्या एका विशेष परिषदेंत प्रमोद सावंत, श्रीपाद नाईक आनी दिगंबर कामत हांणी गोयाच्या आर्थिक विकासा खातीर..."`  
**Checkpoint Evaluated**: `checkpoints/improved_konkani_v2.pt` (Phase 0 Baseline preserved as Current Best)  
**Total Expected Entities**: 5

| Entity Mention | Entity Type | Role in Original Defect | Baseline Outcome | Current Best Outcome | Defect Resolved? |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **पणजींत** | `LOC` | Bound locative case-inflected place name | **MISSED (O)** | **MISSED (O)** | **NO** |
| **प्रमोद सावंत** | `PER` | First list item | **DETECTED (PER)** | **DETECTED (PER)** | YES |
| **श्रीपाद नाईक** | `PER` | Middle list item (dropped by CRF O-bias) | **MISSED (O)** | **MISSED (O)** | **NO** |
| **दिगंबर कामत** | `PER` | Last list item | **DETECTED (PER)** | **DETECTED (PER)** | YES |
| **गोयाच्या** | `LOC` | Bound genitive case-inflected place name | **MISSED (O)** | **MISSED (O)** | **NO** |

**Summary**: **2 / 5 entities detected (40.0%)**. The originally reported defect — missed inflected locations (`पणजींत`, `गोयाच्या`) and middle list suppression (`श्रीपाद नाईक`) — **remains unresolved** in the current best checkpoint.

---

## 2. Sinhala Original Failing Sentence

**Sentence (Raw User Input)**: `"මහාචාර්ය අනුර සිල්වා සහ ආචාර්ය නිමාලි ප්රනාන්දු විසින් පේරාදෙණිය විශ්වවිද්යාලයේදී සංවර්ධනය කරන ලද පද්ධතිය ශ්රී ලංකාවේ ග්රාමීය ප්රදේශවල ගොවීන්ට කාලගුණික වෙනස්කම් සහ බෝග රෝග හඳුනා ගැනීමට උපකාරී වේ ."`  
**Checkpoint Evaluated**: `checkpoints/improved_sinhala_v2.pt` (Phase 6 Combined Model)  
**Total Expected Entities**: 4

| Entity Mention | Entity Type | Role in Original Defect | Baseline Outcome | Current Best Outcome | Defect Resolved? |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **අනුර සිල්වා** | `PER` | Named professor | **MISSED (O)** | **DETECTED (PER)** | **YES** |
| **නිමාලි ප්‍රනාන්දු** | `PER` | Named doctor (with conjunct `ප්‍ර`) | **MISSED (O)** | **MISSED (O)** | **NO** |
| **පේරාදෙණිය විශ්වවිද්‍යාලය** | `ORG` | University name (with conjunct `ද්‍ය`) | **MISSED (O)** | **MISSED (O)** | **NO** |
| **ශ්‍රී ලංකාව** | `LOC` | Country name (with conjunct `ශ්‍රී`) | **MISSED (O)** | **MISSED (O)** | **NO** |

**Summary**: **1 / 4 entities detected (25.0%)**. The baseline suffered complete sentence collapse (0 / 4, 100% missed). The Phase 6 combined model (`improved_sinhala_v2.pt`) successfully breaks the complete collapse by detecting the lead person entity (`මහාචාර්ය අනුර`), but downstream long-sequence suppression still prevents recovery of later entities in this 29-token sequence.

---

## 3. Assamese Long Multi-Entity Sentence (#200)

**Sentence**: Long multi-entity sentence with 42 tokens and 8 entities from `long_multientity_eval_v2.conll`.  
**Checkpoint Evaluated**: `checkpoints/improved_assamese_v2.pt` (Phase 0 Baseline preserved as Current Best)  
**Total Gold Entities**: 8

| Entity Mention | Gold Type | Baseline Prediction | Current Best Prediction | Defect Status |
| :--- | :---: | :---: | :---: | :---: |
| **ভাৰতৰ** | `LOC` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **ভাৰতীয়** | `LOC` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **বেংক** | `ORG` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **ভাৰতীয়** | `LOC` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **বেংক** | `ORG` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **ভাৰতীয়** | `LOC` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **নিগম** | `ORG` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |
| **মুম্বাইত** | `LOC` | `O` (Collapsed) | `O` (Collapsed) | **STILL COLLAPSED** |

**Summary**: **0 / 8 entities detected (0.0%)**. Sentence-level collapse on dense multi-entity long sentences **remains unresolved** in Assamese.
