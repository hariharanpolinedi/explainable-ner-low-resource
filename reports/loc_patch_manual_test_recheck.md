# Manual Test-Case Re-Check: Konkani Location Inflection Verification

**Date**: 2026-09-17  
**Auditor**: Senior ML/NLP Engineer  
**Test Case**: Authentic Konkani Government Press Release Sentence  

```text
पणजींत जाल्ल्या एका विशेष परिषदेंत मुख्यमंत्री डॉ. प्रमोद सावंत आनी केंद्रीय मंत्री श्रीपाद नाईक हांणी गोयाच्या आर्थिक विकासा खातीर नव्या प्रकल्पांची घोशणा केली.
```

---

## 1. Ground Truth Annotation vs Model Predictions

| Token Index | Token | Ground Truth Tag | Baseline (`selftrained_best`) | LOC-Patched (`loc_patched`) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **पणजींत** | **B-LOC** | `O` | **`O`** | **MISSED** |
| 1 | जाल्ल्या | O | `O` | `O` | MATCH |
| 2 | एका | O | `O` | `O` | MATCH |
| 3 | विशेष | O | `O` | `O` | MATCH |
| 4 | परिषदेंत | O | `O` | `O` | MATCH |
| 5 | मुख्यमंत्री | O | `O` | `O` | MATCH |
| 6 | डॉ. | O | `O` | `O` | MATCH |
| 7 | प्रमोद | B-PER | `B-PER` | `B-PER` | MATCH |
| 8 | सावंत | I-PER | `I-PER` | `I-PER` | MATCH |
| 9 | आनी | O | `O` | `O` | MATCH |
| 10 | केंद्रीय | O | `O` | `O` | MATCH |
| 11 | मंत्री | O | `O` | `O` | MATCH |
| 12 | श्रीपाद | B-PER | `O` | `O` | MISSED |
| 13 | नाईक | I-PER | `O` | `O` | MISSED |
| 14 | हांणी | O | `O` | `O` | MATCH |
| **15** | **गोयाच्या** | **B-LOC** | `O` | **`O`** | **MISSED** |
| 16 | आर्थिक | O | `O` | `O` | MATCH |
| 17 | विकासा | O | `O` | `O` | MATCH |
| 18 | खातीर | O | `O` | `O` | MATCH |
| 19 | नव्या | O | `O` | `O` | MATCH |
| 20 | प्रकल्पांची | O | `O` | `O` | MATCH |
| 21 | घोशणा | O | `O` | `O` | MATCH |
| 22 | केली. | O | `O` | `O` | MATCH |

---

## 2. Recovery Analysis for पणजींत and गोयाच्या

- **Token 0 (`पणजींत`)**: Baseline predicted `O`. LOC-Patched predicted **`O`**.
  - *Result*: STILL MISSED.
- **Token 15 (`गोयाच्या`)**: Baseline predicted `O`. LOC-Patched predicted **`O`**.
  - *Result*: STILL MISSED.

---

## 3. Findings

Neither inflected entity was recovered by the retrained checkpoint.
