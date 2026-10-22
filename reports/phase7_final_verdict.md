# Phase 7: Final Plain-Stated Empirical Verdict

> **Phase 7 Deliverable — Definitive Assessment Closing the Reliability Track**  
> Direct, honest answer to whether the originally reported failure mode — model misses genuine entities, sometimes returns none for sentences that clearly contain several, and degrades on longer, multi-entity sentences — is actually reduced.

---

## 1. Summary of Final Verdicts

| Language | Current Best Checkpoint | Plain-Stated Verdict | Primary Evidence |
| :--- | :--- | :---: | :--- |
| **Konkani** | `baseline_v1` (`improved_konkani_v2.pt`) | **UNCHANGED** | Inflected locations (`पणजींत`, `गोयाच्या`) and middle list item (`श्रीपाद नाईक`) remain missed (2/5 detected). Stress-set collapse rate unchanged at 75.76%. Total manifest errors unchanged at 90. |
| **Assamese** | `baseline_v1` (`improved_assamese_v2.pt`) | **UNCHANGED** | Long multi-entity sentence collapse remains at 70.97%. Manual stress sentence #200 completely collapsed (0/8 detected). Total manifest errors unchanged at 86. |
| **Sinhala** | `combined` (`improved_sinhala_v2.pt`) | **MEASURABLY REDUCED** | Breaks complete collapse on the 29-token original failing sentence (from 0 to 1 entity detected). Middle recall on stress set improved from 0.00% to 16.67%. Stress F1 improved from 36.36% to 38.46%. |

---

## 2. Plain-Stated Narrative & Closing of the Reliability Track

1. **Sinhala — Measurably Reduced**:
   The combination of silver LOC precision filtering and conjunct robustness training definitively made progress against complete collapse. On the 29-token failing sentence that completely collapsed under the baseline, the current best model recovers the leading person entity (`මහාචාර්ය අනුර`). On the stress set, middle-entity recall improved from $0.00\%$ to $16.67\%$, and stress F1 improved from $36.36\%$ to $38.46\%$.
2. **Konkani — Unchanged**:
   Neither isolated interventions nor compound retraining resolved the core structural limitation. Inflected case endings missing from the base vocabulary continue to cause representations to degrade into `O`, and CRF transition dynamics still suppress middle entities in lists of $\ge 3$. Because Phase 6 correctly rejected the regressive compound model, the baseline was preserved, leaving performance strictly unchanged ($90$ errors, $75.76\%$ stress collapse).
3. **Assamese — Unchanged**:
   Operating near the ceiling on standard sentences ($98.92\%$ Dev F1), the Assamese model remains completely vulnerable to complete-sentence collapse under long-sequence multi-entity conditions ($70.97\%$ collapse rate). Lexical inflectional augmentation and mild class weighting failed to alter this recurrent state decay ($86$ errors).
4. **Honest Scientific Conclusion**:
   The Reliability Track closes with full empirical honesty: **The problem is measurably reduced in Sinhala, but remains unchanged in Konkani and Assamese.** Localized lexical and weak supervision adjustments can mitigate orthographic and precision issues, but structural sentence collapse under high entity density requires deeper contextual representations beyond static from-scratch word embeddings and linear-chain CRFs.
