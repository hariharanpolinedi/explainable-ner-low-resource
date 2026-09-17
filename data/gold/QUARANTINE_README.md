# Quarantine Notice: `CONTAMINATED_long_multientity_eval_v1.conll`

**Date Quarantined**: 2026-09-17  
**Status**: **DO NOT USE / INVALID BENCHMARK**  
**Audit Finding**: See `reports/verify_selection_bias_findings.md` and `reports/verify_annotation_independence_findings.md`.

---

## Reasons for Quarantine

1. **Severe Selection Bias**: Candidate sentences in v1 were screened for ORG keywords and gazetteers, producing artificial ORG shares of 80.5% (Konkani), 57.0% (Assamese), and 88.7% (Sinhala), compared to natural ~10-30% baselines.
2. **Circular Weak-Supervision Injection**: 98-99% of all ORG entities in v1 were directly injected by `weak_supervision/org_rule_taggers.py:DedicatedORGRuleTagger` into the "gold" file without human validation.
3. **Automated Stand-In Annotation**: "Annotator A" and "Annotator B" were programmatic calls to a synthetic rule script rather than independent human annotators.
4. **Replacement**: Use `data/gold/{lang}/long_multientity_eval_v2.conll`, constructed via length/structure-only sampling and genuine human annotation.
