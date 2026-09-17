# Supplementary Long Multi-Entity Gold-Set Evaluation Report

## Executive Summary

This audit evaluated pre-patch vs post-patch checkpoints on a newly constructed, strictly independent supplementary gold dataset (`data/gold/{lang}/supplementary_long_multientity.conll`).
Unlike the original gold-dev set (which contained 0/0 missed middle entities and virtually zero multi-entity lists in long sentences), this dataset was specifically curated to test long sentences (>= 30 tokens) containing multi-entity comma/conjunction lists.

## Evaluation Results Summary

| Language | Sentences | Gold Middle Ents | Pre-Patch F1 | Post-Patch F1 | Delta F1 | Pre Middle Recall | Post Middle Recall | Delta Middle Recall |
|---|---|---|---|---|---|---|---|---|
| **Konkani** | 12 | 4 | 92.50% | 91.36% | **-1.14%** | 100.00% (4/4) | 100.00% (4/4) | **+0.00%** |
| **Assamese** | 35 | 78 | 94.61% | 94.61% | **+0.00%** | 96.15% (75/78) | 96.15% (75/78) | **+0.00%** |
| **Sinhala** | 28 | 30 | 98.25% | 98.25% | **+0.00%** | 96.67% (29/30) | 96.67% (29/30) | **+0.00%** |

## Detailed Per-Language Findings

### Konkani
- **Sentence Count**: 12 (shortfall noted if below 35 target due to exhaustive raw-corpus ceiling)
- **Total Multi-Entity Middle Targets**: 4

#### Pre-Patch vs Post-Patch Per-Class Breakdown

| Class | Pre Precision | Pre Recall | Pre F1 | Post Precision | Post Recall | Post F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| LOC | 100.00% | 86.05% | 92.50% | 100.00% | 86.05% | 92.50% | +0.00% |
| MISC | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | +0.00% |
| ORG | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | +0.00% |
| PER | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | +0.00% |

#### Middle-Entity Resolution Analysis (Konkani)
- **Pre-Patch**: 4 / 4 correctly detected (100.0%), 0 dropped.
- **Post-Patch**: 4 / 4 correctly detected (100.0%), 0 dropped.
- **Net Impact**: Delta in middle-entity recall is +0.00%.

### Assamese
- **Sentence Count**: 35 (shortfall noted if below 35 target due to exhaustive raw-corpus ceiling)
- **Total Multi-Entity Middle Targets**: 78

#### Pre-Patch vs Post-Patch Per-Class Breakdown

| Class | Pre Precision | Pre Recall | Pre F1 | Post Precision | Post Recall | Post F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| LOC | 93.08% | 97.37% | 95.18% | 93.08% | 97.37% | 95.18% | +0.00% |
| MISC | 91.67% | 100.00% | 95.65% | 91.67% | 100.00% | 95.65% | +0.00% |
| ORG | 84.62% | 73.33% | 78.57% | 84.62% | 73.33% | 78.57% | +0.00% |
| PER | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | +0.00% |

#### Middle-Entity Resolution Analysis (Assamese)
- **Pre-Patch**: 75 / 78 correctly detected (96.15%), 3 dropped.
- **Post-Patch**: 75 / 78 correctly detected (96.15%), 3 dropped.
- **Net Impact**: Delta in middle-entity recall is +0.00%.

### Sinhala
- **Sentence Count**: 28 (shortfall noted if below 35 target due to exhaustive raw-corpus ceiling)
- **Total Multi-Entity Middle Targets**: 30

#### Pre-Patch vs Post-Patch Per-Class Breakdown

| Class | Pre Precision | Pre Recall | Pre F1 | Post Precision | Post Recall | Post F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| LOC | 100.00% | 96.46% | 98.20% | 100.00% | 96.46% | 98.20% | +0.00% |
| MISC | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | +0.00% |
| ORG | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | +0.00% |
| PER | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | +0.00% |

#### Middle-Entity Resolution Analysis (Sinhala)
- **Pre-Patch**: 29 / 30 correctly detected (96.67%), 1 dropped.
- **Post-Patch**: 29 / 30 correctly detected (96.67%), 1 dropped.
- **Net Impact**: Delta in middle-entity recall is +0.00%.

## Audit Conclusions from Supplementary Evaluation

1. **Konkani**: The patched checkpoint represents genuinely retrained weights. On the supplementary long multi-entity set, the empirical delta demonstrates whether the list-augmentation translated to actual generalization on long sentences.
2. **Assamese & Sinhala**: The patched checkpoints are parameter-identical to their pre-patch checkpoints (L2 diff = 0.0000). Consequently, performance on the supplementary set is identically identical (0.00% delta across all metrics) because no weights were updated during patch retraining.
3. **Hypothesis Validation**: The patch's core claim that list-augmentation systematically eliminates middle-entity dropping is unproven for Assamese and Sinhala because retraining did not update their checkpoints, and directly measured on Konkani via this supplementary set.
