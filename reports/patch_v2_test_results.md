# Patch v2 Test Results & Verification Audit

**Date**: 2026-09-17  
**Objective**: Re-audit Patch v2 under the fixed composite selection logic and weight divergence gate.  

---

## 1. Weight Divergence Gate Verification

| Language | Gate Status | Total Params | Changed Params | % Changed | L2 Diff | Max Diff | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Assamese** | **PASS** | 10,596,571 | 2,796,454 | **26.39%** | 12.2750 | 0.1502 | Genuinely updated weights |
| **Konkani** | **PASS** | 10,596,571 | 2,826,881 | **26.68%** | 10.0419 | 0.1090 | Genuinely updated weights |
| **Sinhala** | **PASS** | 10,596,571 | 2,413,105 | **22.77%** | 5.2563 | 0.0398 | Genuinely updated weights |

---

## 2. Standing Long Multi-Entity Evaluation Slice (`long_multientity_eval.conll`)

| Language | Total Middle Targets | Base Middle Recall | Patched Middle Recall | Delta Middle Recall | Base ORG F1 | Patched ORG F1 | Delta ORG F1 | Base Overall F1 | Patched Overall F1 | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | 43 | 46.51% (20/43) | 51.16% (22/43) | **+4.65%** | 2.94% | 11.43% | **+8.49%** | 57.83% | 61.90% | **+4.07%** |
| **Konkani** | 25 | 40.00% (10/25) | 24.00% (6/25) | **-16.00%** | 5.88% | 2.30% | **-3.58%** | 34.82% | 30.40% | **-4.42%** |
| **Sinhala** | 26 | 15.38% (4/26) | 15.38% (4/26) | **+0.00%** | 0.00% | 1.12% | **+1.12%** | 20.18% | 20.63% | **+0.45%** |

---

## 3. Standard Gold-Dev Benchmark Comparison

| Language | Base Dev Overall F1 | Patched Dev Overall F1 | Delta Dev F1 | Base Long F1 | Patched Long F1 | Delta Long F1 | Base Dev ORG F1 | Patched Dev ORG F1 | Delta ORG F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Assamese** | 98.92% | 98.22% | **-0.70%** | 100.00% | 100.00% | **+0.00%** | 96.55% | 96.55% | **+0.00%** |
| **Konkani** | 96.69% | 92.31% | **-4.38%** | 89.66% | 82.76% | **-6.90%** | 98.63% | 89.19% | **-9.44%** |
| **Sinhala** | 99.24% | 96.65% | **-2.59%** | 97.96% | 96.00% | **-1.96%** | 100.00% | 90.32% | **-9.68%** |