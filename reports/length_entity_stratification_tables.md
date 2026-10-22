# Consolidated Length & Entity-Density Stratification Tables

> **Phase 2 Deliverable — Cross-Lingual Failure Breakdown**  
> Directly separates sentence-length effects from entity-density effects across Konkani, Assamese, and Sinhala.

---

## 1. Length Stratification Comparison

| Language | Length Stratum | Sentences | Recall | Entity F1 | Complete Collapses | Collapse Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | L1_Short (<=12) | 88 | 97.30% | 98.63% | **1** | **2.9%** |
| **Konkani** | L2_Medium (13-22) | 71 | 92.73% | 96.23% | **4** | **9.3%** |
| **Konkani** | L3_Long (23-34) | 46 | 60.38% | 75.29% | **9** | **31.0%** |
| **Konkani** | L4_Stress (>34) | 42 | 40.57% | 57.33% | **16** | **44.4%** |
| **Assamese** | L1_Short (<=12) | 64 | 95.65% | 97.78% | **2** | **4.4%** |
| **Assamese** | L2_Medium (13-22) | 89 | 100.00% | 99.17% | **0** | **0.0%** |
| **Assamese** | L3_Long (23-34) | 51 | 73.26% | 84.00% | **8** | **21.6%** |
| **Assamese** | L4_Stress (>34) | 66 | 77.23% | 84.18% | **14** | **24.6%** |
| **Sinhala** | L1_Short (<=12) | 52 | 100.00% | 100.00% | **0** | **0.0%** |
| **Sinhala** | L2_Medium (13-22) | 69 | 100.00% | 98.99% | **0** | **0.0%** |
| **Sinhala** | L3_Long (23-34) | 78 | 89.25% | 91.21% | **5** | **10.4%** |
| **Sinhala** | L4_Stress (>34) | 64 | 74.81% | 83.76% | **13** | **25.0%** |

---

## 2. Entity-Density Stratification Comparison

| Language | Entity Stratum | Sentences | Recall | Entity F1 | Complete Collapses | Collapse Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | E0 (0 Entities) | 104 | N/A | 100.00% | **N/A** | **N/A** |
| **Konkani** | E1 (1 Entity) | 91 | 79.12% | 88.34% | **19** | **20.9%** |
| **Konkani** | E2 (2 Entities) | 27 | 72.22% | 82.98% | **5** | **18.5%** |
| **Konkani** | E3 (3+ Entities) | 25 | 48.11% | 64.97% | **6** | **24.0%** |
| **Assamese** | E0 (0 Entities) | 76 | N/A | 100.00% | **N/A** | **N/A** |
| **Assamese** | E1 (1 Entity) | 134 | 90.30% | 94.16% | **12** | **9.0%** |
| **Assamese** | E2 (2 Entities) | 10 | 50.00% | 66.67% | **4** | **40.0%** |
| **Assamese** | E3 (3+ Entities) | 50 | 79.77% | 86.19% | **8** | **16.0%** |
| **Sinhala** | E0 (0 Entities) | 99 | N/A | 0.00% | **N/A** | **N/A** |
| **Sinhala** | E1 (1 Entity) | 98 | 89.80% | 94.12% | **10** | **10.2%** |
| **Sinhala** | E2 (2 Entities) | 29 | 72.41% | 78.50% | **6** | **20.7%** |
| **Sinhala** | E3 (3+ Entities) | 37 | 88.19% | 93.04% | **2** | **5.4%** |
