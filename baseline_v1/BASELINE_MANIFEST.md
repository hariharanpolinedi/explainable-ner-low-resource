# Baseline v1 Master Manifest & Cryptographic Ledger

> [!IMPORTANT]
> **IMMUTABLE BASELINE V1 SPECIFICATION**  
> `baseline_v1/` is an immutable historical snapshot. Nothing under this directory is ever modified, retrained, or overwritten by any later phase. All v2 development happens outside this directory.
> Every subsequent evaluation, ablation, reliability intervention (Phases 1–7), and Kokborok comparison (Phases 8–17) must diff strictly against the immutable numbers and artifacts recorded here.

- **Snapshot Name**: `baseline_v1`
- **Creation Date**: `2026-09-22`
- **Primary Git Commit Hash**: `3235ff12925d3e5393ab1bfcb8bcd68402746f3e`
- **Git Tag**: `baseline-v1-frozen`
- **Hardware Readiness**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), Driver: `595.79`, CUDA: `13.2`, PyTorch: `2.6.0+cu124` (Readiness check only; zero training/inference during archival)
- **Total Archived Files**: 55
- **Archival Verification**: 100% of copied files independently re-checksummed and verified byte-identical against their workspace source locations.

---

## 1. Archival Scope & Decision Records

### 1.1 Model Checkpoints (`baseline_v1/checkpoints/`)
Contains the three authoritative final checkpoints for the three validated languages:
- `final_konkani.pt` (Phase 10 Self-Training genuine best; byte-identical to `selftrained_konkani_best.pt`)
- `final_assamese.pt` (Phase 11 Augmentation genuine best; byte-identical to `augmented_assamese.pt`)
- `final_sinhala.pt` (Phase 11 Augmentation genuine best; byte-identical to `augmented_sinhala.pt`)

### 1.2 Benchmark Data Splits (`baseline_v1/data/`)
Contains the full, adjudicated gold benchmark splits for Konkani, Assamese, and Sinhala:
- `train.conll` (Expanded 850 sentences/language)
- `dev.conll` (Locked 200 sentences/language)
- `test.conll` (Cryptographically locked 200 sentences/language, matching `reports/gold_test_lockfile.md`)
- Adjudicated evaluation sets: `long_multientity_eval_v2.conll`, `supplementary_long_multientity.conll`
- Quarantined legacy artifacts: `QUARANTINE_README.md`, `CONTAMINATED_long_multientity_eval_v1.conll`

### 1.3 Silver, Pseudo, and Augmented Data Scope Decision
- Full data directories (`data/silver/`, `data/pseudo/`, `data/augmented/`) are intentionally **not** duplicated into `baseline_v1/`.
- *Rationale*: These datasets are storage-heavy and fully deterministic/reproducible from the gold splits and pipeline generators.
- *Traceability Anchor*: The exact volume manifests, token counts, and entity compositions are preserved verbatim in `baseline_v1/paper_assets/dataset_statistics.md`. The complete historical data files remain permanently reconstructible under git commit hash `3235ff12925d3e5393ab1bfcb8bcd68402746f3e`.

### 1.4 Kokborok Dataset Status
- Folder confirmed present at root: `C:\Users\harih\Desktop\NLP-MARK-3\Kokborok_datasets`.
- *Status*: **STRICTLY UNTOUCHED & OUT OF SCOPE FOR PHASE 0**.
- In strict adherence to the phase boundary, no file inside this folder was read, opened, parsed, or characterized. Its contents, format, and volume will be inspected exclusively when Phase 9 begins.

### 1.5 Configuration Files (`baseline_v1/configs/`)
Contains `configs/base_config.yaml` and the language-specific production configurations `configs/final_{konkani,assamese,sinhala}.yaml`.

### 1.6 Publication Assets & Provenance (`baseline_v1/paper_assets/`)
The entire `paper_assets/` package has been mirrored verbatim, including all 10 master markdown records and the 16 visual artifacts in `figures/`.

---

## 2. Complete Cryptographic File Ledger (SHA-256)

| Category | Relative File Path in `baseline_v1/` | Byte Size | SHA-256 Checksum | Match Status vs Original |
| :--- | :--- | :---: | :--- | :---: |
| `checkpoints` | [`checkpoints/final_assamese.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/checkpoints/final_assamese.pt) | 42,400,693 | `2b7688e17b394ba0dd35bc616ed72460b08aaf717d467b8fa84e7170ae37e38d` | **VERIFIED (EXACT)** |
| `checkpoints` | [`checkpoints/final_konkani.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/checkpoints/final_konkani.pt) | 42,401,269 | `58eb383594ee205960836756a7c5c4525c1379d52e7c89284dd9762b7d54a038` | **VERIFIED (EXACT)** |
| `checkpoints` | [`checkpoints/final_sinhala.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/checkpoints/final_sinhala.pt) | 42,400,645 | `6c268138e546c60e0ed9912606b4133f9011515d562674f764d1f8965e377b03` | **VERIFIED (EXACT)** |
| `configs` | [`configs/base_config.yaml`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/configs/base_config.yaml) | 5,255 | `d1345f91a803553d73a7bd340e53a31fc795597164cae276052ca1766fea9f45` | **VERIFIED (EXACT)** |
| `configs` | [`configs/final_assamese.yaml`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/configs/final_assamese.yaml) | 690 | `6bc10adf20fbc072d194279ca46e7e6c7b08863acb2ebb9e95ae0889c02b3e29` | **VERIFIED (EXACT)** |
| `configs` | [`configs/final_konkani.yaml`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/configs/final_konkani.yaml) | 689 | `02a53729fd4056926c44a2fa40090c2654f930f1f16ec7d19cd574c6f46c10e3` | **VERIFIED (EXACT)** |
| `configs` | [`configs/final_sinhala.yaml`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/configs/final_sinhala.yaml) | 689 | `d760f3ef1b5d2af191b6157d412ffea9ae509a287d7acbb6e900ccfdea0f63c4` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/QUARANTINE_README.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/QUARANTINE_README.md) | 1,070 | `92d3c6f5604bd6465574d329d60bcdd61a532ef91204aa015fcf45d38e85dd33` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/.gitkeep`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/.gitkeep) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/CONTAMINATED_long_multientity_eval_v1.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/CONTAMINATED_long_multientity_eval_v1.conll) | 20,580 | `a1df5cbc9da2bc2bc80c82d38cced15beeb42ef949f7396be0815cae7e442050` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/dev.conll) | 64,007 | `d4b5a59b4daa76621777992174cef842ae94e9a1da2081a59100803a7df92a81` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/long_multientity_eval_v2.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/long_multientity_eval_v2.conll) | 24,018 | `bce6d69071a26e3180eae58f54e9e22cf66a9bd6389ca5aa51f26fbb5106b17c` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/supplementary_long_multientity.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/supplementary_long_multientity.conll) | 33,016 | `c608755568bf6b010a4331c3a347d02113d87d47a86ddb883226014d324f28c8` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/test.conll) | 59,817 | `2a49321609716b38a893d7b7ca5aa3c27ba0c90863fbca8ac0d4aa87d143c070` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/assamese/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/assamese/train.conll) | 262,481 | `dc891de435a5c328ffd24d65fc9cec58625f612180752b6e8164b1e4766bfd42` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/.gitkeep`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/.gitkeep) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/CONTAMINATED_long_multientity_eval_v1.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/CONTAMINATED_long_multientity_eval_v1.conll) | 54,814 | `05b796969e0a59805942d845bca1482ca74b771ae2f2e769e5961900f05042ba` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/dev.conll) | 57,855 | `d90e1e4718c69db97da9f00d90c1e3711895378a13c72c9a66466d26bd535898` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/long_multientity_eval_v2.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/long_multientity_eval_v2.conll) | 26,938 | `4884414b52492e5d80270e5594f76f0101593efeb6db7af44beb3c8723bc4aa5` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/supplementary_long_multientity.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/supplementary_long_multientity.conll) | 8,850 | `50f62c1c9e8a7d7f825cc60cc52bc4dea948d002093ab0d40ef9d5641da96367` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/test.conll) | 58,300 | `bd08ebbf60e048d4fce5d9f5fd233f7dbdeb5bf1f066ee0718eb4b04404ab2d1` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/konkani/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/konkani/train.conll) | 258,534 | `baec317909b34d76e802d5a51eed6fca45d785dd6e368480f32a8f7b7017868f` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/.gitkeep`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/.gitkeep) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/CONTAMINATED_long_multientity_eval_v1.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/CONTAMINATED_long_multientity_eval_v1.conll) | 58,116 | `cd5a4003336d54a89683ea0e0b83027c75d80df69c0b3149c784c34f303af056` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/dev.conll) | 72,406 | `3ce4f3011d2878f99a6fa408232e92103971281f6628a6171fabf127390fe3da` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/long_multientity_eval_v2.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/long_multientity_eval_v2.conll) | 28,074 | `645d4b040262fd5184738b75266df99ffdba3108248292857f2f0f905d69f8fd` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/supplementary_long_multientity.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/supplementary_long_multientity.conll) | 22,316 | `1ce03c5dd126670ee43b3abe31de2be21bbff55b1e040b37c70f3627c4871852` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/test.conll) | 71,546 | `47cf890b0e9c76e235e17355806b521f8925525e80256bf6667b2835a16854d0` | **VERIFIED (EXACT)** |
| `data` | [`data/gold/sinhala/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/data/gold/sinhala/train.conll) | 296,090 | `350aaf6095a5187eb2b31012cf3e7b06967d73d35daa4504e2628c567c0765a6` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/MANIFEST.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/MANIFEST.md) | 11,011 | `85b1142c2bb05851b406deee3b19b84215b437fa5beeb0117b9c21f14cc0bce2` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/ablation_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/ablation_table.md) | 13,097 | `533949f0357458e69e0a67588135fb9846f61aa43a5d6b6295e764951c6d63ac` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/architecture_diagram_spec.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/architecture_diagram_spec.md) | 7,807 | `277d130315c3e527b567bb0f45e0c3dd02a0d7ca6d277d19062eb2816035f0a4` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/checkpoint_provenance_final.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/checkpoint_provenance_final.md) | 6,363 | `05fb6d49854e2ca2a822c54133a87be250169080f367aed5f4ff70a60b90b9f2` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/dataset_statistics.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/dataset_statistics.md) | 6,073 | `050bb03e48c6af7022a2a33b41d13bc405078dab8944b6c4f9cc00db78262cf9` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/faithfulness_results.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/faithfulness_results.md) | 9,017 | `7236a0bf6b11d181bf8bad4b12dc4ab0f6df106df2d3615306540a4883542970` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_entities_assamese.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_entities_assamese.png) | 113,612 | `ea382b9d61b75c5efe24891f02a235355c68c21def9258c62237c2db1d86416e` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_entities_konkani.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_entities_konkani.png) | 111,205 | `a7644e7dc65aa1f6991ec191da1b9ec52d42d172262ee6f3856069c79f866f3f` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_entities_sinhala.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_entities_sinhala.png) | 111,561 | `dc6653439b00ed3e6e3b98634066f633a6d7ffcb04748ec4cb53029c957950be` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_tokens_assamese.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_tokens_assamese.png) | 162,803 | `1492a2b63adbb47ea3ce9be8896d104b65a16097c99bab62b6a0db9e9d955ba7` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_tokens_konkani.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_tokens_konkani.png) | 162,887 | `1b30938e76a27712337e2469afa37a9d8768006621ad3555b00cd3e4ec668dc5` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_confusion_matrix_tokens_sinhala.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_confusion_matrix_tokens_sinhala.png) | 167,110 | `c34cace517eba3497faf0a6b59b73b4b6ad0fdd774245523bfcc2e0b196e032e` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/fig_selftraining_curve.png`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/fig_selftraining_curve.png) | 196,893 | `d767df99e1fb901fbf45d25ba887aecca6ff540c7672448a970394e0d136c03f` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_assamese_loc_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_assamese_loc_1.html) | 6,717 | `cb33fcfcf1689a3a292ef8cfaf507dfa71c21e5371c7a95ec5e36d0ae59743a3` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_assamese_org_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_assamese_org_1.html) | 6,714 | `8e4638f62a95fed5d5a877caa1894fa46d126db50e71c1f366336f186e0ecbeb` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_assamese_per_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_assamese_per_1.html) | 8,150 | `aab476a49a67713263825ddd56318050b545b36e4ae8cba739e5ba4393c69979` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_konkani_loc_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_konkani_loc_1.html) | 14,217 | `1c71dcf28d15b181ee62d8f9312a36700f246cc0ebd1c967af7e50dcc17b07f9` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_konkani_org_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_konkani_org_1.html) | 14,214 | `d929875f8452b56e85c8d8666be7733241bb467c47db90f84c087f801d266c7e` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_konkani_per_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_konkani_per_1.html) | 13,639 | `90b98154b769d40ede574bde3043c48b9995ac867a3432865bb0f790bb2b1f95` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_sinhala_loc_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_sinhala_loc_1.html) | 6,607 | `2c00598983fb2433bbf65eb0130309d1a15c72de9143608ecce7bdf04bb4be5c` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_sinhala_org_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_sinhala_org_1.html) | 6,610 | `e5f6e2004523225560caadb4c6e28a90d25a2f959694788b5adc895975caf422` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures/heatmap_sinhala_per_1.html`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures/heatmap_sinhala_per_1.html) | 8,946 | `237c540eaaa7af89e0b36c5293e869621511ae2035f7714873c978b38527c266` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/figures_manifest.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/figures_manifest.md) | 5,083 | `1e06b9a8920801496891f7e19f908309e4430f45d512fa8bbd966a53b2f87931` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/gaps_and_limitations.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/gaps_and_limitations.md) | 9,778 | `1a18dd1adaff193c276d3732f0bff61af507bcfcfddcc28db2fddc519287de98` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/honest_project_history.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/honest_project_history.md) | 10,745 | `eef77fb5ca87952ba6a48dc83d62bb0240c1c5b7ce5e4964c43ba00b784df092` | **VERIFIED (EXACT)** |
| `paper_assets` | [`paper_assets/results_master_table.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/baseline_v1/paper_assets/results_master_table.md) | 7,110 | `061ad8ebdf7ebf83c9cb71271d71ca1c9f4e0135c3f407eabffa0f77f67ab614` | **VERIFIED (EXACT)** |
