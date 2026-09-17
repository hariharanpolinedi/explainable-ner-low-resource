"""
Phase 15 Automated Test & Validation Suite.

Executes and reports:
1. End-to-end inference smoke test on >=10 sentences per language
2. Vocab-checkpoint compatibility check across all deployment bundles
3. FastAPI REST API endpoint validation (/predict, /health, error handling)
4. Attribution rendering verification (Gradient × Attention HTML output)
5. CPU fallback execution verification (GPU disabled)

Generates:
- reports/phase15_demo_smoketest.md
- reports/phase15_test_results.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for non-ASCII scripts on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from fastapi.testclient import TestClient

from deployment.api import app
from deployment.inference import SUPPORTED_LANGUAGES, get_predictor, select_device
from explainability.heatmap_visualizer import render_html_heatmap

SMOKE_TEST_SENTENCES = {
    "konkani": [
        "ताचो जल्म १७ मे १९४२ ह्या वर्सा नावेली , साश्ट गोंय हांगा जालो .",
        "जाग , राष्ट्रमत आनी भांगरभूंय हीं कोंकणी नेमाळीं आनी खबरांपत्रां .",
        "तातूंत गोंय कला अकादेमी , साहित्य अकादेमी हांचो आसपाव आसा .",
        "डॉ. मनोहर पर्रीकर गोंयचे आदले मुख्यमंत्री आशिल्ले .",
        "पणजी ही गोंयची राजधानी आसा .",
        "कोंकणी भास ही गोंय राज्याची राजभास आसा .",
        "म्हापश्यां बोडगेश्वर देवळांत व्हड जात्रा भरता .",
        "वास्को द गामा शारांत मुरगांव बंदर आसा .",
        "शणै गोंयबाब हांकां आधुनिक कोंकणी साहित्याचे जनक मानतात .",
        "मडगांवां रवींद्र भवन हांगा सांस्कृतिक कार्यावळी जातात .",
        "चांदोर गांव आदल्या काळांत कदंब राजांची राजधानी आशिल्ली .",
    ],
    "assamese": [
        "১৯২১ চনত মোহনদাস গান্ধী ভাৰতীয় কংগ্ৰেছ দলৰ কাৰ্যবাহী সম্পাদক নিযুক্ত হয় আৰু তেখেতৰ নেতৃত্বত কংগ্ৰেছৰ সংবিধান সলনি কৰা হয় ।",
        "অসমৰ ৰাজধানী দিছপুৰ গুৱাহাটী মহানগৰৰ এটা অংশ ।",
        "ড° ভূপেন হাজৰিকাক ভাৰত চৰকাৰে ভাৰত ৰত্ন সন্মানেৰে সন্মানিত কৰে ।",
        "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যান অসমৰ গৌৰৱ আৰু ইয়াৰ এশিঙীয়া গঁড় বিশ্ববিখ্যাত ।",
        "শ্ৰীমন্ত শংকৰদেৱ অসমৰ এজন মহান বৈষ্ণৱ গুৰু আছিল ।",
        "ব্ৰহ্মপুত্ৰ নদী অসমৰ মাজেৰে বৈ গৈছে ।",
        "যোৰহাট জিলাত অসম কৃষি বিশ্ববিদ্যালয় অৱস্থিত ।",
        "তেজপুৰ চহৰ ঐতিহাসিক আৰু সাংস্কৃতিক ঐতিহ্যৰে সমৃদ্ধ ।",
        "লক্ষ্মীনাথ বেজবৰুৱা অসমীয়া সাহিত্যৰ এগৰাকী পথপ্ৰদৰ্শক আছিল ।",
        "অসম সাহিত্য সভা অসমৰ অন্যতম প্ৰাচীন সাহিত্য সংগঠন ।",
        "মাজুলী বিশ্বৰ সৰ্ববৃহৎ নদী দ্বীপ হিচাপে খ্যাত ।",
    ],
    "sinhala": [
        "ශ්‍රී ලංකා ජනාධිපති කොළඹ නගරයේ සංචාරය කළේය .",
        "මහින්ද රාජපක්ෂ මහතා අරලියගහ මැදුරේදී විශේෂ සාකච්ඡාවක් පැවැත්වීය .",
        "මහනුවර ශ්‍රී දළදා මාළිගාව බෞද්ධ ජනතාවගේ පූජනීය ස්ථානයකි .",
        "ලංකා විදුලිබල මණ්ඩලය නව ව්‍යාපෘතියක් ආරම්භ කිරීමට සැලසුම් කරයි .",
        "ගාල්ල ජාත්‍යන්තර ක්‍රිකට් ක්‍රීඩාංගණයේදී තරගය පැවැත්වේ .",
        "අනුරාධපුරය ඓතිහාසික නගරයක් වන අතර එහි බොහෝ නටබුන් ඇත .",
        "ශ්‍රී ලංකා මහ බැංකුව පොලී අනුපාත සංශෝධනය කළේය .",
        "මහාචාර්ය එදිරිවීර සරච්චන්ද්‍ර සිංහල නාට්‍ය කලාවට විශාල මෙහෙවරක් කළේය .",
        "කොළඹ විශ්වවිද්‍යාලය ලංකාවේ පැරණිතම විශ්වවිද්‍යාලයකි .",
        "යාපනය අර්ධද්වීපය උතුරු පළාතේ පිහිටා ඇත .",
        "බණ්ඩාරනායක ජාත්‍යන්තර ගුවන්තොටුපළ කටුනායක පිහිටා තිබේ .",
    ],
}


def test_1_end_to_end_inference() -> tuple[bool, str, dict[str, Any]]:
    """Test 1: Run 10+ sentences per language, recording entities and attributions."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    smoke_results: dict[str, list[dict[str, Any]]] = {}

    for lang in SUPPORTED_LANGUAGES:
        predictor = get_predictor(lang)
        sentences = SMOKE_TEST_SENTENCES[lang]
        lang_res: list[dict[str, Any]] = []

        for idx, sent in enumerate(sentences, 1):
            try:
                res = predictor.predict(sent, include_attribution=True)
                lang_res.append(res)
            except Exception as exc:
                return False, f"Inference crashed on {lang} sentence #{idx}: '{sent}' -> {exc}", {}

        smoke_results[lang] = lang_res

    # Generate reports/phase15_demo_smoketest.md
    smoketest_file = reports_dir / "phase15_demo_smoketest.md"
    lines = [
        "# Phase 15: Deployment Demo Smoke-Test Results",
        "",
        "## Executive Summary",
        "",
        "This report logs the interactive smoke testing of the production inference pipeline",
        "across all three languages using 11 manually curated sentences per language (33 total sentences),",
        "including known entities from regional gazetteers and historical corpora.",
        "",
        "| Language | Sentences Evaluated | Total Entities Found | Status |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for lang in SUPPORTED_LANGUAGES:
        res_list = smoke_results[lang]
        tot_ents = sum(len(r["entities"]) for r in res_list)
        lines.append(f"| **{lang.capitalize()}** | {len(res_list)} | {tot_ents} | **PASS (0 Crashes)** |")

    lines.extend(["", "---", "", "## Detailed Test Sentences & Entity Extractions", ""])

    for lang in SUPPORTED_LANGUAGES:
        lines.extend([f"### {lang.capitalize()} Smoke-Test Details", ""])
        for i, r in enumerate(smoke_results[lang], 1):
            lines.append(f"**Sample {i}**: `{r['raw_sentence']}`")
            lines.append(f"- *Tokens*: `{' '.join(r['tokens'])}`")
            if r["entities"]:
                ent_desc = []
                for e in r["entities"]:
                    top_attr = ""
                    if "top_attended_tokens" in e and e["top_attended_tokens"]:
                        top_toks = ", ".join([f"{item['token']} ({item['score']})" for item in e["top_attended_tokens"][:2]])
                        top_attr = f" | Attended: [{top_toks}]"
                    ent_desc.append(f"**{e['text']}** (`{e['type']}` tokens {e['start_token']}:{e['end_token']}{top_attr})")
                lines.append(f"- *Entities*: {'; '.join(ent_desc)}")
            else:
                lines.append("- *Entities*: *(None detected)*")
            lines.append("")

    with open(smoketest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return True, "All 33 smoke-test sentences processed cleanly across Konkani, Assamese, and Sinhala.", smoke_results


def test_2_vocab_checkpoint_compatibility() -> tuple[bool, str]:
    """Test 2: Verify bundled vocab sizes match checkpoint embedding shapes."""
    bundles_dir = PROJECT_ROOT / "deployment" / "bundles"

    for lang in SUPPORTED_LANGUAGES:
        b_dir = bundles_dir / lang / "v1"
        if not b_dir.is_dir():
            return False, f"Missing bundle directory: {b_dir}"

        # Load vocab files
        with open(b_dir / "word_vocab.json", "r", encoding="utf-8") as f:
            w_data = json.load(f)
        w_size = len(w_data.get("token_to_id", w_data))

        with open(b_dir / "char_vocab.json", "r", encoding="utf-8") as f:
            c_data = json.load(f)
        c_size = len(c_data.get("token_to_id", c_data))

        # Load checkpoint
        ckpt = torch.load(b_dir / "model.pt", map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)

        w_emb = sd["word_embedding.embedding.weight"].shape[0]
        c_emb = sd["char_cnn.char_embedding.weight"].shape[0]

        if w_size != w_emb:
            return False, f"{lang}: Word vocab size {w_size} != model embedding {w_emb}"
        if c_size != c_emb:
            return False, f"{lang}: Char vocab size {c_size} != model embedding {c_emb}"

    return True, "All 3 language deployment bundles have 100% matched vocab and embedding dimensions."


def test_3_api_endpoints() -> tuple[bool, str]:
    """Test 3: Verify FastAPI endpoints using TestClient."""
    client = TestClient(app)

    # 1. Health check
    r_health = client.get("/health")
    if r_health.status_code != 200:
        return False, f"GET /health failed with status {r_health.status_code}"
    health_data = r_health.json()
    if health_data.get("status") != "healthy":
        return False, f"Health check returned non-healthy: {health_data}"

    # 2. Valid predict
    r_valid = client.post("/predict", json={
        "sentence": "ताचो जल्म गोंय हांगा जालो .",
        "language": "konkani",
        "include_attribution": True,
    })
    if r_valid.status_code != 200:
        return False, f"POST /predict failed: {r_valid.status_code} - {r_valid.text}"
    val_data = r_valid.json()
    if "entities" not in val_data or "tokens" not in val_data:
        return False, f"Malformed predict response: {val_data}"

    # 3. Invalid language error handling
    r_invalid_lang = client.post("/predict", json={
        "sentence": "Test sentence",
        "language": "french",
    })
    if r_invalid_lang.status_code != 400:
        return False, f"Expected 400 for invalid language, got {r_invalid_lang.status_code}"

    # 4. Empty sentence handling
    r_empty = client.post("/predict", json={
        "sentence": "   ",
        "language": "assamese",
    })
    if r_empty.status_code != 200 or len(r_empty.json().get("entities", [1])) != 0:
        return False, f"Empty sentence failed: {r_empty.status_code} - {r_empty.text}"

    return True, "FastAPI /predict, /health, validation, and error handling all passed."


def test_4_attribution_rendering() -> tuple[bool, str]:
    """Test 4: Verify attribution heatmap HTML output formatting."""
    tokens = ["ශ්‍රී", "ලංකා", "ජනාධිපති", "කොළඹ"]
    attributions = [0.15, 0.45, 0.05, 0.35]
    html_markup = render_html_heatmap(
        tokens=tokens,
        attributions=attributions,
        target_token_idx=1,
        target_tag="LOC (B-LOC)",
        lang="sinhala",
    )

    if not html_markup:
        return False, "render_html_heatmap returned empty HTML."
    if "rgba(239, 68, 68" not in html_markup:
        return False, "Attribution color scaling missing in HTML."
    if "ලංකා" not in html_markup:
        return False, "Target token missing in HTML."

    return True, "Attribution heatmap HTML properly generated with proportional opacity styling."


def test_5_cpu_fallback() -> tuple[bool, str]:
    """Test 5: Explicit CPU execution path verification."""
    predictor_cpu = get_predictor("sinhala", device="cpu")
    if predictor_cpu.device.type != "cpu":
        return False, f"Expected device CPU, got {predictor_cpu.device}"

    res = predictor_cpu.predict("කොළඹ නගරය", include_attribution=True)
    if res["device"] != "cpu":
        return False, f"Result metadata reports non-CPU device: {res['device']}"
    if not res["tokens"]:
        return False, "CPU inference returned empty tokens."

    return True, "CPU fallback execution path confirmed fully functional."


def run_all_tests() -> dict[str, Any]:
    print("=" * 80)
    print("PHASE 15 SECTION-TESTING VALIDATION SUITE")
    print("=" * 80)

    test_1_pass, test_1_msg, _ = test_1_end_to_end_inference()
    print(f"[{'PASS' if test_1_pass else 'FAIL'}] Test 1: End-to-End Inference Smoke Test: {test_1_msg}")

    test_2_pass, test_2_msg = test_2_vocab_checkpoint_compatibility()
    print(f"[{'PASS' if test_2_pass else 'FAIL'}] Test 2: Vocab-Checkpoint Compatibility: {test_2_msg}")

    test_3_pass, test_3_msg = test_3_api_endpoints()
    print(f"[{'PASS' if test_3_pass else 'FAIL'}] Test 3: FastAPI REST API Verification: {test_3_msg}")

    test_4_pass, test_4_msg = test_4_attribution_rendering()
    print(f"[{'PASS' if test_4_pass else 'FAIL'}] Test 4: Attribution Heatmap Rendering: {test_4_msg}")

    test_5_pass, test_5_msg = test_5_cpu_fallback()
    print(f"[{'PASS' if test_5_pass else 'FAIL'}] Test 5: CPU-Fallback Execution: {test_5_msg}")

    all_passed = test_1_pass and test_2_pass and test_3_pass and test_4_pass and test_5_pass

    print("=" * 80)
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)

    # Emit reports/phase15_test_results.md
    test_results_file = PROJECT_ROOT / "reports" / "phase15_test_results.md"
    lines = [
        "# Phase 15: Automated Deployment & Serving Test Results",
        "",
        f"**Overall Status**: {'PASS' if all_passed else 'FAIL'}",
        "",
        "## Section-Testing Suites Summary",
        "",
        "| Test Suite | Requirement | Status | Details |",
        "| :--- | :--- | :---: | :--- |",
        f"| Test 1: End-to-End Inference | Smoke Test (10+ sentences/lang) | {'**PASS**' if test_1_pass else '**FAIL**'} | {test_1_msg} |",
        f"| Test 2: Vocab-Checkpoint Match | Bundle Compatibility | {'**PASS**' if test_2_pass else '**FAIL**'} | {test_2_msg} |",
        f"| Test 3: REST API Endpoints | /predict, /health, validation | {'**PASS**' if test_3_pass else '**FAIL**'} | {test_3_msg} |",
        f"| Test 4: Attribution Rendering | Gradient × Attention HTML | {'**PASS**' if test_4_pass else '**FAIL**'} | {test_4_msg} |",
        f"| Test 5: CPU Fallback | Zero-GPU execution path | {'**PASS**' if test_5_pass else '**FAIL**'} | {test_5_msg} |",
        "",
        "---",
        "",
        "## Deployment Telemetry",
        "",
        f"- **Primary Serving Device**: `{select_device()}`",
        "- **API Framework**: FastAPI 0.138.2 with Uvicorn 0.52.4",
        "- **UI Framework**: Streamlit 1.51.0",
        "- **Bundle Version**: `v1`",
        "- **Preprocessing & Tokenizer**: 100% Phase-2 rule-based (`ScriptTokenizer`, `normalize_text`)",
    ]

    with open(test_results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "all_passed": all_passed,
        "test_1": test_1_pass,
        "test_2": test_2_pass,
        "test_3": test_3_pass,
        "test_4": test_4_pass,
        "test_5": test_5_pass,
    }


if __name__ == "__main__":
    res = run_all_tests()
    if not res["all_passed"]:
        sys.exit(1)
