"""
Blind Test on 5 Kokborok Sentences.
NO SOURCE CODE MODIFICATION.
NO RETRAINING. NO RULE ADDITIONS. NO GAZETTEER ADDITIONS.
Runs raw model predictions exactly as produced by the existing pipeline.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.inference import get_predictor

SENTENCES = [
    {
        "id": "KB_BLIND_1",
        "title": "Sentence 1: Agartala Tripura University Discussion",
        "sentence": "18 September 2026-ot Agartala-ao Tripura University-ao Professor Debashis Jamatia, Dr. Rina Debbarma, Anupam Chakma aru Suman Reang-ni Tripura, Assam, Mizoram aru Bangladesh-ni researchers-mani logot Kokborok, Assamese, Bengali aru Hindi language-ni Natural Language Processing aru Named Entity Recognition-ni upor research discussion khorok.",
        "ground_truth": [
            ("18 September 2026-ot", "MISC"),
            ("Agartala-ao", "LOC"),
            ("Tripura University-ao", "ORG"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Anupam Chakma", "PER"),
            ("Suman Reang-ni", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Bangladesh-ni", "LOC"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("Named Entity Recognition-ni", "MISC"),
        ]
    },
    {
        "id": "KB_BLIND_2",
        "title": "Sentence 2: Dharmanagar MBB University Workshop",
        "sentence": "12 January 2027-ot Dharmanagar-ao Maharaja Bir Bikram University-ao Dr. Priya Jamatia, Professor Michael Das, Rakesh Debbarma aru Nandita Reang-ni AI4Bharat, Google Research, Microsoft Research aru Indian Institute of Technology Guwahati-ni researchers-mani logot artificial intelligence, machine learning, machine translation aru Explainable AI-ni bishoyot ekta workshop-ao jog dai.",
        "ground_truth": [
            ("12 January 2027-ot", "MISC"),
            ("Dharmanagar-ao", "LOC"),
            ("Maharaja Bir Bikram University-ao", "ORG"),
            ("Dr. Priya Jamatia", "PER"),
            ("Professor Michael Das", "PER"),
            ("Rakesh Debbarma", "PER"),
            ("Nandita Reang-ni", "PER"),
            ("AI4Bharat", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("Indian Institute of Technology Guwahati-ni", "ORG"),
            ("artificial intelligence", "MISC"),
            ("machine learning", "MISC"),
            ("machine translation", "MISC"),
            ("Explainable AI-ni", "MISC"),
        ]
    },
    {
        "id": "KB_BLIND_3",
        "title": "Sentence 3: Agartala International Conference Digital Development",
        "sentence": "5 March 2027-ot Agartala-ao anusthito international conference-ao Professor Sushanta Chakma, Dr. Maya Debbarma, Rahul Jamatia, Ananya Reang aru David Noatia-ni Tripura, Meghalaya, Manipur, Mizoram aru West Bengal-ni researchers-mani logot Kokborok language-ni digital development, multilingual dataset, sentiment analysis aru information extraction-ni bishoyot research presentation dai.",
        "ground_truth": [
            ("5 March 2027-ot", "MISC"),
            ("Agartala-ao", "LOC"),
            ("Professor Sushanta Chakma", "PER"),
            ("Dr. Maya Debbarma", "PER"),
            ("Rahul Jamatia", "PER"),
            ("Ananya Reang", "PER"),
            ("David Noatia-ni", "PER"),
            ("Tripura", "LOC"),
            ("Meghalaya", "LOC"),
            ("Manipur", "LOC"),
            ("Mizoram", "LOC"),
            ("West Bengal-ni", "LOC"),
            ("Kokborok", "MISC"),
            ("multilingual dataset", "MISC"),
            ("sentiment analysis", "MISC"),
            ("information extraction-ni", "MISC"),
        ]
    },
    {
        "id": "KB_BLIND_4",
        "title": "Sentence 4: IIT Delhi Joint Project",
        "sentence": "22 April 2027-ot New Delhi-ao Indian Institute of Technology Delhi-ao Dr. Kiran Noktham, Professor Riya Debbarma, Amit Jamatia aru Priya Chakma-ni Ministry of Electronics and Information Technology, NVIDIA, Google Research aru AI4Bharat-ni representatives-mani logot low-resource Indian languages-ni karone multilingual NER, Natural Language Processing, Computer Vision aru machine translation-ni upor notun project suru khorok.",
        "ground_truth": [
            ("22 April 2027-ot", "MISC"),
            ("New Delhi-ao", "LOC"),
            ("Indian Institute of Technology Delhi-ao", "ORG"),
            ("Dr. Kiran Noktham", "PER"),
            ("Professor Riya Debbarma", "PER"),
            ("Amit Jamatia", "PER"),
            ("Priya Chakma-ni", "PER"),
            ("Ministry of Electronics and Information Technology", "ORG"),
            ("NVIDIA", "ORG"),
            ("Google Research", "ORG"),
            ("AI4Bharat-ni", "ORG"),
            ("Indian", "MISC"),
            ("multilingual NER", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("Computer Vision", "MISC"),
            ("machine translation-ni", "MISC"),
        ]
    },
    {
        "id": "KB_BLIND_5",
        "title": "Sentence 5: Agartala Large-Scale Multilingual Research Results",
        "sentence": "10 June 2027-ot Agartala-ao Tripura University-ao Professor Arindam Debbarma, Dr. Nisha Jamatia, Suman Reang, Riya Chakma aru Michael Das-ni Tripura, Assam, Meghalaya, Manipur, Mizoram aru Bangladesh-ni researchers-mani logot University of Dhaka, AI4Bharat, Google Research, Microsoft Research aru Indian Institute of Technology Madras-ni collaboration-ao Kokborok, Assamese, Bengali, Hindi, Tamil aru Telugu language-ni multilingual Named Entity Recognition, Machine Translation, Attention Attribution aru Explainable Artificial Intelligence-ni upor large-scale research project-ni results presentation dai.",
        "ground_truth": [
            ("10 June 2027-ot", "MISC"),
            ("Agartala-ao", "LOC"),
            ("Tripura University-ao", "ORG"),
            ("Professor Arindam Debbarma", "PER"),
            ("Dr. Nisha Jamatia", "PER"),
            ("Suman Reang", "PER"),
            ("Riya Chakma", "PER"),
            ("Michael Das-ni", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Meghalaya", "LOC"),
            ("Manipur", "LOC"),
            ("Mizoram", "LOC"),
            ("Bangladesh-ni", "LOC"),
            ("University of Dhaka", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("Indian Institute of Technology Madras-ni", "ORG"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Tamil", "MISC"),
            ("Telugu", "MISC"),
            ("multilingual Named Entity Recognition", "MISC"),
            ("Machine Translation", "MISC"),
            ("Attention Attribution", "MISC"),
            ("Explainable Artificial Intelligence-ni", "MISC"),
        ]
    },
]

def clean_tok(t):
    import re
    return re.sub(r'-(?:ao|ot|ni|o|bai|rok|rog|no|le|de|bo|se)$', '', t).strip().lower()

def run_blind_test():
    print("=" * 80)
    print("BLIND TEST: 5 KOKBOROK SENTENCES (ZERO CODE MODIFICATION)")
    print("=" * 80)

    predictor = get_predictor("kokborok")
    
    total_gt = 0
    total_pred = 0
    total_matched = 0
    
    detailed_reports = []

    for item in SENTENCES:
        sid = item["id"]
        title = item["title"]
        sent = item["sentence"]
        gt = item["ground_truth"]
        total_gt += len(gt)

        print(f"\n[{sid}] {title}")
        print(f"Sentence: \"{sent}\"\n")
        print(f"Expected Ground Truth ({len(gt)} entities):")
        for text, tag in gt:
            print(f"  - [{tag}] {text}")

        res = predictor.predict(sent, include_attribution=False)
        preds = res["entities"]
        total_pred += len(preds)

        print(f"\nRaw Model Predictions ({len(preds)} entities):")
        for p in preds:
            src = p.get("source", "neural")
            print(f"  + [{p['type']}] \"{p['text']}\" (chars {p['start_char']}-{p['end_char']}, tokens {p['start_token']}-{p['end_token']}) [source: {src}]")

        # Evaluate matches, missed, and misclassified
        matched = []
        missed = []
        misclassified = []

        used_preds = set()

        for exp_text, exp_tag in gt:
            clean_exp = clean_tok(exp_text)
            found = False
            for idx, p in enumerate(preds):
                if idx in used_preds:
                    continue
                clean_p = clean_tok(p["text"])
                # Check overlap / string match
                if clean_exp == clean_p or clean_exp in clean_p or clean_p in clean_exp:
                    if exp_tag == p["type"]:
                        matched.append((exp_text, exp_tag, p["text"]))
                        used_preds.add(idx)
                        found = True
                        break
                    else:
                        misclassified.append((exp_text, exp_tag, p["text"], p["type"]))
                        used_preds.add(idx)
                        found = True
                        break
            if not found:
                missed.append((exp_text, exp_tag))

        spurious = [preds[i] for i in range(len(preds)) if i not in used_preds]

        total_matched += len(matched)
        rec = (len(matched) / len(gt)) * 100 if gt else 0
        prec = (len(matched) / len(preds)) * 100 if preds else 0

        print(f"\nSummary for {sid}:")
        print(f"  Matched: {len(matched)} / {len(gt)} ({rec:.1f}% Recall | {prec:.1f}% Precision)")
        if missed:
            print(f"  Missed ({len(missed)}): {missed}")
        if misclassified:
            print(f"  Misclassified ({len(misclassified)}): {misclassified}")
        if spurious:
            print(f"  Extra/Spurious ({len(spurious)}): {[p['text'] + ' (' + p['type'] + ')' for p in spurious]}")

        detailed_reports.append({
            "id": sid,
            "title": title,
            "sentence": sent,
            "expected_count": len(gt),
            "predicted_count": len(preds),
            "matched_count": len(matched),
            "recall": round(rec, 1),
            "precision": round(prec, 1),
            "matched": matched,
            "missed": missed,
            "misclassified": misclassified,
            "spurious": spurious,
            "raw_predictions": preds
        })

    overall_recall = (total_matched / total_gt) * 100 if total_gt else 0
    overall_prec = (total_matched / total_pred) * 100 if total_pred else 0

    print("\n" + "=" * 80)
    print("OVERALL BLIND TEST RESULTS:")
    print(f"Total Expected Ground Truth Entities: {total_gt}")
    print(f"Total Raw Model Predictions:         {total_pred}")
    print(f"Total Correctly Matched Entities:    {total_matched}")
    print(f"Overall Generalization Recall:       {overall_recall:.1f}%")
    print(f"Overall Generalization Precision:    {overall_prec:.1f}%")
    print("=" * 80)

    out_file = PROJECT_ROOT / "reports" / "blind_test_kokborok_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(detailed_reports, f, ensure_ascii=False, indent=2)
    print(f"\n[REPORT SAVED] {out_file}")

if __name__ == "__main__":
    run_blind_test()
