"""
Automated 10 New Kokborok Sentence NER Evaluation Suite.
Performs dual test: Human linguistic ground truth vs model extraction.
Saves detailed JSON report of results, misses, and attributions.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from deployment.inference import get_predictor

DATASET_10 = [
    {
        "id": "KOK_S1",
        "title": "Sentence 1: MBB University Research Discussion",
        "sentence": "12 September 2026-ot Agartala-ao Maharaja Bir Bikram University-ao Professor Debashis Jamatia, Dr. Rina Debbarma, Dr. Anupam Chakma aru Professor Michael Das-ni logot Tripura, Assam, Mizoram, Manipur aru Bangladesh-ni researchers-mani AI4Bharat, Indian Institute of Technology Guwahati, University of Dhaka aru Tripura University-ni joint research project-ot Kokborok, Assamese, Bengali aru Hindi language-ni Named Entity Recognition aru Natural Language Processing-ni upor discussion khorok.",
        "expected_entities": [
            ("12 September 2026", "MISC"),
            ("Agartala", "LOC"),
            ("Maharaja Bir Bikram University", "ORG"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Professor Michael Das", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Manipur", "LOC"),
            ("Bangladesh", "LOC"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Tripura University", "ORG"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Natural Language Processing", "MISC"),
        ]
    },
    {
        "id": "KOK_S2",
        "title": "Sentence 2: IIT Delhi International Workshop",
        "sentence": "20 October 2026-ot New Delhi-ao Indian Institute of Technology Delhi-ot Dr. Suman Debbarma, Rakesh Jamatia, Priya Chakma, Ananya Das aru Rahul Reang-ni Google Research, Microsoft Research, AI4Bharat aru Ministry of Electronics and Information Technology-ni representatives-mani logot low-resource language research-ni karone Kokborok, Hindi, Assamese, Bengali aru Tamil dataset-ni bishoyot ekta international workshop-ao jog dai.",
        "expected_entities": [
            ("20 October 2026", "MISC"),
            ("New Delhi", "LOC"),
            ("Indian Institute of Technology Delhi", "ORG"),
            ("Dr. Suman Debbarma", "PER"),
            ("Rakesh Jamatia", "PER"),
            ("Priya Chakma", "PER"),
            ("Ananya Das", "PER"),
            ("Rahul Reang", "PER"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Ministry of Electronics and Information Technology", "ORG"),
            ("Kokborok", "MISC"),
            ("Hindi", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Tamil", "MISC"),
        ]
    },
    {
        "id": "KOK_S3",
        "title": "Sentence 3: IIT Guwahati Training Program",
        "sentence": "5 November 2026-ot Guwahati-ao Indian Institute of Technology Guwahati-ot Professor Arindam Sharma, Dr. Maya Debbarma aru Dr. Bimal Jamatia-ni Assam, Tripura, Meghalaya, Manipur aru Mizoram-ni students-mani karone artificial intelligence, machine learning, computer vision, natural language processing aru explainable AI-ni upor training program suru khorok, jekhane Google, Microsoft aru AI4Bharat-ni researchers-mani practical demonstration dai.",
        "expected_entities": [
            ("5 November 2026", "MISC"),
            ("Guwahati", "LOC"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Professor Arindam Sharma", "PER"),
            ("Dr. Maya Debbarma", "PER"),
            ("Dr. Bimal Jamatia", "PER"),
            ("Assam", "LOC"),
            ("Tripura", "LOC"),
            ("Meghalaya", "LOC"),
            ("Manipur", "LOC"),
            ("Mizoram", "LOC"),
            ("artificial intelligence", "MISC"),
            ("machine learning", "MISC"),
            ("computer vision", "MISC"),
            ("natural language processing", "MISC"),
            ("explainable AI", "MISC"),
            ("Google", "ORG"),
            ("Microsoft", "ORG"),
            ("AI4Bharat", "ORG"),
        ]
    },
    {
        "id": "KOK_S4",
        "title": "Sentence 4: University of Calcutta Presentation",
        "sentence": "15 December 2026-ot Kolkata-ao University of Calcutta-ot Dr. Riya Debbarma, Professor Sushanta Chakma, Amit Reang aru Nandita Jamatia-ni Tripura University, University of Dhaka, Indian Institute of Technology Kharagpur aru AI4Bharat-ni researchers-mani logot Kokborok, Bengali, Assamese, Hindi aru Malayalam language-ni multilingual artificial intelligence model-ni development aru evaluation-ni bishoyot research presentation dai.",
        "expected_entities": [
            ("15 December 2026", "MISC"),
            ("Kolkata", "LOC"),
            ("University of Calcutta", "ORG"),
            ("Dr. Riya Debbarma", "PER"),
            ("Professor Sushanta Chakma", "PER"),
            ("Amit Reang", "PER"),
            ("Nandita Jamatia", "PER"),
            ("Tripura University", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Indian Institute of Technology Kharagpur", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Kokborok", "MISC"),
            ("Bengali", "MISC"),
            ("Assamese", "MISC"),
            ("Hindi", "MISC"),
            ("Malayalam", "MISC"),
            ("artificial intelligence", "MISC"),
        ]
    },
    {
        "id": "KOK_S5",
        "title": "Sentence 5: University of Dhaka Project Announcement",
        "sentence": "10 January 2027-ot Dhaka-ao University of Dhaka-ot Professor Michael Rahman, Dr. Anupam Chakma, Rina Debbarma, Suman Jamatia aru David Reang-ni Bangladesh, India, Tripura, Assam aru West Bengal-ni language researchers-mani logot Kokborok, Bengali, Assamese, Hindi aru English language-ni dataset collection, machine translation, sentiment analysis, named entity recognition aru explainable artificial intelligence-ni upor notun research project announce khorok.",
        "expected_entities": [
            ("10 January 2027", "MISC"),
            ("Dhaka", "LOC"),
            ("University of Dhaka", "ORG"),
            ("Professor Michael Rahman", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Rina Debbarma", "PER"),
            ("Suman Jamatia", "PER"),
            ("David Reang", "PER"),
            ("Bangladesh", "LOC"),
            ("India", "LOC"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("West Bengal", "LOC"),
            ("Kokborok", "MISC"),
            ("Bengali", "MISC"),
            ("Assamese", "MISC"),
            ("Hindi", "MISC"),
            ("English", "MISC"),
            ("machine translation", "MISC"),
            ("sentiment analysis", "MISC"),
            ("named entity recognition", "MISC"),
            ("explainable artificial intelligence", "MISC"),
        ]
    },
    {
        "id": "KOK_S6",
        "title": "Sentence 6: Indian Institute of Science Benchmark Test",
        "sentence": "25 February 2027-ot Bengaluru-ao Indian Institute of Science-ot Dr. Kavita Sharma, Professor Rajiv Kumar, Hari Debbarma, Rakesh Chakma aru Priyanka Jamatia-ni Google Research, Microsoft Research, NVIDIA aru AI4Bharat-ni engineers-mani logot low-resource Indian language-ni karone multilingual transformer model, BERT, MuRIL, IndicBERT aru natural language processing system-ni performance test khorok, aru Tripura, Assam, Manipur aru Meghalaya-ni data-ni comparison khorok.",
        "expected_entities": [
            ("25 February 2027", "MISC"),
            ("Bengaluru", "LOC"),
            ("Indian Institute of Science", "ORG"),
            ("Dr. Kavita Sharma", "PER"),
            ("Professor Rajiv Kumar", "PER"),
            ("Hari Debbarma", "PER"),
            ("Rakesh Chakma", "PER"),
            ("Priyanka Jamatia", "PER"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("NVIDIA", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Indian", "MISC"),
            ("BERT", "MISC"),
            ("MuRIL", "MISC"),
            ("IndicBERT", "MISC"),
            ("natural language processing", "MISC"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Manipur", "LOC"),
            ("Meghalaya", "LOC"),
        ]
    },
    {
        "id": "KOK_S7",
        "title": "Sentence 7: IIT Madras Evaluation",
        "sentence": "18 March 2027-ot Chennai-ao Indian Institute of Technology Madras-ot Professor Anjali Rao, Dr. Debashis Jamatia, Dr. Rina Chakma, Suman Reang aru Michael Das-ni AI4Bharat, Indian Institute of Technology Madras, Tripura University aru University of Madras-ni research team-mani logot Kokborok, Tamil, Telugu, Kannada, Malayalam aru Hindi language-ni named entity recognition system-ni accuracy, precision, recall, F1-score aru attention attribution-ni bishoyot detailed evaluation khorok.",
        "expected_entities": [
            ("18 March 2027", "MISC"),
            ("Chennai", "LOC"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("Professor Anjali Rao", "PER"),
            ("Dr. Debashis Jamatia", "PER"),
            ("Dr. Rina Chakma", "PER"),
            ("Suman Reang", "PER"),
            ("Michael Das", "PER"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("Tripura University", "ORG"),
            ("University of Madras", "ORG"),
            ("Kokborok", "MISC"),
            ("Tamil", "MISC"),
            ("Telugu", "MISC"),
            ("Kannada", "MISC"),
            ("Malayalam", "MISC"),
            ("Hindi", "MISC"),
            ("named entity recognition", "MISC"),
        ]
    },
    {
        "id": "KOK_S8",
        "title": "Sentence 8: TIFR Mumbai Framework",
        "sentence": "7 April 2027-ot Mumbai-ao Tata Institute of Fundamental Research-ot Professor Neha Sharma, Dr. Amit Debbarma, Rahul Jamatia, Priya Reang aru Ananya Chakma-ni Maharashtra, Tripura, Assam, Gujarat aru West Bengal-ni researchers-mani logot Google Research, AI4Bharat, Microsoft Research aru Tata Consultancy Services-ni collaboration-ot multilingual NLP, artificial intelligence, machine translation, information extraction aru explainable AI-ni karone notun research framework toiri khorok.",
        "expected_entities": [
            ("7 April 2027", "MISC"),
            ("Mumbai", "LOC"),
            ("Tata Institute of Fundamental Research", "ORG"),
            ("Professor Neha Sharma", "PER"),
            ("Dr. Amit Debbarma", "PER"),
            ("Rahul Jamatia", "PER"),
            ("Priya Reang", "PER"),
            ("Ananya Chakma", "PER"),
            ("Maharashtra", "LOC"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Gujarat", "LOC"),
            ("West Bengal", "LOC"),
            ("Google Research", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Microsoft Research", "ORG"),
            ("Tata Consultancy Services", "ORG"),
            ("multilingual NLP", "MISC"),
            ("artificial intelligence", "MISC"),
            ("machine translation", "MISC"),
            ("information extraction", "MISC"),
            ("explainable AI", "MISC"),
        ]
    },
    {
        "id": "KOK_S9",
        "title": "Sentence 9: University of Hyderabad Seminar",
        "sentence": "22 May 2027-ot Hyderabad-ao University of Hyderabad-ot Dr. Sushil Kumar, Professor Maya Jamatia, Riya Debbarma, Anupam Reang aru David Chakma-ni Telangana, Andhra Pradesh, Odisha, Tripura aru Assam-ni students-mani logot AI4Bharat, Indian Institute of Technology Hyderabad, University of Hyderabad aru Google Research-ni support-ao Kokborok, Telugu, Odia, Assamese aru Hindi language-ni dataset, pretrained model, NER annotation, attention visualization aru explainability experiment-ni bishoyot international seminar-ao presentation dai.",
        "expected_entities": [
            ("22 May 2027", "MISC"),
            ("Hyderabad", "LOC"),
            ("University of Hyderabad", "ORG"),
            ("Dr. Sushil Kumar", "PER"),
            ("Professor Maya Jamatia", "PER"),
            ("Riya Debbarma", "PER"),
            ("Anupam Reang", "PER"),
            ("David Chakma", "PER"),
            ("Telangana", "LOC"),
            ("Andhra Pradesh", "LOC"),
            ("Odisha", "LOC"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Hyderabad", "ORG"),
            ("University of Hyderabad", "ORG"),
            ("Google Research", "ORG"),
            ("Kokborok", "MISC"),
            ("Telugu", "MISC"),
            ("Odia", "MISC"),
            ("Assamese", "MISC"),
            ("Hindi", "MISC"),
            ("NER", "MISC"),
        ]
    },
    {
        "id": "KOK_S10",
        "title": "Sentence 10: Agartala Large-Scale Research Conference (Heavy Stress Test)",
        "sentence": "30 June 2027-ot Agartala-ao Tripura University-ot Professor Debashis Jamatia, Dr. Rina Debbarma, Dr. Anupam Chakma, Professor Michael Das, Suman Reang aru Priya Jamatia-ni Tripura, Assam, Mizoram, Manipur, Meghalaya, West Bengal, Bangladesh aru New Delhi-ni researchers-mani logot AI4Bharat, Indian Institute of Technology Guwahati, Indian Institute of Technology Madras, University of Dhaka, Google Research, Microsoft Research aru Ministry of Electronics and Information Technology-ni collaboration-ot Kokborok, Assamese, Bengali, Hindi, Tamil, Telugu, Kannada aru Malayalam language-ni multilingual Named Entity Recognition, Natural Language Processing, machine translation, attention attribution aru explainable artificial intelligence-ni upor ekta large-scale research conference-ao organize khorok.",
        "expected_entities": [
            ("30 June 2027", "MISC"),
            ("Agartala", "LOC"),
            ("Tripura University", "ORG"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Professor Michael Das", "PER"),
            ("Suman Reang", "PER"),
            ("Priya Jamatia", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Manipur", "LOC"),
            ("Meghalaya", "LOC"),
            ("West Bengal", "LOC"),
            ("Bangladesh", "LOC"),
            ("New Delhi", "LOC"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("Ministry of Electronics and Information Technology", "ORG"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Tamil", "MISC"),
            ("Telugu", "MISC"),
            ("Kannada", "MISC"),
            ("Malayalam", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("machine translation", "MISC"),
            ("explainable artificial intelligence", "MISC"),
        ]
    },
]


def clean_text(t: str) -> str:
    import re
    # Strip common Kokborok and Indic clitics for fuzzy comparison
    t = re.sub(r'-(?:ao|ot|ni|o|bai|rok|rog|no|le|de|bo|se)$', '', t.strip())
    # Strip honorifics
    t = re.sub(r'^(?:Dr\.?|Prof\.?|Professor|Mr\.?|Mrs\.?|Ms\.)\s+', '', t, flags=re.I)
    return t.strip().lower()


def evaluate_10():
    print("=" * 80)
    print("10 NEW KOKBOROK SENTENCES NER BENCHMARK: HUMAN VS MODEL EVALUATION")
    print("=" * 80)

    predictor = get_predictor("kokborok")
    all_results = []
    total_expected = 0
    total_matched = 0

    for item in DATASET_10:
        sid = item["id"]
        title = item["title"]
        sent = item["sentence"]
        expected = item["expected_entities"]
        total_expected += len(expected)

        print(f"\n[{sid}] {title}")
        print(f"  * Expected: {len(expected)} entities")

        res = predictor.predict(sent, include_attribution=True)
        pred_entities = res["entities"]
        print(f"  * Model Extracted: {len(pred_entities)} entities")

        matched = []
        missed = []

        for exp_text, exp_type in expected:
            clean_exp = clean_text(exp_text)
            found = False
            for p in pred_entities:
                clean_pred = clean_text(p["text"])
                if (clean_exp == clean_pred or clean_exp in clean_pred or clean_pred in clean_exp) and exp_type == p["type"]:
                    matched.append((exp_text, exp_type, p["text"]))
                    found = True
                    break
            if not found:
                missed.append((exp_text, exp_type))

        total_matched += len(matched)
        recall = (len(matched) / len(expected) * 100) if expected else 0.0
        prec = (len(matched) / len(pred_entities) * 100) if pred_entities else 0.0

        print(f"  * Correctly Matched: {len(matched)} entities ({recall:.1f}% Recall | {prec:.1f}% Precision)")
        if missed:
            print(f"  * Missed Entities ({len(missed)}): {[(m[0], m[1]) for m in missed]}")
        else:
            print("  * Missed Entities: 0 (FLAWLESS 100% RECOVERY)")

        all_results.append({
            "id": sid,
            "title": title,
            "sentence": sent,
            "expected_count": len(expected),
            "predicted_count": len(pred_entities),
            "matched_count": len(matched),
            "missed": missed,
            "recall": round(recall, 1),
            "precision": round(prec, 1),
            "predicted_entities": pred_entities,
        })

    overall_recall = (total_matched / total_expected * 100) if total_expected else 0.0
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: {total_matched} / {total_expected} ({overall_recall:.1f}% Recall)")
    print("=" * 80)

    out_file = PROJECT_ROOT / "reports" / "10_new_kokborok_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[REPORT SAVED] Results written to {out_file}")


if __name__ == "__main__":
    evaluate_10()
