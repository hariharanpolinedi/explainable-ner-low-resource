"""
Deep search and analysis script to find all organization occurrences across the 20,300 Kokborok sentences.
"""

import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / 'data_collection/unified_public_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Load test sentences to verify 0.00% leakage
with open(PROJECT_ROOT / 'data/gold/kokborok/test.conll', 'r', encoding='utf-8') as f:
    test_lines = f.readlines()
test_text = set()
curr = []
for l in test_lines:
    l = l.strip()
    if not l:
        if curr:
            test_text.add(' '.join(curr))
            curr = []
    else:
        curr.append(l.split()[0])
if curr:
    test_text.add(' '.join(curr))

print(f"Loaded {len(corpus)} total raw records. {len(test_text)} test sentences.")

# Comprehensive dictionary of organization search patterns
org_regexes = [
    # Explicit Named Organizations
    (r'\bIndian National Congress\b', 'Indian National Congress', 'government/political', 'High'),
    (r'\bScheduled Castes Federation\b', 'Scheduled Castes Federation', 'NGO/social/political', 'High'),
    (r'\bColumbia University\b', 'Columbia University', 'education', 'High'),
    (r'\bUniversity of Cambridge\b', 'University of Cambridge', 'education', 'High'),
    (r'\bUniversity of Oxford\b', 'University of Oxford', 'education', 'High'),
    (r'\bHarvard University\b', 'Harvard University', 'education', 'High'),
    (r'\bWorld Health Organization\b', 'World Health Organization', 'healthcare/NGO', 'High'),
    (r'\bWHO\b', 'WHO', 'healthcare/NGO', 'High'),
    (r'\bUnited Nations\b', 'United Nations', 'government/international', 'High'),
    (r'\bUNICEF\b', 'UNICEF', 'NGO/international', 'High'),
    (r'\bUNESCO\b', 'UNESCO', 'cultural/international', 'High'),
    (r'\bNASA\b', 'NASA', 'government/science', 'High'),
    (r'\bISRO\b', 'ISRO', 'government/science', 'High'),
    (r'\bGoogle\b', 'Google', 'company', 'High'),
    (r'\bMicrosoft\b', 'Microsoft', 'company', 'High'),
    (r'\bApple\b', 'Apple', 'company', 'High'),
    (r'\bAmazon\b', 'Amazon', 'company', 'High'),
    (r'\bTata Technologies\b', 'Tata Technologies', 'company', 'High'),
    (r'\bTata\b', 'Tata', 'company', 'High'),
    (r'\bDoordarshan\b', 'Doordarshan', 'media', 'High'),
    (r'\bDD news\b', 'DD news', 'media', 'High'),
    (r'\bAll India Radio\b', 'All India Radio', 'media', 'High'),
    (r'\bAIR\b', 'AIR', 'media', 'High'),
    (r'\bBBC\b', 'BBC', 'media', 'High'),
    (r'\bCNN\b', 'CNN', 'media', 'High'),
    (r'\bTripura University\b', 'Tripura University', 'education', 'High'),
    (r'\bTripura University ni\b', 'Tripura University', 'education', 'High'),
    (r'\bTripura board\b', 'Tripura board', 'government/education', 'High'),
    (r'\bTripura Board\b', 'Tripura Board', 'government/education', 'High'),
    (r'\bTripura Police\b', 'Tripura Police', 'government/police', 'High'),
    (r'\bTripura police\b', 'Tripura police', 'government/police', 'High'),
    (r'\bTripura channel\b', 'Tripura channel', 'media', 'High'),
    (r'\bTripura Hukbar Bedek\b', 'Tripura Hukbar Bedek', 'media/government', 'High'),
    (r'\bTripura Gramin Bank\b', 'Tripura Gramin Bank', 'company/banking', 'High'),
    (r'\bTripura Sorkar\b', 'Tripura Sorkar', 'government', 'High'),
    (r'\bTripura sorkar\b', 'Tripura sorkar', 'government', 'High'),
    (r'\bTTAADC\b', 'TTAADC', 'government/tribal council', 'High'),
    (r'\bTSF\b', 'TSF', 'NGO/student federation', 'High'),
    (r'\bTSR\b', 'TSR', 'government/police', 'High'),
    (r'\bTBSE\b', 'TBSE', 'government/education', 'High'),
    (r'\bMBB College\b', 'MBB College', 'education', 'High'),
    (r'\bMBB University\b', 'MBB University', 'education', 'High'),
    (r'\bNIT Agartala\b', 'NIT Agartala', 'education', 'High'),
    (r'\bTIT\b', 'TIT', 'education', 'High'),
    (r'\bITIs\b', 'ITIs', 'education', 'High'),
    (r'\bITI\b', 'ITI', 'education', 'High'),
    (r'\bONGC\b', 'ONGC', 'company/energy', 'High'),
    (r'\bNEEPCO\b', 'NEEPCO', 'company/energy', 'High'),
    (r'\bKokborok Tei Hukumu Mission\b', 'Kokborok Tei Hukumu Mission', 'cultural/NGO', 'High'),
    (r'\bKokborok Sahitya Sabha\b', 'Kokborok Sahitya Sabha', 'cultural/NGO', 'High'),
    (r'\bTripura Baptist Christian Union\b', 'Tripura Baptist Christian Union', 'NGO/cultural', 'High'),
    (r'\bTBCU\b', 'TBCU', 'NGO/cultural', 'High'),
    (r'\bAgartala Municipal Corporation\b', 'Agartala Municipal Corporation', 'government', 'High'),
    (r'\bChini Khorang\b', 'Chini Khorang', 'media/cultural', 'High'),
    (r'\bHukbar Bedek\b', 'Hukbar Bedek', 'media/government', 'High'),
    (r'\bKokborok Department\b', 'Kokborok Department', 'education', 'High'),
    (r'\bKokborok department\b', 'Kokborok department', 'education', 'High'),
    (r'\bState Bank of India\b', 'State Bank of India', 'company/banking', 'High'),
    (r'\bSBI\b', 'SBI', 'company/banking', 'High'),
    (r'\bReserve Bank of India\b', 'Reserve Bank of India', 'government/banking', 'High'),
    (r'\bRBI\b', 'RBI', 'government/banking', 'High'),
    (r'\bSupreme Court\b', 'Supreme Court', 'government/judiciary', 'High'),
    (r'\bHigh Court\b', 'High Court', 'government/judiciary', 'High'),
    (r'\bHigh Court of Tripura\b', 'High Court of Tripura', 'government/judiciary', 'High'),
    (r'\bTripura High Court\b', 'Tripura High Court', 'government/judiciary', 'High'),
    (r'\bRed Cross\b', 'Red Cross', 'NGO/healthcare', 'High'),
    (r'\bFIFA\b', 'FIFA', 'sports', 'High'),
    (r'\bICC\b', 'ICC', 'sports', 'High'),
    (r'\bBCCI\b', 'BCCI', 'sports', 'High'),
    (r'\bTripura Cricket Association\b', 'Tripura Cricket Association', 'sports', 'High'),
    (r'\bTripura Football Association\b', 'Tripura Football Association', 'sports', 'High'),
    (r'\bTCA\b', 'TCA', 'sports', 'High'),
    (r'\bTFA\b', 'TFA', 'sports', 'High'),
    (r'\bCongress\b', 'Congress', 'government/political', 'High'),
    (r'\bBJP\b', 'BJP', 'government/political', 'High'),
    (r'\bCPIM\b', 'CPIM', 'government/political', 'High'),
    (r'\bCPI\(M\)\b', 'CPI(M)', 'government/political', 'High'),
    (r'\bIPFT\b', 'IPFT', 'government/political', 'High'),
    (r'\bINPT\b', 'INPT', 'government/political', 'High'),
    (r'\bTIPRA Motha\b', 'TIPRA Motha', 'government/political', 'High'),
    (r'\bTipra Motha\b', 'Tipra Motha', 'government/political', 'High'),
    (r'\bTMP\b', 'TMP', 'government/political', 'High'),
]

# Let's also find all occurrences of institutional patterns
inst_regex = re.compile(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Za-z]+){0,3}\s+(?:University|College|School|Board|Department|Police|Hospital|Mission|Council|Committee|Association|Federation|Bank|Company|Agency|Office|Samiti|Sangha|Sabha|Party|Union|Club|Institute|Academy|Bedek|Corporation|Society))\b', re.IGNORECASE)

print("Starting scan...")
detected = []
seen_sents = set()

for item in corpus:
    k = item['kok'].strip()
    if not k or k in seen_sents:
        continue
    seen_sents.add(k)
    
    # Leakage check
    is_test = k in test_text
    
    # 1. Match specific dictionary
    for pattern, name, cat, conf in org_regexes:
        m = re.search(pattern, k)
        if m:
            detected.append({
                'source': item['source'],
                'source_url': item['source_url'],
                'license': item['license'],
                'doc_id': item['doc_id'],
                'sent_id': item['sent_id'],
                'kok': k,
                'en': item.get('en', ''),
                'matched_span': m.group(0),
                'canonical_org': name,
                'org_type': cat,
                'confidence': conf,
                'in_test': is_test
            })

print(f"Total raw candidate mentions found: {len(detected)}")
print(f"In test set: {sum(1 for d in detected if d['in_test'])}")

# Analyze canonical org frequencies
freq = Counter(d['canonical_org'] for d in detected if not d['in_test'])
print("\nTop detected organizations (non-test):")
for org, count in freq.most_common(30):
    print(f"  {org}: {count}")
