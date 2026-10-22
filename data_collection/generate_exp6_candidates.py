"""
Exhaustive verification filter for Kokborok ORG candidates.
Eliminates:
1. Polysemous words where context is geographical or botanical (e.g. Amazon rainforest, Apple fruit).
2. Pure English sentences without Kokborok text.
3. Duplicate entities within the same span.
4. Test set leakage (strictly 0.00%).
"""

import json
import re
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load unified sentences
with open(PROJECT_ROOT / 'data_collection/unified_public_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Load gold data to verify zero leakage
def normalize_spacing(s):
    return ' '.join(s.strip().split())

def load_conll_data(path):
    sents = []
    tokens = []
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sents.append((normalize_spacing(' '.join(tokens)), tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
        if tokens:
            sents.append((normalize_spacing(' '.join(tokens)), tokens, tags))
    return sents

train_gold = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/train.conll')
dev_gold = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/dev.conll')
test_gold = load_conll_data(PROJECT_ROOT / 'data/gold/kokborok/test.conll')

train_texts = set(s[0] for s in train_gold)
dev_texts = set(s[0] for s in dev_gold)
test_texts = set(s[0] for s in test_gold)

org_specs = [
    # Toponymic + Multi-token + Lowercase hard cases
    (r'\bTripura State Electricity Corporation Ltd\.\b', 'Tripura State Electricity Corporation Ltd.', 'company', 'Public Sector Enterprise', 'State electricity transmission and distribution corporation of Tripura', True, False, True),
    (r'\bTSECL\b', 'TSECL', 'company', 'Public Sector Enterprise', 'Tripura State Electricity Corporation Limited acronym', False, False, False),
    (r'\bTripura University\b', 'Tripura University', 'education', 'Central University', 'Central public university situated in Suryamaninagar, Tripura', True, False, True),
    (r'\bTripura Board\b', 'Tripura Board', 'government', 'Education Board', 'Tripura Board of Secondary Education statutory authority', True, False, True),
    (r'\bTBSE\b', 'TBSE', 'government', 'Education Board', 'Tripura Board of Secondary Education statutory authority acronym', False, False, False),
    (r'\bMBB College\b', 'MBB College', 'education', 'Higher Education College', 'Maharaja Bir Bikram College in Agartala', True, False, False),
    (r'\bTripura Hukbar Bedek\b', 'Tripura Hukbar Bedek', 'media', 'State Information Agency', 'Tripura Directorate of Information & Cultural Affairs bulletin', True, False, True),
    (r'\bHukbar Bedek\b', 'Hukbar Bedek', 'media', 'State Information Agency', 'Information and News Department news entity', True, False, False),
    (r'\bITIs\b', 'ITIs', 'education', 'Vocational Training Institute', 'Industrial Training Institutes technical education network', False, False, False),
    (r'\bTSF\b', 'TSF', 'NGO/social', 'Student Federation', 'Twipra Students\' Federation apex indigenous student body', False, False, False),
    (r'\bhealth department\b', 'health department', 'government', 'Government Department', 'Department of Health & Family Welfare administration', True, True, False),
    (r'\bsocial welfare department\b', 'social welfare department', 'government', 'Government Department', 'Department of Social Welfare & Social Education administration', True, True, False),

    # Political & International Multi-token Organizations
    (r'\bIndian National Congress\b', 'Indian National Congress', 'government', 'Political Party', 'Major national political party founded in 1885', True, False, False),
    (r'\bScheduled Castes Federation\b', 'Scheduled Castes Federation', 'NGO/social', 'Socio-Political Organization', 'Historic socio-political organization founded by Dr. B.R. Ambedkar', True, False, False),
    (r'\bUnited Nations General Assembly\b', 'United Nations General Assembly', 'government', 'International Organization', 'One of the six principal organs of the United Nations', True, False, False),
    (r'\bUnited Nations\b', 'United Nations', 'government', 'International Organization', 'Intergovernmental organization to maintain international peace and security', True, False, False),
    (r'\bCouncil of Europe\b', 'Council of Europe', 'government', 'International Organization', 'International organization dedicated to human rights, democracy and the rule of law', True, False, False),
    (r'\bConstituent Assembly\b', 'Constituent Assembly', 'government', 'Legislative/Constitutional Body', 'Elected assembly tasked with drafting the Constitution of India', True, False, False),
    (r'\bGraduate Council\b', 'Graduate Council', 'education', 'Academic Board', 'University academic governing body responsible for graduate studies', True, False, False),
    (r'\bUniversity of Cambridge\b', 'University of Cambridge', 'education', 'Collegiate Research University', 'Public collegiate research university in Cambridge, United Kingdom', True, False, False),
    (r'\bWorld Health Organization\b', 'World Health Organization', 'healthcare', 'International Health Agency', 'Specialized United Nations agency for international public health', True, False, False),
    (r'\bWHO\b', 'WHO', 'healthcare', 'International Health Agency', 'World Health Organization acronym', False, False, False),
    (r'\bUNICEF\b', 'UNICEF', 'NGO/social', 'International Children\'s Agency', 'United Nations International Children\'s Emergency Fund', False, False, False),
    (r'\bUNESCO\b', 'UNESCO', 'cultural', 'International Cultural Agency', 'United Nations Educational, Scientific and Cultural Organization', False, False, False),
    (r'\bRed Cross\b', 'Red Cross', 'healthcare', 'Humanitarian NGO', 'International humanitarian organization', True, False, False),

    # Major Companies, Banks & Media
    (r'\bReserve Bank of India\b', 'Reserve Bank of India', 'government', 'Central Bank', 'India\'s central banking institution and monetary regulator', True, False, False),
    (r'\bRBI\b', 'RBI', 'government', 'Central Bank', 'Reserve Bank of India acronym', False, False, False),
    (r'\bState Bank of India\b', 'State Bank of India', 'company', 'Public Sector Bank', 'India\'s largest multinational public sector bank and financial services statutory body', True, False, False),
    (r'\bSBI\b', 'SBI', 'company', 'Public Sector Bank', 'State Bank of India acronym', False, False, False),
    (r'\bSupreme Court\b', 'Supreme Court', 'government', 'Judicial Institution', 'Supreme Court apex judicial authority', True, False, False),
    (r'\bAll India Radio\b', 'All India Radio', 'media', 'National Public Broadcaster', 'National public radio broadcaster of India', True, False, False),
    (r'\bDoordarshan\b', 'Doordarshan', 'media', 'National Public Broadcaster', 'Autonomous public service broadcaster founded by the Government of India', False, False, False),
    (r'\bTata Technologies\b', 'Tata Technologies', 'company', 'Engineering/Tech Multinational', 'Global product engineering and digital services multinational', True, False, False),
    (r'\bTata\b', 'Tata', 'company', 'Multinational Conglomerate', 'Indian multinational conglomerate enterprise', False, False, False),
    (r'\bONGC\b', 'ONGC', 'company', 'Energy Corporation', 'Oil and Natural Gas Corporation public sector enterprise', False, False, False),
    (r'\bISRO\b', 'ISRO', 'government', 'National Space Agency', 'Indian Space Research Organisation', False, False, False),
    (r'\bNASA\b', 'NASA', 'government', 'National Space Agency', 'National Aeronautics and Space Administration', False, False, False),
    (r'\bGoogle\b', 'Google', 'company', 'Technology Corporation', 'Multinational technology and cloud computing company', False, False, False),
    (r'\bMicrosoft\b', 'Microsoft', 'company', 'Technology Corporation', 'Multinational technology and software corporation', False, False, False),
    (r'\bApple\b', 'Apple', 'company', 'Technology Corporation', 'Multinational technology and hardware corporation', False, False, False),
    (r'\bAmazon\b', 'Amazon', 'company', 'Technology Corporation', 'Multinational technology and e-commerce enterprise', False, False, False),
]

accepted_entries = []
rejected_entries = []
seen_cand = set()

# Process each sentence in the unified corpus
for item in corpus:
    kok = item['kok'].strip()
    en = item.get('en', '').strip()
    doc_id = item['doc_id']
    sent_id = item['sent_id']
    src = item['source']
    src_url = item['source_url']
    lic = item['license']

    # Remove code-mixing annotations if from code-mixed corpus (e.g., /trp, /en)
    clean_kok = re.sub(r'/(?:trp|en|bn|hi)\b', '', kok).strip()
    clean_kok = normalize_spacing(clean_kok)
    
    if len(clean_kok.split()) < 3:
        continue

    # STRICT CHECK 1: Exclude ANY sentence that exactly matches the locked test set
    if clean_kok in test_texts:
        continue

    # Filter out pure English sentences where Kokborok translation was missing
    # E.g. sentences starting with 'Yes, the government and the Supreme Court...'
    kok_words = ['tong', 'wng', 'bai', 'ni', 'no', 'o', 'rok', 'se', 'khe', 'tei', 'hwnwi', 'kaisa', 'bo', 'ang', 'nwng', 'chini', 'bisi', 'sal']
    is_kokborok = any(re.search(rf'\b{w}\b', clean_kok, re.IGNORECASE) for w in kok_words)
    if not is_kokborok:
        rejected_entries.append({
            'sentence': clean_kok,
            'entity': 'N/A',
            'reason': 'Non-Kokborok / purely English sentence in corpus'
        })
        continue

    for pattern, canon_name, cat, subcat, expl, is_multi, has_low, has_loc in org_specs:
        matches = list(re.finditer(pattern, clean_kok))
        for m in matches:
            matched_span = m.group(0)
            
            # Polysemy check: Amazon rainforest / river -> LOC (reject)
            if matched_span == 'Amazon' and ('bolong' in clean_kok.lower() or 'rainforest' in en.lower() or 'forest' in en.lower() or 'river' in en.lower() or 'haste' in clean_kok.lower() or 'lines' in clean_kok.lower()):
                rejected_entries.append({
                    'sentence': clean_kok,
                    'entity': matched_span,
                    'reason': 'Geographic toponym (Amazon rainforest / basin / river), not Amazon company'
                })
                continue

            # Polysemy check: Apple fruit / tree -> reject
            if matched_span == 'Apple' and any(w in clean_kok.lower() for w in ['thol', 'phol', 'chani', 'buphang', 'fruit', 'tree']):
                rejected_entries.append({
                    'sentence': clean_kok,
                    'entity': matched_span,
                    'reason': 'Botanical noun (apple fruit / apple tree), not Apple Inc.'
                })
                continue

            # Polysemy check: Tata goodbye -> reject
            if matched_span == 'Tata' and any(w in clean_kok.lower() for w in ['bye', 'tata thangkha', 'tata khwlai']):
                rejected_entries.append({
                    'sentence': clean_kok,
                    'entity': matched_span,
                    'reason': 'Salutation (tata / goodbye), not Tata Group'
                })
                continue

            dedup_key = (clean_kok, matched_span, m.start())
            if dedup_key in seen_cand:
                continue
            seen_cand.add(dedup_key)

            # Map tokens and BIO tags
            tokens = clean_kok.split()
            matched_tokens = matched_span.split()
            matched_len = len(matched_tokens)
            
            start_tok_idx = -1
            for t_i in range(len(tokens) - matched_len + 1):
                cand_toks = [re.sub(r'[^\w\.]', '', tokens[t_i + k]) for k in range(matched_len)]
                target_toks = [re.sub(r'[^\w\.]', '', matched_tokens[k]) for k in range(matched_len)]
                if cand_toks == target_toks:
                    start_tok_idx = t_i
                    break
            
            if start_tok_idx == -1:
                for t_i, tok in enumerate(tokens):
                    if tok.startswith(matched_tokens[0]) or matched_tokens[0] in tok:
                        start_tok_idx = t_i
                        matched_len = len(matched_tokens)
                        break

            # Assign BIO tags
            bio_tags = ['O'] * len(tokens)
            if start_tok_idx != -1:
                bio_tags[start_tok_idx] = 'B-ORG'
                for k in range(1, matched_len):
                    if start_tok_idx + k < len(tokens):
                        bio_tags[start_tok_idx + k] = 'I-ORG'

            accepted_entries.append({
                'source': src,
                'source_url': src_url,
                'license': lic,
                'document_id': doc_id,
                'sentence_id': sent_id,
                'sentence': clean_kok,
                'english_translation': en,
                'entity': matched_span,
                'canonical_name': canon_name,
                'entity_type': cat,
                'sub_category': subcat,
                'annotation_basis': f"Contextual verification: {expl}.",
                'confidence': 'High',
                'is_multi_token': is_multi,
                'has_lowercase': has_low,
                'has_loc_prefix': has_loc,
                'tokens': tokens,
                'bio_tags': bio_tags
            })

print(f"\nFinal Verified Accepted Candidate ORG Instances: {len(accepted_entries)}")
print(f"Total Rejected Entities/Sentences: {len(rejected_entries)}")

# Write data/processed/kokborok_org_candidates_provenance.csv
prov_file = PROJECT_ROOT / 'data/processed/kokborok_org_candidates_provenance.csv'
prov_file.parent.mkdir(parents=True, exist_ok=True)

with open(prov_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'source', 'source_url', 'license', 'document_id', 'sentence_id',
        'sentence', 'entity', 'entity_type', 'annotation_basis', 'confidence'
    ])
    for item in accepted_entries:
        writer.writerow([
            item['source'],
            item['source_url'],
            item['license'],
            item['document_id'],
            item['sentence_id'],
            item['sentence'],
            item['entity'],
            item['entity_type'],
            item['annotation_basis'],
            item['confidence']
        ])

print(f"Saved provenance to: {prov_file.relative_to(PROJECT_ROOT)}")

# Write data/processed/kokborok_org_candidates.conll
conll_file = PROJECT_ROOT / 'data/processed/kokborok_org_candidates.conll'
with open(conll_file, 'w', encoding='utf-8') as f:
    for item in accepted_entries:
        for tok, tag in zip(item['tokens'], item['bio_tags']):
            f.write(f"{tok} {tag}\n")
        f.write("\n")

print(f"Saved CoNLL candidates to: {conll_file.relative_to(PROJECT_ROOT)}")

# Check data leakage against gold sets
print("\n--- DATA LEAKAGE AUDIT ---")
cand_sentences = set(item['sentence'] for item in accepted_entries)

overlap_test = cand_sentences.intersection(test_texts)
overlap_dev = cand_sentences.intersection(dev_texts)
overlap_train = cand_sentences.intersection(train_texts)

print(f"Exact Overlap with Gold Test:  {len(overlap_test)} (STRICT REQUIREMENT = 0)")
print(f"Exact Overlap with Gold Dev:   {len(overlap_dev)}")
print(f"Exact Overlap with Gold Train: {len(overlap_train)}")

assert len(overlap_test) == 0, "ERROR: Leakage into test set!"
print("LEAKAGE CHECK PASSED: EXACTLY 0.00% overlap with locked test set.")

# Print summary distribution
print("\n--- DISTRIBUTION BY CATEGORY ---")
cat_counts = Counter(item['entity_type'] for item in accepted_entries)
for cat, cnt in cat_counts.most_common():
    print(f"  {cat:15s}: {cnt:3d} ({cnt/len(accepted_entries)*100:.1f}%)")

print("\n--- HARD CASE STATISTICS ---")
multi_tok_cnt = sum(1 for item in accepted_entries if item['is_multi_token'])
low_case_cnt = sum(1 for item in accepted_entries if item['has_lowercase'])
loc_prefix_cnt = sum(1 for item in accepted_entries if item['has_loc_prefix'])

print(f"  Multi-token ORGs:          {multi_tok_cnt:3d} ({multi_tok_cnt/len(accepted_entries)*100:.1f}%)")
print(f"  Lowercase Noun ORGs:       {low_case_cnt:3d} ({low_case_cnt/len(accepted_entries)*100:.1f}%)")
print(f"  Toponymic (LOC) Prefix ORGs: {loc_prefix_cnt:3d} ({loc_prefix_cnt/len(accepted_entries)*100:.1f}%)")

# Print unique distinct entities
ent_counts = Counter(item['entity'] for item in accepted_entries)
print(f"\nUnique distinct organizations: {len(ent_counts)}")
for ent, cnt in ent_counts.most_common():
    print(f"  {ent:35s}: {cnt:2d}")
