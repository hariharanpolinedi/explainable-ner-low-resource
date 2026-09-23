"""
Curate Kokborok gazetteers from public domain geographic, administrative, and cultural sources.
"""

from pathlib import Path

GAZ_DIR = Path("data/gazetteers/kokborok")
GAZ_DIR.mkdir(parents=True, exist_ok=True)

PLACES = [
    # Districts & Subdivisions
    "Tripura", "Tripurao", "West Tripura", "South Tripura", "North Tripura", "Dhalai", "Gomati", "Khowai", "Sepahijala", "Unakoti",
    # Cities & Towns
    "Agartala", "Agartalao", "Agartala-o", "Agartala Station-o", "Kailashahar", "Kailashahar-o", "Udaipur", "Udaipuro", "Dharmanagar",
    "Belonia", "Bilonia", "Bilonia-o", "Ambassa", "Khowai", "Teliamura", "Sabroom", "Sonamura", "Amarpur", "Santirbazar", "Kumarghat",
    "Kanchanpur", "Panisagar", "Gandacherra", "Bishalgarh", "Jirania", "Mohanpur", "Melaghar", "Ranirbazar",
    # Hills & Ranges
    "Jampui", "Jampui Hills", "Atharamura", "Boromura", "Longtharai", "Shakhan", "Deotamura",
    # Rivers (Twima)
    "Howrah", "Howrah Twima", "Gomati Twima", "Manu", "Manu Twima", "Dhalai Twima", "Khowai Twima", "Muhuri", "Feni", "Juri", "Deo",
    # Regional Toponyms & Landmarks
    "Ujjayanta Palace", "Neermahal", "Tripura Sundari", "Unakoti", "Pilak", "Boxanagar", "Chabimura", "Kasba", "Kamalasagar",
    "Baramura", "Khumpui Academy", "Dashamighat", "Battala", "Radhanagar", "Akhaura",
]

PERSONS = [
    # Historical Kings & Royals
    "Bir Bikram", "Bir Bikram Kishore Manikya", "Radha Kishore Manikya", "Birendra Kishore Manikya",
    "Kirit Bikram Kishore Manikya", "Pradyot Bikram Kishore Manikya", "Maharaja Bir Bikram",
    # Literary & Cultural Figures
    "Alindra Debbarma", "Sudhanya Debbarma", "Chandrakanta Murasingh", "Naphurai Jamatia", "Dhirendra Debbarma",
    "Nagendra Jamatia", "Bikashrai Debbarma", "Shephali Debbarma", "Narendra Debbarma", "Bimal Debbarma",
    # Political & Social Leaders
    "Dasarath Deb", "Nripen Chakraborty", "Manik Sarkar", "Biplab Kumar Deb", "Manik Saha",
    "Jiten Choudhury", "Bijoy Kumar Hrangkhawl", "Animesh Debbarma", "Mevar Kumar Jamatia",
    # Common Names in Corpus
    "Mohan", "Kanika", "Bwskang", "Khumbar", "Subrai", "Mailuma", "Khumpui", "Hachuk", "Hamkrwng",
    "Kwplai", "Bwtwk", "Chwngsa", "Dugra", "Khumtiya",
]

ORGS = [
    # State & Government Bodies
    "Tripura Legislative Assembly", "Tripura Government", "Tripura State", "Government of Tripura",
    "Tripura Tribal Areas Autonomous District Council", "TTAADC",
    # Academic & Research Institutions
    "Tripura University", "Maharaja Bir Bikram University", "MBB University", "NIT Agartala",
    "Tripura Medical College", "Agartala Government Medical College", "ICFAI University Tripura",
    "Tripura Board of Secondary Education", "TBSE", "Directorate of Kokborok",
    # Cultural & Literary Bodies
    "Twipra Students Federation", "TSF", "Kokborok Tei Hukumu Mission", "KTHM",
    "Tripura Cultural Research Institute", "Kokborok Sahitya Sabha", "Borok Peoples Human Rights Organisation",
    # Commercial & Corporate Bodies in Corpus
    "Tata Technologies", "Tata", "ONGC Tripura", "TSECL", "Tripura Gramin Bank", "State Bank of India",
]

# Write out unique sorted lists
with open(GAZ_DIR / "places.txt", "w", encoding="utf-8") as f:
    for p in sorted(set(PLACES)):
        f.write(p.strip() + "\n")

with open(GAZ_DIR / "persons.txt", "w", encoding="utf-8") as f:
    for p in sorted(set(PERSONS)):
        f.write(p.strip() + "\n")

with open(GAZ_DIR / "orgs.txt", "w", encoding="utf-8") as f:
    for o in sorted(set(ORGS)):
        f.write(o.strip() + "\n")

print(f"Places: {len(set(PLACES))} entries")
print(f"Persons: {len(set(PERSONS))} entries")
print(f"Orgs: {len(set(ORGS))} entries")
