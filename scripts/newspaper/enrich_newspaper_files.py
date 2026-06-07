"""
Newspaper Metadata Enrichment Script

Extracts structured metadata from Revolutionary War era newspaper text files:
- Named entities (people, places, organizations)
- War-related content (battles, military terms, keywords)
- Subject classification
- Entity density for relevance ranking

Usage:
    python scripts/newspaper/enrich_newspaper_files.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

# Revolutionary War battles and engagements
BATTLES = [
    "Lexington", "Concord", "Bunker Hill", "Breed's Hill",
    "Quebec", "Long Island", "White Plains", "Trenton", "Princeton",
    "Brandywine", "Germantown", "Saratoga", "Monmouth",
    "Savannah", "Charleston", "Camden", "King's Mountain",
    "Cowpens", "Guilford Courthouse", "Yorktown",
    "Fort Ticonderoga", "Fort Washington", "Valley Forge"
]

# Colonial/State names (historical variations)
PLACES = {
    # States/Colonies
    "Massachusetts", "Virginia", "Pennsylvania", "New York",
    "Connecticut", "Rhode Island", "New Hampshire", "Maryland",
    "Delaware", "North Carolina", "South Carolina", "Georgia",
    "New Jersey", "Vermont", "Maine",
    # Major cities
    "Boston", "Philadelphia", "New York", "Charleston",
    "Baltimore", "Wilmington", "Norfolk", "Richmond",
    "Savannah", "Newport", "Providence", "Albany",
    "Portland", "Scarborough", "York", "Georgetown",
    # Foreign
    "England", "Britain", "London", "France", "Paris",
    "Spain", "Canada", "Quebec", "Halifax", "Constantinople",
    "Jamaica", "Grenada", "Liverpool", "Rhodes", "Ireland", "Scotland",
    "Amsterdam", "Brussels", "Lisbon", "Rome", "Athens",
    # General regions
    "America", "Europe", "Asia", "Africa"
}

# Military units and organizations
MILITARY_UNITS = [
    "Continental Army", "Continental Congress", "British Army",
    "Royal Navy", "Hessian", "Dragoons", "Regiment", "Battalion",
    "Company", "Militia", "Minutemen", "Redcoats"
]

# Military ranks and titles
RANKS = [
    "General", "Colonel", "Major", "Captain", "Lieutenant",
    "Sergeant", "Corporal", "Private", "Admiral", "Commander",
    "Ensign", "Esq", "Dr", "Mr", "Mrs", "Miss", "Rev"
]

# War-related keywords
WAR_KEYWORDS = {
    "military": ["army", "navy", "troops", "soldiers", "militia", "regiment",
                 "battalion", "company", "garrison", "fleet", "squadron"],
    "combat": ["battle", "engagement", "skirmish", "siege", "attack",
               "assault", "defense", "retreat", "victory", "defeat"],
    "personnel": ["enlistment", "recruitment", "desertion", "prisoner",
                  "casualty", "wounded", "killed", "captured"],
    "supplies": ["provisions", "ammunition", "supplies", "arms", "weapons",
                 "cannon", "musket", "powder", "rations"],
    "operations": ["march", "maneuver", "deployment", "encampment",
                   "fortification", "blockade", "expedition"]
}

# Political figures (Revolutionary War era)
NOTABLE_PEOPLE = [
    "George Washington", "Benjamin Franklin", "Thomas Jefferson",
    "John Adams", "Samuel Adams", "Patrick Henry", "John Hancock",
    "Alexander Hamilton", "James Madison", "Thomas Paine",
    "Marquis de Lafayette", "Baron von Steuben", "Nathanael Greene",
    "Henry Knox", "Benedict Arnold", "Horatio Gates", "Charles Lee",
    # British
    "King George", "Lord Cornwallis", "General Howe", "General Clinton",
    "General Burgoyne", "Admiral Howe", "Lord North"
]


def extract_people(text: str) -> List[str]:
    """Extract people mentioned in the text."""
    people = set()
    
    # Specific bad names to exclude (OCR errors only - not real names)
    bad_names = {
        # Geographic terms misread as names
        "British India", "MillCreek", "Nov Chester", "Oct Penn", 
        "Pennsylvania County", "Sussex Hundred",
        
        # OCR phrase errors
        "This Couch", "This Son", "You Son", "For Rented", "Fora Son", 
        "The Son", "This Cough", "Sore Asthmas",
        
        # Month + place combinations (OCR errors)
        "Jan Wilmington", "July Wilmington", "June Wilmington",
        "April Portland", "June Portland", "Fune Portland", "Fune Scarbrrough",
        "Mareh Georgetown", "May Bfon", "Jaa York", "June Mewgloucfter",
        "January Connecticut", "October Hose", "November Town",
        # Compound phrases (OCR errors)
        "Camden Town", "Dorchester Philadelphia", "United States", "Natural United",
        "Now Alexandria", "Now Counzty", "Now Martinsburg", "Virginia Country",
        "Maryland County", "Onthe Girls", "For London", "Girls Pumps",
        "Boy Shoes", "White Stones", "Gold Blue", "Mens Threads",
        "Prussian Lead", "Cash Pine", "Together Articles", "Last America", "Ins America",
        
        # Market + name (likely "Market Street" or price listings)
        "Market Delaware", "Market Harwcod", "Market Smith", "Market Walker",
        
        # Obvious OCR garbage
        "Ap Neck", "OtiCce", "Forest Board", "DoLiars",
        # OCR errors for titles
        "Efq", "Efg"  # Should be "Esq" (Esquire)
    }
    
    # Common words that appear in ads/listings (not people)
    ad_words = {
        # Location/building terms
        "Wharf", "Store", "Street", "Corner", "Mreet", "Town", "County", "Country",
        "Ground", "Situated",
        # Products/commodities
        "Wine", "Duck", "Book", "Cotton", "Souchong", "Fowl", "Geese", 
        "Chick", "Lozenges", "Kettes", "Rats", "Worm", "Sale", "Flour", "Sugar",
        "Tooth", "Market", "Stones", "Vessels", "Ships", "Articles", "Together",
        "Butter", "Barley", "Pepper", "Cloves", "Salt", "Rice", "Wheat", "Oats",
        "Teas", "Black", "Green", "Hyson", "Bohea",
        # Plants/garden
        "Evergreens", "Shrubs", "Fruit", "Kitchen", "Plants", "Green", "Flowers",
        "Nursery", "Seed", "Trees", "Biennial", "Annual",
        # Fruits and vegetables
        "Celery", "Endive", "Turnip", "Lettuce", "Squash", "Cabbage", "Carrot",
        "Potato", "Onion", "Beans", "Peas", "Corn", "Tomato", "Cucumber",
        "Radish", "Beet", "Parsnip", "Spinach", "Asparagus",
        "Lemon", "Bergamot", "Pears", "Apples", "Quince", "Apricots",
        "Orange", "Peach", "Plum", "Cherry", "Grape", "Melon",
        # Clothing/textiles
        "Shoes", "Pumps", "Glassware", "Cutlery", "Hose", "Ties", "Gloves",
        "Silks", "Stockings", "Umbrellas", "Threads", "Lead", "Hats", "Ribbons",
        "Bonnets", "Caps", "Boots", "Slippers",
        # Reference materials
        "Navigator", "Pilot", "Dictionary", "Reports", "Laws", "Geography",
        "Entries", "Scales", "Dividers", "Declarations", "Journal", "Published",
        "Didionaries", "Dictionaries",
        # Academic subjects
        "Divinity", "Surgery", "Prophesies", "Mathmatics", "Mathematics",
        "Theology", "Philosophy", "Grammar", "Rhetoric", "Logic",
        # General words
        "Young", "Boxes", "Bags", "Basin", "Bed", "Coast", "Daily",
        "Sewing", "Valuable", "Nothing", "Eat", "Gentlemen", "Natural",
        "Miscellaneous", "Gold", "Blue", "Imported", "Prussian", "Cash", "Pine",
        "Count", "Fist", "Whool", "Medicines", "Heap", "No", "Quilting", 
        "Fancyquilting", "Opium", "Peruvian", "Fancy"
    }
    
    # Check for notable people
    for person in NOTABLE_PEOPLE:
        if person in text:
            people.add(person)
    
    # Extract rank + name patterns (e.g., "General Washington", "Captain Smith")
    for rank in RANKS:
        # Two-word names with rank
        pattern = rf'\b{rank}\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\b'
        matches = re.findall(pattern, text)
        for match in matches:
            full_with_rank = f"{rank} {match}"
            name_only = match.strip()
            # Check bad names before adding
            if full_with_rank not in bad_names and name_only not in bad_names:
                people.add(full_with_rank)
                people.add(name_only)
    
    # Extract "Lastname, Firstname" format (common in lists and records)
    lastname_first_pattern = r'\b([A-Z][a-z]+),\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\b'
    matches = re.findall(lastname_first_pattern, text)
    for lastname, firstname in matches:
        full_name = f"{firstname} {lastname}".strip()
        # Check bad names and ad words
        contains_ad_word = any(ad_word in full_name for ad_word in ad_words)
        if len(full_name) > 5 and full_name not in bad_names and not contains_ad_word:
            people.add(full_name)
    
    # Extract capitalized names (simple heuristic)
    # Pattern: Title Case Name (2-3 words)
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    matches = re.findall(pattern, text)
    
    # Filter out common words, places, and OCR errors
    common_words = {"The", "This", "That", "These", "Those", "From", "With", "When", "Where",
                   "Some", "Many", "Most", "Such", "Other", "Being", "Having", "During",
                   "For", "You", "Your", "Our", "Their", "His", "Her", "Its", "Which",
                   "In", "Of", "On", "At", "By", "To", "Now", "Men", "Women", "Girls", "Boys"}
    
    # OCR error patterns to exclude
    ocr_error_patterns = [
        r'\d',  # Contains digits  
        r'[A-Z]{3,}',  # All caps (likely abbreviation or OCR error)
        r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}',  # 5+ consonants in a row (OCR garbage)
        r'(Abridyment|Journats|Maitfers|Englith|Midtikcn|Readmg|Broying|Sbangler)',  # Misspelled words
        r'^(Sir|The|No)\s[A-Z][a-z]{2,4}$',  # "Sir Jam", "The Lozenyes", "No Middiings"
    ]
    
    for match in matches:
        match_clean = match.strip()
        
        # Check if it's in the bad names list
        if match_clean in bad_names:
            continue
        
        # Check if it contains any ad words (e.g., "Alfo Wharf", "Boxes Cotton")
        contains_ad_word = any(ad_word in match_clean for ad_word in ad_words)
        if contains_ad_word:
            continue
        
        # Check if it matches any OCR error pattern
        is_ocr_error = any(re.search(pattern, match_clean) for pattern in ocr_error_patterns)
        
        # Skip if it's a place, common word, OCR error, or too short
        if (match_clean not in PLACES and 
            match_clean not in common_words and 
            not is_ocr_error and
            len(match_clean) > 5 and
            not match_clean.startswith("The ") and
            not match_clean.startswith("For ") and
            not match_clean.startswith("This ")):
            people.add(match_clean)
    
    # Remove duplicates and sort
    # Limit to top 50 to capture more names for cross-referencing
    return sorted(list(people))[:50]


def extract_places(text: str) -> List[str]:
    """Extract places mentioned in the text."""
    places = set()
    
    for place in PLACES:
        if re.search(rf'\b{place}\b', text, re.IGNORECASE):
            places.add(place)
    
    return sorted(list(places))


def extract_battles(text: str) -> List[str]:
    """Extract battles/engagements mentioned."""
    battles = set()
    
    for battle in BATTLES:
        # Check for "Battle of X" or just "X"
        if re.search(rf'\b(?:Battle\s+of\s+)?{battle}\b', text, re.IGNORECASE):
            battles.add(battle)
    
    return sorted(list(battles))


def extract_military_units(text: str) -> List[str]:
    """Extract military units mentioned."""
    units = set()
    
    for unit in MILITARY_UNITS:
        if re.search(rf'\b{unit}\b', text, re.IGNORECASE):
            units.add(unit)
    
    # Extract numbered regiments (e.g., "5th Regiment", "2nd Battalion")
    regiment_pattern = r'\b(\d+(?:st|nd|rd|th)\s+(?:Regiment|Battalion|Company))\b'
    matches = re.findall(regiment_pattern, text, re.IGNORECASE)
    for match in matches:
        units.add(match)
    
    return sorted(list(units))


def extract_war_keywords(text: str) -> Dict[str, List[str]]:
    """Extract war-related keywords by category."""
    found_keywords = {}
    text_lower = text.lower()
    
    for category, keywords in WAR_KEYWORDS.items():
        found = []
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword)
        if found:
            found_keywords[category] = sorted(found)
    
    return found_keywords


def classify_subject(text: str, war_keywords: Dict, battles: List, military_units: List) -> List[str]:
    """Classify the subject/topic of the article."""
    subjects = []
    text_lower = text.lower()
    
    # Military news
    if battles or military_units or war_keywords.get("combat") or war_keywords.get("military"):
        subjects.append("military_news")
    
    # Political
    if any(word in text_lower for word in ["congress", "government", "parliament", "treaty", "law"]):
        subjects.append("political")
    
    # Economic/Trade
    if any(word in text_lower for word in ["trade", "commerce", "merchant", "goods", "price", "market"]):
        subjects.append("economic")
    
    # Medicine/Health Advertisement
    medicine_keywords = ["elixir", "medicine", "remedy", "cure", "cough", "fever", "pain", 
                        "disease", "afflict", "health", "physician", "apothecary", "lozenges",
                        "pills", "ointment", "tonic", "balsam", "cordial", "consumption"]
    if any(word in text_lower for word in medicine_keywords):
        subjects.append("medicine")
        subjects.append("advertisement")
    
    # General Advertisement (if not already medicine)
    elif any(word in text_lower for word in ["for sale", "wanted", "notice", "reward", "to be sold", "just received"]):
        subjects.append("advertisement")
    
    # Troop movements
    if war_keywords.get("operations"):
        subjects.append("troop_movements")
    
    # Casualties/Personnel
    if war_keywords.get("personnel"):
        subjects.append("personnel")
    
    return subjects if subjects else ["general"]


def calculate_war_relevance(battles: List, military_units: List, war_keywords: Dict, people: List) -> float:
    """Calculate a relevance score for war-related content (0-1)."""
    score = 0.0
    
    # Battles are highly relevant
    score += len(battles) * 0.15
    
    # Military units
    score += len(military_units) * 0.10
    
    # War keywords
    total_keywords = sum(len(kw_list) for kw_list in war_keywords.values())
    score += total_keywords * 0.05
    
    # Military ranks in people
    military_people = [p for p in people if any(rank in p for rank in RANKS)]
    score += len(military_people) * 0.10
    
    return min(score, 1.0)  # Cap at 1.0


def assess_ocr_quality(text: str) -> str:
    """Assess OCR quality based on text characteristics."""
    if len(text) < 100:
        return "unknown"
    
    # Count OCR error indicators
    error_indicators = 0
    sample = text[:2000]  # Check first 2000 chars
    
    # Check for excessive special characters
    special_chars = sum(1 for c in sample if not c.isalnum() and not c.isspace() and c not in '.,;:!?-"\'')
    if special_chars > len(sample) * 0.1:  # More than 10% special chars
        error_indicators += 1
    
    # Check for broken words (excessive single letters)
    words = sample.split()
    single_chars = sum(1 for w in words if len(w) == 1 and w.isalpha())
    if len(words) > 0 and single_chars / len(words) > 0.15:  # More than 15% single letters
        error_indicators += 1
    
    # Check for long s (ſ) - indicates historical text but not necessarily bad OCR
    if 'ſ' in sample or 'Ʃ' in sample:
        # Historical text, but not an error indicator
        pass
    
    # Check for excessive uppercase in middle of words (OCR confusion)
    mixed_case = sum(1 for w in words if len(w) > 2 and any(c.isupper() for c in w[1:]))
    if len(words) > 0 and mixed_case / len(words) > 0.1:
        error_indicators += 1
    
    # Classify quality
    if error_indicators == 0:
        return "good"
    elif error_indicators == 1:
        return "fair"
    else:
        return "poor"


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text to handle common historical newspaper OCR errors."""
    # Replace long s (ſ) with regular s
    text = text.replace('ſ', 's')
    text = text.replace('Ʃ', 'S')
    
    # Common OCR character substitutions
    ocr_fixes = {
        'vv': 'w',  # Double v often used for w
        'VV': 'W',
        'ii': 'u',  # In some contexts
        'rn': 'm',  # Common OCR confusion
        'cl': 'd',  # Common OCR confusion
        '0': 'O',   # In names
    }
    
    # Apply fixes cautiously (only in specific contexts to avoid false positives)
    # These are applied to improve name matching
    
    return text


def enrich_newspaper_file(text_file: Path) -> Optional[Dict]:
    """Extract metadata from a single newspaper text file."""
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Normalize OCR errors
        text = normalize_ocr_text(text)
        
        # Extract entities
        people = extract_people(text)
        places = extract_places(text)
        battles = extract_battles(text)
        military_units = extract_military_units(text)
        war_keywords = extract_war_keywords(text)
        
        # Classification
        subjects = classify_subject(text, war_keywords, battles, military_units)
        war_relevance = calculate_war_relevance(battles, military_units, war_keywords, people)
        
        # Calculate entity count for ranking
        entity_count = len(people) + len(places) + len(battles) + len(military_units)
        
        # Assess OCR quality (heuristic based on text characteristics)
        ocr_quality = assess_ocr_quality(text)
        
        # Build enriched metadata
        enriched = {
            "people_mentioned": people,
            "places_mentioned": places,
            "battles_mentioned": battles,
            "military_units": military_units,
            "war_keywords": war_keywords,
            "subject_tags": subjects,
            "entity_count": entity_count,
            "war_relevance_score": round(war_relevance, 3),
            "has_war_content": war_relevance > 0.1,
            "ocr_quality": ocr_quality,
            "extraction_note": "Historical newspaper OCR quality varies. Names and entities may be incomplete or inaccurate."
        }
        
        return enriched
        
    except Exception as e:
        print(f"  ❌ Error processing {text_file.name}: {e}")
        return None


def main():
    """Main enrichment process."""
    
    # Path to newspaper text files
    base_dir = Path(__file__).parent.parent.parent / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    
    # Find all text files
    text_files = list(base_dir.glob("*.txt"))
    
    if not text_files:
        print(f"❌ No text files found in {base_dir}")
        return
    
    print(f"\n✓  Found {len(text_files)} newspaper text files\n")
    print("⏳ Processing files...")
    
    processed = 0
    enriched_count = 0
    errors = 0
    
    for text_file in text_files:
        # Read existing metadata
        metadata_file = text_file.with_suffix('').with_suffix('.txt').parent / f"{text_file.stem}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {"lccn": text_file.stem}
        
        # Extract enriched metadata
        enriched = enrich_newspaper_file(text_file)
        
        if enriched:
            # Merge with existing metadata
            metadata.update(enriched)
            enriched_count += 1
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        else:
            errors += 1
        
        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(text_files)} files...")
    
    print(f"  Processed {processed}/{len(text_files)} files...")
    print("\n" + "=" * 70)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal files: {len(text_files)}")
    print(f"Processed: {processed}")
    print(f"Enriched: {enriched_count}")
    print(f"Errors: {errors}")
    print(f"\n✓  Saved {enriched_count} enriched metadata files to: {base_dir}")


if __name__ == "__main__":
    main()
