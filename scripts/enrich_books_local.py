"""
Book Metadata Enrichment Script (Local/Offline)

Extracts structured metadata from historical book text files using pattern matching:
- Battles and locations
- Military personnel and ranks
- Time periods and events
- War-related keywords
- Subject classification

No API required - uses local pattern matching like the newspaper enrichment.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter
import time

# Revolutionary War battles and engagements
BATTLES = [
    "Lexington", "Concord", "Bunker Hill", "Breed's Hill",
    "Quebec", "Long Island", "White Plains", "Trenton", "Princeton",
    "Brandywine", "Germantown", "Saratoga", "Monmouth",
    "Savannah", "Charleston", "Camden", "King's Mountain",
    "Cowpens", "Guilford Courthouse", "Yorktown",
    "Fort Ticonderoga", "Fort Washington", "Valley Forge",
    "Freeman's Farm", "Bemis Heights", "Stony Point",
    "Paoli", "Oriskany", "Bennington", "Eutaw Springs",
    "Rhode Island", "Harlem Heights", "Fort Lee"
]

# Colonial/State names
PLACES = {
    "Massachusetts", "Virginia", "Pennsylvania", "New York",
    "Connecticut", "Rhode Island", "New Hampshire", "Maryland",
    "Delaware", "North Carolina", "South Carolina", "Georgia",
    "New Jersey", "Vermont", "Maine",
    "Boston", "Philadelphia", "New York City", "Charleston",
    "Baltimore", "Wilmington", "Norfolk", "Richmond",
    "Savannah", "Newport", "Providence", "Albany",
    "Cambridge", "Brooklyn", "Manhattan", "Staten Island",
    "England", "Britain", "London", "France", "Paris",
    "Spain", "Canada", "Quebec", "Halifax"
}

# Notable Revolutionary War figures
NOTABLE_PEOPLE = [
    "George Washington", "Benjamin Franklin", "Thomas Jefferson",
    "John Adams", "Samuel Adams", "Patrick Henry", "John Hancock",
    "Alexander Hamilton", "James Madison", "Thomas Paine",
    "Nathanael Greene", "Henry Knox", "Benedict Arnold", 
    "Horatio Gates", "Charles Lee", "Anthony Wayne",
    "Daniel Morgan", "Francis Marion", "John Paul Jones",
    "Marquis de Lafayette", "Baron von Steuben",
    "Israel Putnam", "William Prescott", "Ethan Allen",
    "John Stark", "Richard Montgomery", "Hugh Mercer",
    "King George III", "Lord Cornwallis", "William Howe", 
    "Henry Clinton", "John Burgoyne", "Thomas Gage",
    "Lord North", "Banastre Tarleton"
]

# Military ranks
RANKS = [
    "General", "Colonel", "Major", "Captain", "Lieutenant",
    "Sergeant", "Corporal", "Private", "Admiral", "Commander",
    "Brigadier", "Ensign", "Sir"
]

# Military units
MILITARY_UNITS = [
    "Continental Army", "Continental Congress", "British Army",
    "Royal Navy", "Hessian", "Dragoons", "Regiment", "Battalion",
    "Company", "Militia", "Minutemen", "Redcoats", "Grenadiers"
]

# War-related keywords
WAR_KEYWORDS = {
    "military": ["army", "navy", "troops", "soldiers", "militia", "regiment",
                 "battalion", "company", "garrison", "fleet"],
    "combat": ["battle", "engagement", "skirmish", "siege", "attack",
               "assault", "defense", "retreat", "victory", "defeat"],
    "personnel": ["enlistment", "recruitment", "desertion", "prisoner",
                  "casualty", "wounded", "killed", "captured"],
    "supplies": ["provisions", "ammunition", "supplies", "arms", "weapons",
                 "cannon", "musket", "powder"]
}


def extract_book_info(text: str) -> Dict[str, str]:
    """Extract basic book information from the text header."""
    lines = text.split('\n')[:50]
    
    info = {
        'title': '',
        'author': '',
        'publication_year': ''
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith('Title:'):
            info['title'] = line.replace('Title:', '').strip()
        elif line.startswith('Author:'):
            info['author'] = line.replace('Author:', '').strip()
        elif 'Release date:' in line or 'Publication' in line:
            year_match = re.search(r'\b(17|18|19|20)\d{2}\b', line)
            if year_match:
                info['publication_year'] = year_match.group(0)
    
    return info


def extract_people(text: str) -> Dict[str, List[str]]:
    """Extract people mentioned in the text, categorized."""
    american = set()
    british = set()
    other = set()
    
    # Check for notable people
    for person in NOTABLE_PEOPLE:
        if re.search(rf'\b{person}\b', text, re.IGNORECASE):
            # Categorize by name
            if any(name in person for name in ["Washington", "Franklin", "Jefferson", "Adams", 
                                               "Hamilton", "Madison", "Paine", "Greene", "Knox",
                                               "Arnold", "Gates", "Lee", "Wayne", "Morgan", 
                                               "Marion", "Jones", "Lafayette", "Steuben",
                                               "Putnam", "Prescott", "Allen", "Stark", 
                                               "Montgomery", "Mercer"]):
                american.add(person)
            elif any(name in person for name in ["George III", "Cornwallis", "Howe", 
                                                 "Clinton", "Burgoyne", "Gage", "North", "Tarleton"]):
                british.add(person)
            else:
                other.add(person)
    
    # Extract rank + name patterns
    for rank in RANKS:
        pattern = rf'\b{rank}\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\b'
        matches = re.findall(pattern, text)
        for match in matches:
            full_name = f"{rank} {match}"
            # Add to "other" by default (can't easily categorize)
            if full_name not in american and full_name not in british:
                other.add(full_name)
    
    return {
        "american": sorted(list(american))[:30],
        "british": sorted(list(british))[:30],
        "other": sorted(list(other))[:30]
    }


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
        if re.search(rf'\b(?:Battle\s+of\s+)?{battle}\b', text, re.IGNORECASE):
            battles.add(battle)
    
    return sorted(list(battles))


def extract_military_units(text: str) -> List[str]:
    """Extract military units mentioned."""
    units = set()
    
    for unit in MILITARY_UNITS:
        if re.search(rf'\b{unit}\b', text, re.IGNORECASE):
            units.add(unit)
    
    # Extract numbered regiments
    regiment_pattern = r'\b(\d+(?:st|nd|rd|th)\s+(?:Regiment|Battalion|Company))\b'
    matches = re.findall(regiment_pattern, text, re.IGNORECASE)
    for match in matches:
        units.add(match)
    
    return sorted(list(units))[:20]


def extract_time_period(text: str) -> str:
    """Extract the primary time period covered."""
    # Look for date patterns
    year_pattern = r'\b(17[7-8]\d)\b'
    years = re.findall(year_pattern, text)
    
    if years:
        year_counts = Counter(years)
        most_common = year_counts.most_common(5)
        
        if len(most_common) >= 2:
            min_year = min(y[0] for y in most_common)
            max_year = max(y[0] for y in most_common)
            return f"{min_year}-{max_year}"
        elif most_common:
            return most_common[0][0]
    
    return "Revolutionary War Era (1775-1783)"


def extract_key_events(text: str, battles: List[str]) -> List[str]:
    """Extract key events mentioned."""
    events = set()
    
    # Add battles as events
    for battle in battles:
        events.add(f"Battle of {battle}")
    
    # Look for common Revolutionary War events
    event_patterns = [
        (r'\bDeclaration\s+of\s+Independence\b', 'Declaration of Independence'),
        (r'\bBoston\s+Tea\s+Party\b', 'Boston Tea Party'),
        (r'\bStamp\s+Act\b', 'Stamp Act'),
        (r'\bContinental\s+Congress\b', 'Continental Congress'),
        (r'\bTreaty\s+of\s+Paris\b', 'Treaty of Paris'),
        (r'\bValley\s+Forge\b', 'Valley Forge Encampment'),
        (r'\bSurrender\s+at\s+Yorktown\b', 'Surrender at Yorktown')
    ]
    
    for pattern, event_name in event_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            events.add(event_name)
    
    return sorted(list(events))[:15]


def calculate_war_relevance(battles: List, military_units: List, people_dict: Dict) -> float:
    """Calculate relevance score for Revolutionary War (0-1)."""
    score = 0.0
    
    score += len(battles) * 0.10
    score += len(military_units) * 0.05
    score += len(people_dict.get('american', [])) * 0.05
    score += len(people_dict.get('british', [])) * 0.05
    
    return min(score, 1.0)


def classify_topics(text: str, battles: List, military_units: List) -> List[str]:
    """Classify the book's topics."""
    topics = []
    text_lower = text.lower()
    
    if battles or military_units:
        topics.append("military")
    
    if any(word in text_lower for word in ["congress", "government", "parliament", "treaty", "law"]):
        topics.append("political")
    
    if any(word in text_lower for word in ["navy", "naval", "ship", "fleet", "sea"]):
        topics.append("naval")
    
    if any(word in text_lower for word in ["spy", "intelligence", "secret", "espionage"]):
        topics.append("espionage")
    
    if any(word in text_lower for word in ["prison", "prisoner", "captured", "captive"]):
        topics.append("prisoners")
    
    if any(word in text_lower for word in ["loyalist", "tory", "british sympathizer"]):
        topics.append("loyalists")
    
    if any(word in text_lower for word in ["african", "negro", "slave", "colored"]):
        topics.append("african_americans")
    
    return topics if topics else ["general_history"]


def enrich_book(book_path: Path) -> Dict:
    """Enrich a single book with metadata."""
    print(f"\n{'='*70}")
    print(f"Processing: {book_path.name}")
    print(f"{'='*70}")
    
    # Read book
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract basic info
    book_info = extract_book_info(text)
    
    if not book_info['title']:
        book_info['title'] = book_path.stem.replace('_', ' ').title()
    
    print(f"Title: {book_info['title']}")
    print(f"Author: {book_info['author'] or 'Unknown'}")
    print(f"Length: {len(text):,} characters")
    
    # Use first 100,000 chars for analysis (representative sample)
    text_sample = text[:100000]
    
    # Extract metadata
    print("Extracting metadata...")
    people = extract_people(text_sample)
    places = extract_places(text_sample)
    battles = extract_battles(text_sample)
    military_units = extract_military_units(text_sample)
    time_period = extract_time_period(text_sample)
    key_events = extract_key_events(text_sample, battles)
    topics = classify_topics(text_sample, battles, military_units)
    war_relevance = calculate_war_relevance(battles, military_units, people)
    
    # Build metadata
    metadata = {
        "source_file": book_path.name,
        "file_path": str(book_path),
        "book_title": book_info['title'],
        "author": book_info['author'] or "Unknown",
        "publication_year": book_info['publication_year'],
        "text_length": len(text),
        "time_period_covered": time_period,
        "battles_mentioned": battles,
        "military_personnel": people,
        "locations": places,
        "military_units": military_units,
        "key_events": key_events,
        "topics": topics,
        "war_relevance_score": round(war_relevance, 2),
        "has_military_content": len(battles) > 0 or len(military_units) > 0,
        "enrichment_date": time.strftime('%Y-%m-%d'),
        "enrichment_method": "local_pattern_matching"
    }
    
    # Print summary
    print(f"[OK] Metadata extracted:")
    print(f"   - Battles: {len(battles)}")
    print(f"   - American personnel: {len(people['american'])}")
    print(f"   - British personnel: {len(people['british'])}")
    print(f"   - Locations: {len(places)}")
    print(f"   - Military units: {len(military_units)}")
    print(f"   - War relevance: {war_relevance:.2f}")
    print(f"   - Topics: {', '.join(topics)}")
    
    return metadata


def main():
    """Main enrichment workflow."""
    print("="*70)
    print("BOOK METADATA ENRICHMENT (LOCAL)")
    print("="*70)
    
    # Set up paths
    books_dir = Path("data/documents/books/text_files")
    output_dir = books_dir
    
    if not books_dir.exists():
        print(f"[ERROR] Books directory not found: {books_dir}")
        return
    
    # Find all book files
    book_files = sorted(books_dir.glob("*.txt"))
    book_files = [f for f in book_files if not f.name.endswith('_metadata.json') 
                  and f.name != 'list_of_american_revolution_books.txt']
    
    print(f"\nFound {len(book_files)} books to process")
    print(f"Output directory: {output_dir}")
    
    # Process each book
    enriched_count = 0
    skipped_count = 0
    
    for i, book_path in enumerate(book_files, 1):
        print(f"\n[{i}/{len(book_files)}]")
        
        # Check if metadata already exists
        metadata_path = output_dir / f"{book_path.stem}_metadata.json"
        if metadata_path.exists():
            print(f"[SKIP] {book_path.name} (metadata already exists)")
            skipped_count += 1
            continue
        
        try:
            # Enrich metadata
            metadata = enrich_book(book_path)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVED] {metadata_path.name}")
            enriched_count += 1
            
        except Exception as e:
            print(f"[ERROR] Error processing {book_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ENRICHMENT SUMMARY")
    print("="*70)
    print(f"Total books: {len(book_files)}")
    print(f"Enriched: {enriched_count}")
    print(f"Skipped: {skipped_count}")
    print("="*70)
    
    if enriched_count > 0:
        print("\n[SUCCESS] Next steps:")
        print("   1. Review the generated metadata files")
        print("   2. Rebuild the pipeline:")
        print("      python scripts/rebuild_pipeline.py")
        print("   3. Test queries with enhanced book metadata!")


if __name__ == "__main__":
    main()
