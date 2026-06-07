"""
Re-Enrich Newspaper Metadata Using Improved OCR

After running improve_ocr.py, this script re-enriches all newspaper metadata
using the improved OCR text files (*_improved.txt).

Usage:
    python scripts/newspaper/re_enrich_with_improved_ocr.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# Import extraction functions from enrich_newspaper_files.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

from enrich_newspaper_files import (
    extract_people, extract_places, extract_battles,
    extract_military_units, extract_war_keywords,
    classify_subject, calculate_war_relevance
)


def re_enrich_newspaper(text_file: Path, metadata_file: Path) -> Dict:
    """
    Re-enrich a newspaper using improved OCR text.
    """
    
    # Load original metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load improved OCR text
    with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Extract entities
    people = extract_people(text)
    places = extract_places(text)
    battles = extract_battles(text)
    military_units = extract_military_units(text)
    war_keywords = extract_war_keywords(text)
    subjects = classify_subject(text, war_keywords, battles, military_units)
    
    # Calculate metrics
    entity_count = len(people) + len(places) + len(battles) + len(military_units)
    war_relevance = calculate_war_relevance(battles, military_units, war_keywords, people)
    
    # Update metadata with improved extractions
    metadata.update({
        "people_mentioned": people,
        "places_mentioned": places,
        "battles_mentioned": battles,
        "military_units": military_units,
        "war_keywords": war_keywords,
        "subject_tags": subjects,
        "entity_count": entity_count,
        "war_relevance_score": war_relevance,
        "has_war_content": war_relevance > 0,
        "ocr_improved": True,
        "improved_ocr_file": text_file.name
    })
    
    return metadata


def main():
    """Main execution."""
    
    base_dir = Path(__file__).parent.parent.parent
    text_dir = base_dir / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    
    # Find all improved OCR files
    improved_files = list(text_dir.glob("*_improved.txt"))
    
    print(f"\n{'=' * 70}")
    print("🔄 RE-ENRICHING NEWSPAPERS WITH IMPROVED OCR")
    print(f"{'=' * 70}\n")
    print(f"📂 Found {len(improved_files)} improved OCR files\n")
    
    if not improved_files:
        print("❌ No improved OCR files found!")
        print("   Run: python scripts/newspaper/improve_ocr.py first")
        return
    
    stats = {
        'processed': 0,
        'errors': 0,
        'people_before': 0,
        'people_after': 0,
        'battles_before': 0,
        'battles_after': 0,
        'war_relevance_improved': 0
    }
    
    for improved_file in improved_files:
        # Get LCCN from filename
        lccn = improved_file.stem.replace('_improved', '')
        metadata_file = text_dir / f"{lccn}_metadata.json"
        
        if not metadata_file.exists():
            print(f"⚠️  Skipping {lccn}: No metadata file found")
            continue
        
        try:
            # Load original metadata for comparison
            with open(metadata_file, 'r', encoding='utf-8') as f:
                original = json.load(f)
            
            # Re-enrich
            updated = re_enrich_newspaper(improved_file, metadata_file)
            
            # Track improvements
            stats['people_before'] += len(original.get('people_mentioned', []))
            stats['people_after'] += len(updated['people_mentioned'])
            stats['battles_before'] += len(original.get('battles_mentioned', []))
            stats['battles_after'] += len(updated['battles_mentioned'])
            
            if updated['war_relevance_score'] > original.get('war_relevance_score', 0):
                stats['war_relevance_improved'] += 1
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(updated, f, indent=2, ensure_ascii=False)
            
            stats['processed'] += 1
            
            # Print progress
            if stats['processed'] % 10 == 0:
                print(f"✓ Processed {stats['processed']}/{len(improved_files)} newspapers")
        
        except Exception as e:
            print(f"❌ Error processing {lccn}: {e}")
            stats['errors'] += 1
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("📊 RE-ENRICHMENT SUMMARY")
    print(f"{'=' * 70}\n")
    print(f"Successfully processed: {stats['processed']}/{len(improved_files)}")
    print(f"Errors: {stats['errors']}")
    print(f"\n📈 Improvements:")
    print(f"  People mentioned: {stats['people_before']} → {stats['people_after']} "
          f"({stats['people_after'] - stats['people_before']:+d})")
    print(f"  Battles mentioned: {stats['battles_before']} → {stats['battles_after']} "
          f"({stats['battles_after'] - stats['battles_before']:+d})")
    print(f"  War relevance improved: {stats['war_relevance_improved']} newspapers")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
