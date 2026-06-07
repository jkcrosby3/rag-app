"""
Cross-Reference Veterans Script

Finds connections between pension files and newspaper mentions.
Links veteran names from pension files to newspaper articles that mention them.

Usage:
    python scripts/newspaper/cross_reference_veterans.py
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_pension_metadata(pension_dir: Path) -> Dict[str, str]:
    """Load veteran names from pension metadata files."""
    veterans = {}  # filename -> veteran_name
    
    metadata_files = list(pension_dir.glob("*_metadata.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            veteran_name = metadata.get("veteran_name", "")
            
            # Skip special markers
            if veteran_name and not veteran_name.startswith("["):
                file_id = metadata_file.stem.replace("_metadata", "")
                veterans[file_id] = veteran_name
        except Exception as e:
            print(f"  ⚠️  Error reading {metadata_file.name}: {e}")
    
    return veterans


def load_newspaper_metadata(newspaper_dir: Path) -> Dict[str, Dict]:
    """Load newspaper metadata with people mentioned."""
    newspapers = {}  # lccn -> metadata
    
    metadata_files = list(newspaper_dir.glob("*_metadata.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            lccn = metadata.get("lccn", metadata_file.stem.replace("_metadata", ""))
            newspapers[lccn] = metadata
        except Exception as e:
            print(f"  ⚠️  Error reading {metadata_file.name}: {e}")
    
    return newspapers


def normalize_name(name: str) -> str:
    """Normalize name for comparison (remove ranks, extra spaces)."""
    # Remove military ranks
    ranks = ["General", "Colonel", "Major", "Captain", "Lieutenant", 
             "Sergeant", "Corporal", "Private", "Admiral", "Commander"]
    
    for rank in ranks:
        name = name.replace(f"{rank} ", "")
    
    return name.strip().lower()


def find_matches(veterans: Dict[str, str], newspapers: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Find veterans mentioned in newspapers."""
    matches = defaultdict(list)  # veteran_name -> list of newspaper matches
    
    # Create normalized lookup
    veteran_normalized = {normalize_name(name): (file_id, name) 
                         for file_id, name in veterans.items()}
    
    # Check each newspaper
    for lccn, newspaper in newspapers.items():
        people_mentioned = newspaper.get("people_mentioned", [])
        
        for person in people_mentioned:
            person_normalized = normalize_name(person)
            
            # Check if this person matches a veteran
            if person_normalized in veteran_normalized:
                file_id, veteran_name = veteran_normalized[person_normalized]
                
                match_info = {
                    "newspaper_lccn": lccn,
                    "newspaper_title": newspaper.get("newspaper_title", "Unknown"),
                    "issue_date": newspaper.get("issue_date", "Unknown"),
                    "place_of_publication": newspaper.get("place_of_publication", "Unknown"),
                    "mentioned_as": person,
                    "subject_tags": newspaper.get("subject_tags", []),
                    "war_relevance_score": newspaper.get("war_relevance_score", 0),
                    "pension_file_id": file_id,
                    "web_url": newspaper.get("Web_URL", "")
                }
                
                matches[veteran_name].append(match_info)
    
    return dict(matches)


def main():
    """Main cross-reference process."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent / "data" / "documents" / "smithsonian"
    pension_dir = base_dir / "pension_files" / "text_files"
    newspaper_dir = base_dir / "newspapers" / "text_files"
    output_file = base_dir / "veteran_newspaper_cross_references.json"
    
    print("\n" + "=" * 70)
    print("🔗 VETERAN-NEWSPAPER CROSS-REFERENCE")
    print("=" * 70)
    
    # Load data
    print("\n⏳ Loading pension metadata...")
    veterans = load_pension_metadata(pension_dir)
    print(f"  ✓ Loaded {len(veterans)} veteran names from pension files")
    
    print("\n⏳ Loading newspaper metadata...")
    newspapers = load_newspaper_metadata(newspaper_dir)
    print(f"  ✓ Loaded {len(newspapers)} newspaper metadata files")
    
    # Find matches
    print("\n⏳ Finding cross-references...")
    matches = find_matches(veterans, newspapers)
    
    # Statistics
    total_matches = sum(len(newspaper_list) for newspaper_list in matches.values())
    
    print("\n" + "=" * 70)
    print("📊 CROSS-REFERENCE RESULTS")
    print("=" * 70)
    print(f"\nVeterans with newspaper mentions: {len(matches)}")
    print(f"Total newspaper mentions: {total_matches}")
    
    if matches:
        print(f"\n🎯 Top Veterans by Newspaper Mentions:")
        sorted_matches = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (veteran_name, newspaper_list) in enumerate(sorted_matches[:10], 1):
            print(f"  {i}. {veteran_name}: {len(newspaper_list)} mention(s)")
            for mention in newspaper_list[:3]:  # Show first 3 newspapers
                print(f"     - {mention['newspaper_title']} ({mention['issue_date']})")
            if len(newspaper_list) > 3:
                print(f"     ... and {len(newspaper_list) - 3} more")
    
    # Save results
    print(f"\n💾 Saving cross-references to: {output_file}")
    
    # Convert to serializable format
    output_data = {
        "summary": {
            "total_veterans": len(veterans),
            "veterans_with_mentions": len(matches),
            "total_mentions": total_matches
        },
        "matches": matches
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(matches)} veteran cross-references")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
