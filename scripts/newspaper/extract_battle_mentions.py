"""
Enhanced Battle Extraction from Newspapers

Extracts battle mentions with context from Revolutionary War newspapers.
Goes beyond simple name matching to find contextual references.

Usage:
    python scripts/newspaper/extract_battle_mentions.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

# Revolutionary War battles and engagements
BATTLES = {
    "Lexington": ["Lexington", "Lexington and Concord"],
    "Concord": ["Concord"],
    "Bunker Hill": ["Bunker Hill", "Bunker's Hill", "Breed's Hill"],
    "Quebec": ["Quebec"],
    "Long Island": ["Long Island"],
    "White Plains": ["White Plains"],
    "Trenton": ["Trenton"],
    "Princeton": ["Princeton"],
    "Brandywine": ["Brandywine"],
    "Germantown": ["Germantown", "German Town"],
    "Saratoga": ["Saratoga"],
    "Monmouth": ["Monmouth"],
    "Savannah": ["Savannah"],
    "Charleston": ["Charleston", "Charles Town"],
    "Camden": ["Camden"],
    "King's Mountain": ["King's Mountain", "Kings Mountain"],
    "Cowpens": ["Cowpens", "Cow Pens"],
    "Guilford Courthouse": ["Guilford", "Guilford Courthouse"],
    "Yorktown": ["Yorktown", "York Town"],
    "Fort Ticonderoga": ["Ticonderoga", "Fort Ticonderoga"],
    "Fort Washington": ["Fort Washington"],
    "Valley Forge": ["Valley Forge"]
}

# Battle-related keywords that indicate military action
BATTLE_KEYWORDS = [
    "battle", "engagement", "skirmish", "siege", "attack", "assault",
    "victory", "defeat", "rout", "action", "affair", "combat",
    "fought", "fighting", "conflict"
]

# Context patterns that indicate a battle reference
CONTEXT_PATTERNS = [
    r'\b(battle|engagement|siege|attack|victory|defeat)\s+(?:of|at|near|in)\s+(\w+)',
    r'\b(\w+)\s+(?:battle|engagement|siege|attack|victory|defeat)',
    r'\b(?:at|near|in)\s+(\w+).*?\b(battle|engagement|fought|fighting)',
    r'\b(defeated|victorious|routed)\s+(?:at|near|in)\s+(\w+)',
]


def extract_battle_mentions(text: str) -> List[Dict]:
    """
    Extract battle mentions with context.
    
    Returns list of dicts with:
    - battle_name: Standardized battle name
    - mention_text: How it was mentioned in text
    - context: Surrounding text (50 chars before/after)
    - keywords: Battle-related keywords found nearby
    """
    
    mentions = []
    
    # Method 1: Direct battle name matching with context
    for standard_name, variations in BATTLES.items():
        for variation in variations:
            # Case-insensitive search
            pattern = re.compile(rf'\b{re.escape(variation)}\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Check if battle keywords are nearby (within 100 chars)
                nearby_start = max(0, match.start() - 100)
                nearby_end = min(len(text), match.end() + 100)
                nearby_text = text[nearby_start:nearby_end].lower()
                
                found_keywords = [kw for kw in BATTLE_KEYWORDS if kw in nearby_text]
                
                mentions.append({
                    'battle_name': standard_name,
                    'mention_text': match.group(),
                    'context': context.strip(),
                    'keywords': found_keywords,
                    'confidence': 'high' if found_keywords else 'medium'
                })
    
    # Method 2: Pattern-based extraction (find battles mentioned with keywords)
    for pattern in CONTEXT_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Extract location name from match
            groups = match.groups()
            location = groups[0] if len(groups) > 0 else None
            keyword = groups[1] if len(groups) > 1 else None
            
            if location:
                # Check if location matches a known battle
                matched_battle = None
                for standard_name, variations in BATTLES.items():
                    if any(location.lower() == var.lower() for var in variations):
                        matched_battle = standard_name
                        break
                
                if matched_battle:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    mentions.append({
                        'battle_name': matched_battle,
                        'mention_text': match.group(),
                        'context': context.strip(),
                        'keywords': [keyword] if keyword else [],
                        'confidence': 'high'
                    })
    
    # Deduplicate mentions (same battle, similar context)
    unique_mentions = []
    seen = set()
    
    for mention in mentions:
        key = (mention['battle_name'], mention['context'][:30])
        if key not in seen:
            seen.add(key)
            unique_mentions.append(mention)
    
    return unique_mentions


def analyze_battle_coverage(text_dir: Path, metadata_dir: Path) -> Dict:
    """
    Analyze battle coverage across all newspapers.
    """
    
    battle_stats = defaultdict(lambda: {
        'mention_count': 0,
        'newspapers': [],
        'contexts': [],
        'keywords': Counter()
    })
    
    newspaper_stats = []
    
    # Process each newspaper
    text_files = list(text_dir.glob("*.txt"))
    
    print(f"\n{'=' * 70}")
    print("⚔️  BATTLE MENTION EXTRACTION")
    print(f"{'=' * 70}\n")
    print(f"📂 Processing {len(text_files)} newspapers...\n")
    
    for text_file in text_files:
        lccn = text_file.stem
        
        # Skip improved OCR files
        if '_improved' in lccn:
            continue
        
        # Load text
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Load metadata
        metadata_file = metadata_dir / f"{lccn}_metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract battle mentions
        mentions = extract_battle_mentions(text)
        
        if mentions:
            print(f"📰 {metadata.get('newspaper_title', 'Unknown')}")
            print(f"   Date: {metadata.get('issue_date', 'Unknown')}")
            print(f"   Battles mentioned: {len(mentions)}")
            
            for mention in mentions:
                print(f"   - {mention['battle_name']}: \"{mention['mention_text']}\"")
                print(f"     Keywords: {', '.join(mention['keywords']) if mention['keywords'] else 'none'}")
                
                # Update stats
                battle_stats[mention['battle_name']]['mention_count'] += 1
                battle_stats[mention['battle_name']]['newspapers'].append({
                    'lccn': lccn,
                    'title': metadata.get('newspaper_title'),
                    'date': metadata.get('issue_date')
                })
                battle_stats[mention['battle_name']]['contexts'].append(mention['context'])
                
                for kw in mention['keywords']:
                    battle_stats[mention['battle_name']]['keywords'][kw] += 1
            
            print()
            
            newspaper_stats.append({
                'lccn': lccn,
                'title': metadata.get('newspaper_title'),
                'date': metadata.get('issue_date'),
                'battle_mentions': len(mentions),
                'battles': [m['battle_name'] for m in mentions]
            })
    
    return {
        'battle_stats': dict(battle_stats),
        'newspaper_stats': newspaper_stats
    }


def main():
    """Main execution."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    text_dir = base_dir / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    metadata_dir = base_dir / "data" / "documents" / "smithsonian" / "newspapers" / "metadata"
    output_file = base_dir / "data" / "documents" / "smithsonian" / "battle_mentions.json"
    
    # Analyze
    results = analyze_battle_coverage(text_dir, metadata_dir)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("📊 SUMMARY")
    print(f"{'=' * 70}\n")
    
    battle_stats = results['battle_stats']
    
    if battle_stats:
        print(f"Total battles mentioned: {len(battle_stats)}")
        print(f"Total mentions: {sum(b['mention_count'] for b in battle_stats.values())}")
        print(f"Newspapers with battle mentions: {len(results['newspaper_stats'])}")
        
        print(f"\n🏆 Top 10 Most Mentioned Battles:\n")
        sorted_battles = sorted(battle_stats.items(), 
                               key=lambda x: x[1]['mention_count'], 
                               reverse=True)[:10]
        
        for i, (battle, stats) in enumerate(sorted_battles, 1):
            print(f"  {i}. {battle}: {stats['mention_count']} mention(s)")
            print(f"     In {len(stats['newspapers'])} newspaper(s)")
            if stats['keywords']:
                top_keywords = stats['keywords'].most_common(3)
                kw_str = ', '.join(f"{kw} ({count})" for kw, count in top_keywords)
                print(f"     Keywords: {kw_str}")
            print()
    else:
        print("❌ No battle mentions found")
        print("\nPossible reasons:")
        print("  - Newspapers don't cover battles directly")
        print("  - OCR quality prevents matching")
        print("  - Battles mentioned with different terminology")
    
    # Save results
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
