"""
Book Metadata Enrichment Script

Enriches historical book metadata with:
- People mentioned (military personnel, political figures)
- Battles and locations
- Military units and operations
- Time periods and events
- War relevance scoring
- Subject tags

Adapted from newspaper enrichment workflow.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Book metadata extraction prompt
BOOK_ENRICHMENT_PROMPT = """You are a historical metadata extraction specialist. Analyze this historical book about the American Revolutionary War era and extract structured metadata.

BOOK TITLE: {title}
AUTHOR: {author}
TEXT SAMPLE (first 3000 chars):
{text_sample}

Extract the following metadata in JSON format:

{{
  "book_title": "Full title of the book",
  "author": "Author name",
  "publication_year": "YYYY if available",
  "time_period_covered": "Date range or era (e.g., '1775-1783', 'Revolutionary War Era')",
  "primary_subject": "Main topic (e.g., 'Battle of Lexington', 'Naval Operations', 'Military Leadership')",
  "battles_mentioned": ["List of specific battles mentioned"],
  "military_personnel": {{
    "american": ["Names of American officers/leaders"],
    "british": ["Names of British officers/leaders"],
    "other": ["Names of other military personnel"]
  }},
  "locations": ["States, cities, geographic locations mentioned"],
  "military_units": ["Regiments, militias, naval vessels mentioned"],
  "key_events": ["Major events described in the book"],
  "topics": ["political", "military", "social", "economic", "naval", "espionage", etc.],
  "war_relevance_score": 0.0-1.0 (how relevant to Revolutionary War),
  "has_military_content": true/false,
  "historical_significance": "Brief description of book's historical value"
}}

Guidelines:
- Extract names as they appear (e.g., "General George Washington", "Captain John Parker")
- Include both famous and lesser-known figures
- Be specific with battle names (e.g., "Battle of Bunker Hill" not just "Bunker Hill")
- Include geographic context for locations (e.g., "Boston, Massachusetts")
- War relevance score should be high (0.8-1.0) for books focused on Revolutionary War
- Return ONLY valid JSON, no other text"""


def extract_book_info(text: str) -> Dict[str, str]:
    """Extract basic book information from the text header."""
    lines = text.split('\n')[:50]  # Check first 50 lines
    
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
            # Try to extract year
            import re
            year_match = re.search(r'\b(17|18|19|20)\d{2}\b', line)
            if year_match:
                info['publication_year'] = year_match.group(0)
    
    return info


def enrich_book_metadata(book_path: Path, api_key: str) -> Dict[str, Any]:
    """
    Enrich a single book with metadata using Claude API.
    
    Args:
        book_path: Path to book text file
        api_key: Anthropic API key
        
    Returns:
        Dictionary containing enriched metadata
    """
    print(f"\n{'='*70}")
    print(f"Processing: {book_path.name}")
    print(f"{'='*70}")
    
    # Read book content
    with open(book_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Extract basic info from header
    book_info = extract_book_info(full_text)
    
    # Use first 3000 characters for API call (to fit in context)
    # This captures the title, author, and beginning content
    text_sample = full_text[:3000]
    
    # If no title found in header, use filename
    if not book_info['title']:
        book_info['title'] = book_path.stem.replace('_', ' ').title()
    
    print(f"Title: {book_info['title']}")
    print(f"Author: {book_info['author'] or 'Unknown'}")
    print(f"Text length: {len(full_text):,} characters")
    print(f"Sending to Claude API...")
    
    # Create Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Retry logic for API calls
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Call Claude API
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": BOOK_ENRICHMENT_PROMPT.format(
                        title=book_info['title'],
                        author=book_info['author'] or 'Unknown',
                        text_sample=text_sample
                    )
                }]
            )
            break  # Success, exit retry loop
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                print(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Final attempt failed, raise the error
    
    try:
        
        # Parse JSON response
        response_text = message.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif response_text.startswith('```'):
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        metadata = json.loads(response_text)
        
        # Add file information
        metadata['source_file'] = book_path.name
        metadata['file_path'] = str(book_path)
        metadata['text_length'] = len(full_text)
        metadata['enrichment_date'] = time.strftime('%Y-%m-%d')
        
        print(f"✅ Successfully enriched metadata")
        print(f"   - Battles: {len(metadata.get('battles_mentioned', []))}")
        print(f"   - American personnel: {len(metadata.get('military_personnel', {}).get('american', []))}")
        print(f"   - British personnel: {len(metadata.get('military_personnel', {}).get('british', []))}")
        print(f"   - Locations: {len(metadata.get('locations', []))}")
        print(f"   - War relevance: {metadata.get('war_relevance_score', 0):.2f}")
        
        return metadata
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        print(f"Response text: {response_text[:500]}")
        raise
    except Exception as e:
        print(f"❌ Error during enrichment: {e}")
        raise


def main():
    """Main enrichment workflow."""
    print("="*70)
    print("BOOK METADATA ENRICHMENT")
    print("="*70)
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)
    
    # Set up paths
    books_dir = Path("data/documents/books/text_files")
    output_dir = books_dir  # Save metadata files alongside book files
    
    if not books_dir.exists():
        print(f"❌ Error: Books directory not found: {books_dir}")
        sys.exit(1)
    
    # Find all book files
    book_files = sorted(books_dir.glob("*.txt"))
    
    # Filter out metadata files and the list file
    book_files = [f for f in book_files if not f.name.endswith('_metadata.json') 
                  and f.name != 'list_of_american_revolution_books.txt']
    
    print(f"\nFound {len(book_files)} books to process")
    print(f"Output directory: {output_dir}")
    
    # Process each book
    enriched_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, book_path in enumerate(book_files, 1):
        print(f"\n[{i}/{len(book_files)}] Processing: {book_path.name}")
        
        # Check if metadata already exists
        metadata_path = output_dir / f"{book_path.stem}_metadata.json"
        if metadata_path.exists():
            print(f"⏭️  Skipping (metadata already exists)")
            skipped_count += 1
            continue
        
        try:
            # Enrich metadata
            metadata = enrich_book_metadata(book_path, api_key)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved metadata to: {metadata_path.name}")
            enriched_count += 1
            
            # Rate limiting - wait 2 seconds between API calls
            if i < len(book_files):
                print("⏳ Waiting 2 seconds...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Error processing {book_path.name}: {e}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ENRICHMENT SUMMARY")
    print("="*70)
    print(f"Total books: {len(book_files)}")
    print(f"Enriched: {enriched_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Errors: {error_count}")
    print("="*70)
    
    if enriched_count > 0:
        print("\n✅ Next steps:")
        print("   1. Review the generated metadata files")
        print("   2. Run the rebuild pipeline to incorporate the metadata:")
        print("      python scripts/rebuild_pipeline.py")
        print("   3. Test queries with enhanced book metadata!")


if __name__ == "__main__":
    main()
