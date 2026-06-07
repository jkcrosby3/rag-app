"""
LLM-based enrichment for pension files with low-confidence regex extraction.
Uses GPT-4o-mini or local Llama to extract structured data from narrative text.
"""

import os
import json
import glob
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai package not installed. Run: pip install openai")

# Configuration
INPUT_DIR = Path("data/documents/smithsonian/pension_files/text_files")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set in environment
MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo" for cheaper option

# Confidence thresholds for identifying low-confidence files
LOW_CONFIDENCE_CRITERIA = {
    'veteran_name': lambda m: m.get('veteran_name') is None,
    'family_info': lambda m: m.get('wife_name') is None and m.get('children_count') is None,
    'pension_amount': lambda m: m.get('pension_amount') is None,
}


def identify_low_confidence_files() -> List[str]:
    """Identify files that need LLM enhancement based on missing critical fields."""
    
    metadata_files = glob.glob(str(INPUT_DIR / "*_metadata.json"))
    low_confidence = []
    
    print(f"\n🔍 Analyzing {len(metadata_files)} metadata files...")
    
    for meta_file in metadata_files:
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Check if any critical field is missing
        needs_enhancement = any(
            check_func(metadata) 
            for check_func in LOW_CONFIDENCE_CRITERIA.values()
        )
        
        if needs_enhancement:
            # Get corresponding text file
            text_file = meta_file.replace('_metadata.json', '.txt')
            if os.path.exists(text_file):
                low_confidence.append(text_file)
    
    print(f"✓  Found {len(low_confidence)} files needing LLM enhancement")
    return low_confidence


def extract_with_llm(text: str, filename: str) -> Dict[str, Any]:
    """Use LLM to extract structured metadata from pension file text."""
    
    # Truncate text to first 4000 chars to stay within token limits
    text_sample = text[:4000]
    
    prompt = f"""Extract structured metadata from this Revolutionary War pension document.

Document: {filename}

Text:
{text_sample}

Extract the following information and return as JSON:
{{
  "veteran_name": "Full name of the veteran (first and last)",
  "rank": "Military rank (e.g., Private, Captain, Major)",
  "unit": "Military unit or regiment",
  "wife_name": "Name of wife or widow",
  "wife_age": "Age of wife (number only)",
  "marriage_date": "Marriage date (YYYY or YYYY-MM-DD format)",
  "children": [
    {{"name": "Child name", "age": age_number, "married": true/false, "spouse": "Spouse name if married"}}
  ],
  "pension_amount": "Pension amount (number only, e.g., 27.22)",
  "pension_amount_units": "dollars_per_month or dollars_per_year or dollars_per_annum",
  "service_duration_months": "Service duration in months (number only)",
  "battles_mentioned": ["Battle name 1", "Battle name 2"]
}}

Rules:
- Only extract information explicitly stated in the text
- Use null for missing information
- For names, extract first name only if full name not clear
- Correct obvious OCR errors (e.g., "Torry" → "Torrey")
- Return valid JSON only, no explanation"""

    try:
        if not OPENAI_API_KEY:
            print("⚠️  OPENAI_API_KEY not set. Skipping LLM extraction.")
            return {}
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a historical document extraction expert. Extract structured data accurately from Revolutionary War pension documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"⚠️  Error processing {filename}: {e}")
        return {}


def merge_metadata(existing: Dict[str, Any], llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge LLM-extracted data with existing metadata, preferring non-null values."""
    
    merged = existing.copy()
    
    # Update fields where existing is None and LLM has data
    for key, value in llm_data.items():
        if value is not None and (existing.get(key) is None or existing.get(key) == []):
            merged[key] = value
    
    # Add LLM enhancement metadata
    merged['llm_enhanced'] = True
    merged['llm_enhancement_date'] = datetime.now().isoformat()
    merged['enrichment_version'] = '1.1'
    
    return merged


def main(analyze_only: bool = False):
    """Main execution: identify low-confidence files and enhance with LLM."""
    
    print("="*70)
    print("📜 LLM-BASED PENSION FILE ENHANCEMENT")
    print("="*70)
    
    # Step 1: Identify files needing enhancement
    low_confidence_files = identify_low_confidence_files()
    
    if not low_confidence_files:
        print("\n✓  All files have high-confidence extractions. No LLM enhancement needed.")
        return
    
    # Show breakdown by missing field
    print("\n📊 Missing Fields Breakdown:")
    missing_counts = {'veteran_name': 0, 'family_info': 0, 'pension_amount': 0}
    
    for text_file in low_confidence_files:
        meta_file = text_file.replace('.txt', '_metadata.json')
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for field, check_func in LOW_CONFIDENCE_CRITERIA.items():
            if check_func(metadata):
                missing_counts[field] += 1
    
    for field, count in missing_counts.items():
        print(f"  {field}: {count} files ({count/len(low_confidence_files)*100:.1f}%)")
    
    if analyze_only or not OPENAI_AVAILABLE:
        print(f"\n✓  Analysis complete. {len(low_confidence_files)} files identified for LLM enhancement.")
        if not OPENAI_AVAILABLE:
            print("\n⚠️  Install openai package to run LLM enhancement: pip install openai")
        return
    
    print(f"\n⏳ Processing {len(low_confidence_files)} files with LLM...")
    print(f"   Model: {MODEL}")
    print(f"   Estimated cost: ${len(low_confidence_files) * 0.0002:.2f} (approximate)")
    
    # Ask for confirmation
    response = input("\nProceed with LLM enhancement? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 2: Process files with LLM
    enhanced_count = 0
    
    for i, text_file in enumerate(low_confidence_files, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(low_confidence_files)} files...")
        
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract with LLM
        llm_data = extract_with_llm(text, os.path.basename(text_file))
        
        if not llm_data:
            continue
        
        # Load existing metadata
        meta_file = text_file.replace('.txt', '_metadata.json')
        with open(meta_file, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
        
        # Merge and save
        merged_metadata = merge_metadata(existing_metadata, llm_data)
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=2)
        
        enhanced_count += 1
    
    print(f"\n✓  Enhanced {enhanced_count} files with LLM extraction")
    print(f"✓  Metadata files updated with llm_enhanced flag")


if __name__ == "__main__":
    main()
