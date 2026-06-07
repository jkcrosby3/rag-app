"""Count unique soldiers in pension files by analyzing the actual source documents."""
import re
from pathlib import Path
from collections import defaultdict
import json

def extract_soldier_info(text):
    """Extract soldier name and record ID from pension file text."""
    soldier_info = {}
    
    # Extract Record ID
    record_match = re.search(r'Record ID:\s*(\d+)', text)
    if record_match:
        soldier_info['record_id'] = record_match.group(1)
    
    # Extract title which often contains the soldier's name
    title_match = re.search(r'title:\s*(.+?)(?:\n|$)', text)
    if title_match:
        title = title_match.group(1).strip()
        soldier_info['title'] = title
        
        # Try to extract name from title
        # Pattern: "Name, Rank" or "Name of Place"
        name_patterns = [
            r'^Revolutionary War Pension.*?(?:File|Application)\s+(?:B\.L\.\s*Wt\.\s*[\d,-]+\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),',
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, title)
            if name_match:
                soldier_info['name'] = name_match.group(1).strip()
                break
    
    return soldier_info

def main():
    """Count unique soldiers in pension files."""
    pension_dir = Path('data/documents/smithsonian/pension_files/text_files')
    
    if not pension_dir.exists():
        print(f"Error: Directory not found: {pension_dir}")
        return
    
    print(f"Scanning pension files in: {pension_dir}")
    print("=" * 80)
    
    soldiers = {}
    files_processed = 0
    files_with_names = 0
    
    # Process all .txt files
    for txt_file in sorted(pension_dir.glob('*.txt')):
        files_processed += 1
        
        if files_processed % 1000 == 0:
            print(f"Processed {files_processed} files, found {files_with_names} with soldier names...")
        
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            info = extract_soldier_info(text)
            
            if 'record_id' in info:
                record_id = info['record_id']
                
                if 'name' in info:
                    soldiers[record_id] = {
                        'name': info['name'],
                        'title': info.get('title', ''),
                        'file': txt_file.name
                    }
                    files_with_names += 1
                else:
                    # Store record without name
                    soldiers[record_id] = {
                        'name': None,
                        'title': info.get('title', ''),
                        'file': txt_file.name
                    }
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS:")
    print(f"  Total pension files processed: {files_processed}")
    print(f"  Files with soldier names extracted: {files_with_names}")
    print(f"  Unique record IDs: {len(soldiers)}")
    print("=" * 80)
    
    # Show sample of soldiers found
    print("\nSample of soldiers found (first 20):")
    count = 0
    for record_id, info in sorted(soldiers.items()):
        if info['name']:
            print(f"  {record_id}: {info['name']}")
            count += 1
            if count >= 20:
                break
    
    # Save results
    output_file = Path('data/soldier_names.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(soldiers, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Statistics
    named_soldiers = sum(1 for s in soldiers.values() if s['name'])
    print(f"\nStatistics:")
    print(f"  Soldiers with names: {named_soldiers}")
    print(f"  Records without extracted names: {len(soldiers) - named_soldiers}")

if __name__ == '__main__':
    main()
