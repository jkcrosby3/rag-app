import json
from pathlib import Path
import sys
sys.path.append('scripts')
from enrich_pension_files import extract_veteran_name

# The final 6 files missing names
missing_files = [
    '111759201',
    '144059163',
    '144119845',
    '144125303',
    '144125569',
    '144125879',
]

base_dir = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files')

print("Testing veteran name extraction on final 6 files:\n")
print("=" * 80)

found = 0
still_missing = []

for file_id in missing_files:
    text_file = base_dir / f"{file_id}.txt"
    
    if not text_file.exists():
        print(f"❌ {file_id}: File not found")
        continue
    
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Try to extract veteran name
    veteran_name = extract_veteran_name(text, file_id)
    
    if veteran_name:
        print(f"✓ {file_id}: {veteran_name}")
        found += 1
    else:
        print(f"✗ {file_id}: NO NAME FOUND")
        still_missing.append(file_id)

print("=" * 80)
print(f"\nResults: {found}/{len(missing_files)} names found")
print(f"Still missing: {len(still_missing)} files")

if still_missing:
    print(f"\nFiles still missing names:")
    for file_id in still_missing:
        print(f"  - {file_id}")
else:
    print("\n🎉 ALL FILES NOW HAVE VETERAN NAMES!")
