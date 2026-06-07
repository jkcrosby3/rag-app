import sys
sys.path.append('scripts')
from pathlib import Path

# Test file 144125303
file_id = '144125303'
text_file = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files') / f"{file_id}.txt"

with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Add debug output to extract_veteran_name
import re

# Check multi-veteran
warrant_count = text.count('WARRANT\nNUMBER')
print(f"Warrant count: {warrant_count}")

# Test the pattern directly
pattern = r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),'
match = re.search(pattern, text[:2000], re.MULTILINE)

if match:
    print(f"✓ Pattern matched!")
    print(f"  Raw match: {repr(match.group(0)[:100])}")
    print(f"  Group 1: {repr(match.group(1))}")
    print(f"  Groups: {match.groups()}")
    
    # Simulate the name extraction logic
    if len(match.groups()) >= 2 and match.group(2):
        print(f"  → Would use Lastname, Firstname format")
    else:
        name = match.group(1).strip()
        print(f"  → Using group 1: {repr(name)}")
        
        # Apply cleaning
        name_cleaned = re.sub(r',?\s*(Continental|Private|Captain|Major|Colonel|Deceased|Mass|Conn|Line).*$', '', name, flags=re.IGNORECASE)
        print(f"  → After cleaning: {repr(name_cleaned)}")
else:
    print("✗ Pattern did not match")

# Now test the actual function
from enrich_pension_files import extract_veteran_name
result = extract_veteran_name(text, file_id)
print(f"\nActual extract_veteran_name result: {repr(result)}")
