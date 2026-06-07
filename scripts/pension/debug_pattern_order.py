import re
from pathlib import Path

# Test file 144125303
file_id = '144125303'
text_file = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files') / f"{file_id}.txt"

with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

# All patterns from enrich_pension_files.py
patterns = [
    (r'(?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la|\s+l[ae])?|Du|Le|La|Del|O\'|Mc|Mac)\s+([A-Z][a-z]+),\s+([A-Z][a-z]+)', 'European prefix, Lastname Firstname'),
    (r'([A-Z][a-z]+)\s+(?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la)?|Du|Le|La|Del)\s+([A-Z][a-z]+)', 'Firstname European Lastname'),
    (r'NAME\s+((?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la)?|Du|Le|La|Del|O\'|Mc|Mac)\s+[A-Z][a-z]+),\s+([A-Z][a-z]+)', 'NAME European'),
    (r'title:.*?File\s+[SWBR]\.?\s*[LW][t.]*\s*[\d,-]+,\s+for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+),', 'title File for 3-word,'),
    (r'title:.*?File\s+[SWBR]\.?\s*[LW]?[t.]?\s*[\d,.-]+,\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+),', 'title File 3-word,'),
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+),', 'title for 3-word with initial,'),
    (r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+),', 'title , 3-word,'),
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'title for 2-word,'),
    (r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'title , 2-word,'),
]

print(f"Testing file {file_id}:")
print(f"First 500 chars:\n{text[:500]}\n")
print("=" * 80)

for pattern, desc in patterns:
    match = re.search(pattern, text[:2000], re.MULTILINE)
    if match:
        print(f"✓ MATCH: {desc}")
        print(f"  Pattern: {pattern}")
        print(f"  Groups: {match.groups()}")
        print(f"  Full match: {repr(match.group(0)[:100])}")
        break
    else:
        print(f"✗ {desc}")
