import re
from pathlib import Path

# Read the 4 files and check their title lines
base_dir = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files')

files = ['111759201', '144125303', '144125569', '144125879']

# Patterns that should match
patterns = [
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'for [2-word],'),
    (r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', ', [2-word],'),
]

for file_id in files:
    text_file = base_dir / f"{file_id}.txt"
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract title line
    title_line = None
    for line in text.split('\n'):
        if line.startswith('title:'):
            title_line = line
            break
    
    print(f"\n{file_id}:")
    if title_line:
        print(f"  Title: {repr(title_line)}")
        
        for pattern, desc in patterns:
            match = re.search(pattern, title_line)
            if match:
                print(f"  ✓ {desc}: '{match.group(1)}'")
                break
            else:
                print(f"  ✗ {desc}")
    else:
        print("  ❌ No title line found")
