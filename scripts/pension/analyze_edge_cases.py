import re
from pathlib import Path

# Read the 6 remaining files and analyze their content
base_dir = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files')

edge_cases = {
    '111759201': 'Massanello Womack',
    '144059163': 'John Whiting',
    '144119845': 'Abner Mack',
    '144125303': 'John Majory',
    '144125569': 'John Majors',
    '144125879': 'John Majors',
}

print("=" * 80)
print("EDGE CASE ANALYSIS")
print("=" * 80)

for file_id, expected_name in edge_cases.items():
    text_file = base_dir / f"{file_id}.txt"
    
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n{file_id}: {expected_name}")
    print("-" * 80)
    
    # Show title line
    for i, line in enumerate(lines[:10], 1):
        if 'title:' in line:
            print(f"  Title (line {i}): {line.strip()}")
            break
    
    # Find lines with the name parts
    firstname = expected_name.split()[0]
    lastname = expected_name.split()[-1]
    
    print(f"\n  Looking for '{firstname}' and '{lastname}':")
    
    for i, line in enumerate(lines[:50], 1):
        line_clean = line.strip()
        if firstname in line_clean or lastname in line_clean:
            print(f"    Line {i}: {line_clean}")
