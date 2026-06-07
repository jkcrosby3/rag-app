import sys
sys.path.append('scripts')
from enrich_pension_files import extract_veteran_name
from pathlib import Path

# Test file 144059163
file_id = '144059163'
text_file = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files') / f"{file_id}.txt"

with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

veteran_name = extract_veteran_name(text, file_id)

print(f"File {file_id}:")
print(f"  Veteran name: {veteran_name}")
print(f"\nWarrant count check:")
print(f"  'WARRANT\\nNUMBER' occurrences: {text.count('WARRANT' + chr(10) + 'NUMBER')}")
