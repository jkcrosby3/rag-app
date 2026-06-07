from pathlib import Path
import sys
sys.path.append('scripts')
from enrich_pension_files import extract_veteran_name

file_id = '144125303'
base_dir = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files')
text_file = base_dir / f"{file_id}.txt"

with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"File: {file_id}")
print(f"First 500 chars:\n{text[:500]}\n")

veteran_name = extract_veteran_name(text, file_id)

if veteran_name:
    print(f"✓ Extracted name: {veteran_name}")
else:
    print(f"✗ NO NAME FOUND")
