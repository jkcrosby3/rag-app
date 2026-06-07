import json
from pathlib import Path

# Directory containing metadata files
metadata_dir = Path(r'C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app\data\documents\smithsonian\pension_files\text_files')

missing_names = []

# Check all metadata files
for metadata_file in metadata_dir.glob('*.json'):
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Check if veteran_name is missing or empty
        if not metadata.get('veteran_name'):
            # Get the corresponding text file name
            text_file = metadata_file.stem  # Remove .json extension
            missing_names.append(text_file)
    except Exception as e:
        print(f"Error reading {metadata_file}: {e}")

# Sort the list
missing_names.sort()

print(f"Found {len(missing_names)} files missing veteran names:\n")
for name in missing_names:
    print(name)

# Save to file
output_file = metadata_dir / 'missing_veteran_names.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Files missing veteran names ({len(missing_names)} total):\n\n")
    for name in missing_names:
        f.write(f"{name}\n")

print(f"\n✓ Saved list to: {output_file}")
