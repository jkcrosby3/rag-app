import re
from pathlib import Path

# Test the specific patterns
test_files = {
    '111759201': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 16,584, Massanello Womack, Va.',
    '144045355': 'title: Revolutionary War Pension and Bounty Land Warrant Application File B.L.Wt. 1139-200, for Thomas Tredwell Jackson',
    '144058310': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 5,614, for William Storke Jett, Virginia',
    '144064129': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 38885, for Elisha Edwards Johnson, South Carolina',
}

patterns = [
    (r'title:.*?File\s+[SWBR]\.?\s*[LW][t.]*\s*[\d,-]+,\s+for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)', 'File [num], for [3-word name] (no comma)'),
    (r'title:.*?File\s+[SWBR]\.?\s*[LW]?[t.]?\s*[\d,.-]+,\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+),', 'File [num], [3-word name], (with comma)'),
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)', 'for [3-word name] (no comma)'),
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'for [2-word name]'),
    (r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', ', [2-word name],'),
]

for file_id, title_line in test_files.items():
    print(f"\n{file_id}:")
    print(f"  Title: {title_line}")
    print(f"  Testing patterns:")
    
    found = False
    for pattern, desc in patterns:
        match = re.search(pattern, title_line)
        if match:
            print(f"    ✓ {desc}: '{match.group(1)}'")
            found = True
            break
        else:
            print(f"    ✗ {desc}")
    
    if not found:
        print(f"    ❌ NO PATTERN MATCHED")
