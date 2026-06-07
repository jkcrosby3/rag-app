import re

# Test why existing patterns aren't matching
test_cases = {
    '111759201': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 16,584, Massanello Womack, Va.',
    '144125303': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 43,733, for John Majory, Massachusetts',
    '144125569': 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 11,026, for John Majors, Maryland',
}

# Current patterns from enrich_pension_files.py
patterns = [
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'for [2-word],'),
    (r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', ', [2-word],'),
]

for file_id, title in test_cases.items():
    print(f"\n{file_id}:")
    print(f"  Title: {title}")
    
    for pattern, desc in patterns:
        match = re.search(pattern, title)
        if match:
            print(f"  ✓ {desc}: '{match.group(1)}'")
            break
        else:
            print(f"  ✗ {desc}")
