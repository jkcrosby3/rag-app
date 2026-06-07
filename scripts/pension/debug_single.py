import re

title = 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 43,733, for John Majory, Massachusetts'

# Test in order
patterns = [
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),', 'for [2-word],'),
    (r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)', 'for [3-word] (no comma)'),
]

print(f"Title: {title}\n")

for pattern, desc in patterns:
    match = re.search(pattern, title)
    if match:
        print(f"✓ {desc}: '{match.group(1)}'")
    else:
        print(f"✗ {desc}")
