import re

title = 'title: Revolutionary War Pension and Bounty Land Warrant Application File S. 43,733, for John Majory, Massachusetts'

# Line 80 pattern
pattern = r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+),'

match = re.search(pattern, title)
if match:
    print(f"Line 80 pattern MATCHES: '{match.group(1)}'")
else:
    print("Line 80 pattern does NOT match")

# Simpler two-word pattern
pattern2 = r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),'
match2 = re.search(pattern2, title)
if match2:
    print(f"Two-word pattern MATCHES: '{match2.group(1)}'")
