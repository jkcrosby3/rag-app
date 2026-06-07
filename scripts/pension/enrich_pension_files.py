#!/usr/bin/env python3
"""
Enrich Pension Files with Veteran Metadata

Extracts veteran information from pension file text and creates enriched metadata JSON files.
This script processes existing text files and adds structured metadata including:
- Veteran name
- Military rank
- Unit/Regiment
- Service dates
- Pension information
- State information

Usage:
    # Process all pension files
    python scripts/enrich_pension_files.py
    
    # Process specific number of files (for testing)
    python scripts/enrich_pension_files.py --limit 100
    
    # Specify custom directory
    python scripts/enrich_pension_files.py --input-dir "data/documents/smithsonian/pension_files/text_files"
    
    # Dry run (don't save, just show what would be extracted)
    python scripts/enrich_pension_files.py --dry-run --limit 10
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Military ranks to search for (in order of precedence)
RANKS = [
    'Major General', 'Brigadier General', 'General',
    'Colonel', 'Lieutenant Colonel', 'Lt Colonel', 'Lt. Colonel',
    'Major',
    'Captain', 'Capt', 'Capt.',
    'Lieutenant', 'First Lieutenant', '1st Lieutenant', 'Second Lieutenant', '2nd Lieutenant', 'Lt', 'Lt.',
    'Ensign',
    'Sergeant', 'Sergt', 'Sergt.', 'Sgt', 'Sgt.',
    'Corporal', 'Corp', 'Corp.',
    'Private', 'Pvt', 'Pvt.', 'Pri', 'Pri.',
    'Drummer', 'Fifer',
    'Seaman', 'Sailor', 'Mariner'  # Naval ranks
]

# States (for service and residence)
STATES = [
    'Massachusetts', 'Virginia', 'Pennsylvania', 'New York', 'Maryland',
    'Connecticut', 'New Jersey', 'South Carolina', 'North Carolina', 'Georgia',
    'New Hampshire', 'Rhode Island', 'Delaware', 'Vermont', 'Maine'
]


def extract_veteran_name(text: str, filename: str) -> Optional[str]:
    """Extract veteran name from text or filename."""
    
    # Check if this is a NARA administrative document with no veteran data
    if 'NARA Archival Administrative Sheets' in text:
        return '[Administrative Document - No Veteran Data]'
    
    # Check if this is a multi-veteran file (multiple warrant cards in one record)
    warrant_count = text.count('WARRANT\nNUMBER')
    if warrant_count >= 2:
        return '[Multiple Veterans - Incomplete Records]'
    
    # Try to extract from title line in text
    patterns = [
        # Multi-word surnames with European prefixes
        # "Van Buskirk, Peter" or "Nicholas Van Alstine" (Dutch/German/French)
        r'(?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la|\s+l[ae])?|Du|Le|La|Del|O\'|Mc|Mac)\s+([A-Z][a-z]+),\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+(?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la)?|Du|Le|La|Del)\s+([A-Z][a-z]+)',
        # "NAME Van Buskirk, Peter" from card format
        r'NAME\s+((?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la)?|Du|Le|La|Del|O\'|Mc|Mac)\s+[A-Z][a-z]+),\s+([A-Z][a-z]+)',
        # Three-word names with middle name or initial (First Middle Last or First M. Last)
        # Check patterns with trailing commas FIRST to avoid greedy matching
        # "File S. 5,614, for William Storke Jett, Virginia" format (with trailing comma)
        r'title:.*?File\s+[SWBR]\.?\s*[LW][t.]*\s*[\d,-]+,\s+for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+),',
        # "File W. 22,526, John Henry Waufle, New York" format (without "for")
        r'title:.*?File\s+[SWBR]\.?\s*[LW]?[t.]?\s*[\d,.-]+,\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+),',
        r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+),',
        r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+),',
        # "title: ... for Joel Hunt, Connecticut" from header (two-word names with comma)
        r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+),',
        # "title: ... John Townsend, Continental" or "Denis Trammell, Ga. S.C." (two-word names)
        r'title:.*?,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),',
        # "File S. 5,614, for William Storke Jett, Virginia" - three-word name WITHOUT trailing comma (greedy, check last)
        r'title:.*?for\s+([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        # "Ga.         Trammell, Denis       R.10672" structured format
        r'^[A-Z][a-z]+\.\s+([A-Z][a-z]+),\s+([A-Z][a-z]+)\s+[RWS]\.',
        # "Trask Jesse" format (Lastname Firstname without comma)
        r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)$',
        # "A Mack, Abner" format with letter prefix (bounty land warrant cards)
        r'^[A-Z]\s+([A-Z][a-z]+),\s+([A-Z][a-z]+)',
        # "Johnson, Benjamin Private" or "Johnson, James Quartermaster Sergeant" (bounty land warrant cards)
        r'^([A-Z][a-z]+),\s+([A-Z][a-z]+)\s+(?:Private|Sergeant|Lieutenant|Captain|Major|Colonel|Quartermaster|Corporal|Ensign)',
        # "Wheeler, Benjamin" format (lines 14-16 in structured section)
        r'^([A-Z][a-z]+),\s+([A-Z][a-z]+)$',
        # "Benjamin Wheeler of Boston" narrative format
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+of\s+(?:the\s+)?(?:City|Town|County)\s+of',
        # "File S. 42,508, John Torrey, Continental"
        r'File\s+[SBW]\.?\s*[\d,-]+,?\s*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+),',
        # "B. L. Wt. 2,153-400, Joseph Torrey, Continental" (with spaces)
        r'B\.\s*L\.\s*Wt\.\s*[\d,-]+,\s*([A-Z][a-z]+\s+[A-Z][a-z]+),',
        # "Roll 2401\nTorrey, Asa"
        r'Roll \d+\s*\n([A-Z][a-z]+,\s*[A-Z][a-z]+)',
        # Last resort: any proper name after pension/file keywords
        r'(?:Pension|File).*?([A-Z][a-z]{2,}(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]{2,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.MULTILINE)  # Search first 2000 chars
        if match:
            # Handle "Lastname, Firstname" format
            if len(match.groups()) >= 2 and match.group(2):
                # Check if this is a multi-word surname pattern (has prefix in match)
                full_match = match.group(0)
                if any(prefix in full_match for prefix in ['Van ', 'Von ', 'De ', 'Du ', 'Le ', 'La ', 'Del ', "O'", 'Mc', 'Mac']):
                    # For "Van Buskirk, Peter" → extract full surname with prefix
                    # match.group(1) has the main part, need to get prefix from full match
                    prefix_match = re.search(r'((?:Van(?:\s+der|\s+den)?|Von(?:\s+der)?|De(?:\s+la)?|Du|Le|La|Del|O\'|Mc|Mac)\s+[A-Z][a-z]+)', full_match)
                    if prefix_match:
                        lastname = prefix_match.group(1)
                        firstname = match.group(2) if 'NAME' in pattern else match.group(2)
                        name = f"{firstname} {lastname}"
                    else:
                        name = f"{match.group(2)} {match.group(1)}"
                else:
                    # Format: "Wheeler, Benjamin" → "Benjamin Wheeler"
                    name = f"{match.group(2)} {match.group(1)}"
            else:
                name = match.group(1).strip()
            
            # Clean up common suffixes and invalid names
            name = re.sub(r',?\s*\b(Continental|Private|Captain|Major|Colonel|Deceased|Mass|Conn|Line)\b.*$', '', name, flags=re.IGNORECASE)
            # Skip invalid names
            if name.lower() in ['and bounty', 'microfilm target', 'bounty land', 'war pension']:
                continue
            if len(name) > 5:  # Must be reasonable length
                return name.strip()
    
    return None


def extract_rank(text: str) -> Optional[str]:
    """Extract military rank from text."""
    
    # Search for rank patterns
    for rank in RANKS:
        # Look for rank in various contexts
        patterns = [
            rf'\b{re.escape(rank)}\b\s+in\s+(?:the\s+)?(?:Col\.|Colonel)',  # "Major in the Col. Hazen's"
            rf'(?:was\s+a\s+|as\s+a\s+|rank\s+of\s+){re.escape(rank)}\b',  # "was a Major"
            rf'\b{re.escape(rank)}\b\s+(?:of|in)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)|Col\.)',  # "Captain of the 1st"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize rank name
                rank_normalized = rank.replace('.', '').strip()
                if rank_normalized.lower() in ['lt', 'lieut']:
                    return 'Lieutenant'
                elif rank_normalized.lower() in ['capt']:
                    return 'Captain'
                elif rank_normalized.lower() in ['sgt', 'sergt']:
                    return 'Sergeant'
                elif rank_normalized.lower() in ['corp']:
                    return 'Corporal'
                elif rank_normalized.lower() in ['pvt', 'pri']:
                    return 'Private'
                return rank_normalized
    
    return None


def extract_unit(text: str) -> Optional[str]:
    """Extract military unit/regiment or naval vessel from text."""
    
    patterns = [
        r"(?:ship|vessel|board)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "ship Black Prince", "board the Black Prince"
        r"(Col\.\s+[A-Z][a-z]+(?:'s)?\s+Regiment)",  # "Col. Hazen's Regiment"
        r"(Colonel\s+[A-Z][a-z]+(?:'s)?\s+Regiment)",  # "Colonel Hazen's Regiment"
        r"(\d+(?:st|nd|rd|th)\s+Regiment)",  # "1st Regiment"
        r"(\d+(?:st|nd|rd|th)\s+[A-Z][a-z]+\s+Regiment)",  # "1st Virginia Regiment"
        r"(Continental\s+(?:Army|Line|Troops))",  # "Continental Army"
        r"([A-Z][a-z]+\s+(?:Line|Militia|Troops))",  # "Virginia Line", "Pennsylvania Militia"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            unit = match.group(1).strip()
            # For naval vessels, add "ship" prefix if not already present
            if 'ship' in pattern.lower() or 'board' in pattern.lower():
                return f"Ship {unit}"
            return unit
    
    return None


def extract_pension_number(text: str) -> Optional[str]:
    """Extract pension or bounty land warrant number with classification letter."""
    
    patterns = [
        r'([RSWB])\.?\s*([\d,]+)',  # "R.10672" or "S. 42,508" (R=Rejected, S=Service, W=Widow, B=Bounty)
        r'B\.L\.\s*Wt\.\s*([\d,-]+)',  # "B.L. Wt. 2,153-400"
        r'Certificate.*?(?:No\.|Number|numbered)\s*(\d+)',  # "Certificate No. 10231"
        r'Pension\s+(?:No\.|Number)\s*(\d+)',  # "Pension No. 12345"
        r'Warrant\s+(?:No\.|Number)\s*(\d+)',  # "Warrant No. 12345"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # For R/S/W/B patterns, include the letter prefix
            if len(match.groups()) >= 2 and match.group(1) in ['R', 'S', 'W', 'B']:
                return f"{match.group(1)}.{match.group(2).strip()}"
            else:
                return match.group(1).strip()
    
    return None


def extract_pension_amount(text: str) -> Dict[str, Optional[str]]:
    """Extract pension amount/rate with units."""
    
    result = {
        'pension_amount': None,
        'pension_amount_units': None
    }
    
    patterns = [
        r'(\d+)\s+Dollars?\s+(\d+)\s+Cents?\s+per\s+(annum|year|month)',  # "467 Dollars 82 Cents per annum"
        r'\$(\d+),(\d+)½?\s+per\s+(annum|year|month)',  # "$36,12½ per annum" (with fraction)
        r'rate of\s+([\d.]+)\s*(Dollars?)\s+per\s+(month|year|annum)',  # "rate of Eight Dollars per month"
        r'rate of\s+\$([\d.]+)\s+per\s+(month|year|annum)',  # "rate of $8 per month"
        r'(\d+)\s+Dollars?\s+per\s+(month|year|annum)',  # "8 Dollars per month" (without "rate of")
        r'\$(\d+\.\d{2})\s+per\s+(month|year|annum)',  # "$8.00 per month"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Check if this is the "$XX,YY½" format (with comma and optional fraction)
            if ',' in pattern and len(match.groups()) >= 3:
                # Format: "$36,12½ per annum"
                dollars = match.group(1)
                cents = match.group(2)
                amount = f"{dollars}.{cents}"
                period = match.group(3)
            # Check if this is the "Dollars Cents" format
            elif len(match.groups()) >= 3 and match.group(2) and match.group(2).isdigit():
                # Format: "27 Dollars 22 Cents per annum"
                dollars = match.group(1)
                cents = match.group(2)
                amount = f"{dollars}.{cents}"
                period = match.group(3)
            else:
                amount = match.group(1)
                # Convert word numbers to digits
                word_to_num = {
                    'eight': '8', 'ten': '10', 'twelve': '12', 'fifteen': '15',
                    'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50'
                }
                for word, num in word_to_num.items():
                    if word in amount.lower():
                        amount = num
                
                # Extract units (dollars per month/year/annum)
                if len(match.groups()) >= 3 and 'Dollar' in str(match.group(2)):
                    period = match.group(3)
                elif len(match.groups()) >= 2:
                    period = match.group(2)
                else:
                    period = 'month'
            
            result['pension_amount'] = amount
            result['pension_amount_units'] = f'dollars_per_{period.lower()}'
            break
    
    return result


def extract_pension_issue_info(text: str) -> Dict[str, Optional[str]]:
    """Extract pension issue date and location."""
    
    info = {
        'pension_issue_date': None,
        'pension_issue_location': None
    }
    
    # Look for "Certificate of Pension issued the 4 of May 1819"
    issue_pattern = r'Certificate.*?issued.*?(\d{1,2}).*?(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[.,\s]*(\d{4})'
    match = re.search(issue_pattern, text, re.IGNORECASE)
    if match:
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        info['pension_issue_date'] = f"{year}-{month}-{day}"
    
    # Look for "sent to [agent], [city] [state]"
    location_pattern = r'sent to.*?,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([A-Z][a-z.]+)'
    match = re.search(location_pattern, text)
    if match:
        city = match.group(1)
        state = match.group(2)
        info['pension_issue_location'] = f"{city}, {state}"
    
    return info


def extract_property_value(text: str) -> Optional[str]:
    """Extract total property value from schedule."""
    
    patterns = [
        r'total amount in value.*?is\s+([\w\s]+?)(?:dollars|cents)',  # "total amount in value...is seventeen cents"
        r'property.*?schedule.*?\$(\d+\.\d{2})',  # "property...schedule...$17.66"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


def extract_death_info(text: str) -> Dict[str, Optional[str]]:
    """Extract death information."""
    
    info = {
        'death_date': None,
        'death_context': None
    }
    
    # Look for death dates
    death_patterns = [
        r'Died\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[.,\s]*\d{4})',
        r'died\s+(?:on\s+)?(?:the\s+)?(\d{1,2}).*?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[.,\s]*(\d{4})',
        r'(?:killed|slain)\s+(?:in|at|during)\s+(?:the\s+)?(?:battle|action)\s+(?:of\s+)?([A-Z][a-z]+)',
    ]
    
    for pattern in death_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info['death_date'] = match.group(0).strip()
            # Check context for how they died
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end]
            
            if re.search(r'killed|slain|battle|action|wound', context, re.IGNORECASE):
                info['death_context'] = 'died_in_service'
            elif re.search(r'pensioner|after.*war', context, re.IGNORECASE):
                info['death_context'] = 'died_as_pensioner'
            break
    
    # Check for widow's pension
    if re.search(r'widow.*pension|pension.*widow', text, re.IGNORECASE):
        if not info['death_context']:
            info['death_context'] = 'widow_pension'
    
    return info


def parse_family_record_section(text: str, veteran_surname: Optional[str]) -> Dict[str, Any]:
    """Parse structured FAMILY RECORD sections with marriages and births."""
    
    info = {
        'marriages': [],  # {child_name, spouse_name, marriage_year}
        'births': [],     # {name, birth_date, is_parent}
    }
    
    # Look for MARRIAGES section
    marriages_match = re.search(r'MARRIAGES\.(.*?)(?:BIRTHS|FAMILY RECORD|$)', text, re.IGNORECASE | re.DOTALL)
    if marriages_match:
        marriages_text = marriages_match.group(1)
        
        # Pattern: "Name1 Name2 was married to Name3 Name4 year"
        # OCR variations: "Marie", "Marrieto", "maried"
        marriage_pattern = r'([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:was|W)\s+[Mm]ar[ir]e?d?\s*to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*.*?(\d{4})'
        
        matches = re.findall(marriage_pattern, marriages_text)
        for match in matches:
            first_name = match[0]
            surname = match[1]
            spouse = match[2]
            year = match[3]
            
            # Correct surname if it's mangled and we know veteran's surname
            if veteran_surname and surname.lower() not in ['torrey', veteran_surname.lower()]:
                # Check if it's a fuzzy match (OCR error)
                if len(surname) >= 4 and (surname[:2] == veteran_surname[:2] or surname[-2:] == veteran_surname[-2:]):
                    surname = veteran_surname
            
            info['marriages'].append({
                'child_name': f"{first_name} {surname}",
                'spouse_name': spouse,
                'marriage_year': year
            })
    
    # Look for BIRTHS section
    births_match = re.search(r'BIRTHS\.(.*?)(?:MARRIAGES|$)', text, re.IGNORECASE | re.DOTALL)
    if births_match:
        births_text = births_match.group(1)
        
        # Pattern: "Name1 Name2 Was Born month day year"
        birth_pattern = r'([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:W\s+)?[Ww]as\s+[Bb]or[ne]+\s+([A-Za-z]+\s+\d{1,2})\s+.*?(\d{4})'
        
        matches = re.findall(birth_pattern, births_text)
        for match in matches:
            first_name = match[0]
            surname = match[1]
            date_part = match[2]
            year = match[3]
            
            # Correct surname
            if veteran_surname and surname.lower() not in ['torrey', veteran_surname.lower()]:
                if len(surname) >= 4:
                    surname = veteran_surname
            
            # First two entries are usually parents
            is_parent = len(info['births']) < 2
            
            info['births'].append({
                'name': f"{first_name} {surname}",
                'birth_date': f"{date_part} {year}",
                'is_parent': is_parent
            })
    
    return info


def extract_family_info(text: str) -> Dict[str, Any]:
    """Extract family member information."""
    
    info = {
        'wife_mentioned': False,
        'wife_name': None,
        'wife_age': None,
        'wife_age_units': None,
        'marriage_date': None,
        'children_count': None,
        'children_names': [],
        'children_details': [],  # List of {name, age, married, spouse}
        'family_members': [],
        'family_record_data': None  # Structured data from FAMILY RECORD sections
    }
    
    # Look for wife name
    wife_name_patterns = [
        r'([A-Z][a-z]+)\s+[A-Z][a-z]+\s*\n\s*widow of',  # "Abigail Torrey\nwidow of"
        r'(?:Widow|widow)\s+([A-Z][a-z]+)\s+[A-Z][a-z]+',  # "widow Abigail Torrey"
        r'(?:his|my)\s+wife\s+named\s+([A-Z][a-z]+)',
        r'wife\s+([A-Z][a-z]+)\s+aged',
        r'widow\s+([A-Z][a-z]+)',
    ]
    
    invalid_wife_names = ['aged', 'named', 'and', 'the', 'of', 'or', 'at', 'in', 'application', 'for', 'to']
    
    for pattern in wife_name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info['wife_mentioned'] = True
            name = match.group(1)
            if name.lower() not in invalid_wife_names and len(name) > 2:
                info['wife_name'] = name
            break
    
    # Look for marriage date
    marriage_patterns = [
        r'(?:married|marriage).*?(\d{1,2}).*?(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[.,\s]*(\d{4})',
        r'(?:married|marriage).*?(\d{4})',
    ]
    
    for pattern in marriage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 3:
                info['marriage_date'] = f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
            else:
                info['marriage_date'] = match.group(1)
            break
    
    # Look for wife age
    wife_age_patterns = [
        r'wife.*?aged\s+(\d+)\s*(years?)?',
        r'wife.*?(\d+)\s+years',
    ]
    
    for pattern in wife_age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info['wife_mentioned'] = True
            if match.group(1).isdigit():
                info['wife_age'] = int(match.group(1))
                if len(match.groups()) >= 2 and match.group(2):
                    info['wife_age_units'] = match.group(2).lower()
                else:
                    info['wife_age_units'] = 'years'
            break
    
    # Look for children names, ages, and married status
    children_patterns = [
        # "son John aged 20", "daughter Mary aged 15"
        r'(?:son|daughter)\s+([A-Z][a-z]+)\s+aged\s+(\d+)',
        # "John aged 20 years"
        r'([A-Z][a-z]+)\s+aged\s+(\d+)\s+years',
        # "son John, 20 years"
        r'(?:son|daughter)\s+([A-Z][a-z]+),?\s+(\d+)\s+years',
    ]
    
    children_found = []
    invalid_names = ['Wife', 'Widow', 'And', 'The', 'His', 'My', 'Of', 'At', 'In', 'Or', 'Application']
    
    for pattern in children_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                name = match[0]
                age = match[1]
                if name not in invalid_names and len(name) > 2:
                    children_found.append({'name': name, 'age': int(age), 'married': False, 'spouse': None})
    
    # Look for married children with spouse names
    # Pattern: "daughter Mary married to John Smith", "son James and his wife Sarah"
    married_patterns = [
        r'(?:son|daughter)\s+([A-Z][a-z]+)\s+(?:married to|wife of|husband of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:son|daughter)\s+([A-Z][a-z]+)\s+and\s+(?:his wife|her husband)\s+([A-Z][a-z]+)',
    ]
    
    for pattern in married_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                child_name = match[0]
                spouse_name = match[1]
                if child_name not in invalid_names and spouse_name not in invalid_names:
                    # Check if child already in list
                    found = False
                    for child in children_found:
                        if child['name'] == child_name:
                            child['married'] = True
                            child['spouse'] = spouse_name
                            found = True
                            break
                    if not found:
                        children_found.append({'name': child_name, 'age': None, 'married': True, 'spouse': spouse_name})
    
    if children_found:
        info['children_names'] = [c['name'] for c in children_found]
        info['children_details'] = children_found
        info['children_count'] = len(children_found)
    else:
        # Fallback: just count children mentioned
        count_patterns = [
            r'(\d+)\s+(?:sons?|daughters?|children)',
            r'(three|four|five|six|seven|eight)\s+(?:sons?|daughters?|children)',
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                count_str = match.group(1)
                if count_str.isdigit():
                    count = int(count_str)
                    # Sanity check: children count should be reasonable (1-20)
                    if 1 <= count <= 20:
                        info['children_count'] = count
                else:
                    # Convert word to number
                    word_to_num = {'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
                    info['children_count'] = word_to_num.get(count_str.lower())
                break
    
    # Parse structured FAMILY RECORD sections if present
    if 'FAMILY RECORD' in text or 'MARRIAGES' in text or 'BIRTHS' in text:
        # Try to extract veteran surname for OCR correction
        veteran_surname = None
        surname_match = re.search(r'([A-Z][a-z]+),\s+[A-Z][a-z]+', text[:2000])  # "Torrey, John" format
        if surname_match:
            veteran_surname = surname_match.group(1)
        
        family_record = parse_family_record_section(text, veteran_surname)
        if family_record['marriages'] or family_record['births']:
            info['family_record_data'] = family_record
            
            # Update children count from family record if not already set
            if not info['children_count'] and family_record['births']:
                # Count non-parent births
                children = [b for b in family_record['births'] if not b.get('is_parent', False)]
                if children:
                    info['children_count'] = len(children)
    
    return info


def extract_military_pay(text: str) -> Dict[str, Optional[str]]:
    """Extract military pay during service (different from pension)."""
    
    result = {
        'military_pay_amount': None,
        'military_pay_units': None
    }
    
    patterns = [
        r'service as (?:a )?(?:private|sergeant|captain|major|colonel|lieutenant)\s+at\s+\$(\d+)\s+per\s+(month)',  # "service as private at $60 per month"
        r'monthly\s+(?:wages|pay|salary).*?(\d+)\s+dollars?\s+per\s+(month)',  # "monthly wages of forty dollars per month"
        r'monthly\s+pay.*?increased.*?(\d+)\s+dollars?\s+per\s+(month)',  # "monthly pay was increased to 140 dollars per month"
        r'(?:army pay|military pay|pay.*?service).*?\$?(\d+\.?\d*)\s*(?:dollars?)?\s*per\s+(month|year|day)',
        r'(?:received|paid).*?(\d+)\s*(?:dollars?|cents)\s*per\s+(month|year|day).*?(?:during|while|in)\s+(?:service|war)',
        r'half\s+pay.*?(\d+)\s*(?:dollars?)\s*per\s+(month|year)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = match.group(1)
            
            # Convert word numbers to digits
            word_to_num = {
                'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
                'one hundred': '100', 'one hundred and forty': '140', 'one hundred and fifty': '150',
                'two hundred': '200', 'three hundred': '300'
            }
            
            amount_lower = amount.lower()
            for word, num in word_to_num.items():
                if word in amount_lower:
                    amount = num
                    break
            
            result['military_pay_amount'] = amount
            if len(match.groups()) >= 2 and match.group(2):
                result['military_pay_units'] = f'dollars_per_{match.group(2).lower()}'
            else:
                result['military_pay_units'] = 'dollars_per_month'
            break
    
    return result


def extract_service_dates(text: str) -> Dict[str, Any]:
    """Extract service start and end dates, duration, and battles."""
    
    dates = {
        'service_start': None,
        'service_end': None,
        'service_duration_months': None,
        'battles_mentioned': []
    }
    
    # Look for date ranges
    date_patterns = [
        r'(?:served|service).*?(?:from\s+)?(\d{4}).*?(?:to|until|through)\s+(\d{4})',
        r'(?:enlisted|entered).*?(\d{4}).*?(?:discharged|ended).*?(\d{4})',
        r'(\d{4})\s*-\s*(\d{4})',  # "1776-1783"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dates['service_start'] = match.group(1)
            dates['service_end'] = match.group(2)
            break
    
    # If no range found, look for individual dates
    if not dates['service_start']:
        enlist_match = re.search(r'(?:enlisted|entered|began).*?(\d{4})', text, re.IGNORECASE)
        if enlist_match:
            dates['service_start'] = enlist_match.group(1)
    
    if not dates['service_end']:
        discharge_match = re.search(r'(?:discharged|ended|concluded).*?(\d{4})', text, re.IGNORECASE)
        if discharge_match:
            dates['service_end'] = discharge_match.group(1)
    
    # Look for service duration in months
    duration_patterns = [
        r'served\s+(?:for\s+)?(\d+)\s+months?',
        r'(?:service|tour).*?(\d+)\s+months?',
        r'(\d+)\s+months?.*?(?:service|tour)',
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dates['service_duration_months'] = int(match.group(1))
            break
    
    # Look for battle mentions
    battle_patterns = [
        r'[Bb]attle of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'[Bb]attle at ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+[Bb]attle',
    ]
    
    battles = []
    for pattern in battle_patterns:
        matches = re.findall(pattern, text)
        for battle in matches:
            if battle not in battles and battle not in ['The', 'And', 'His', 'That']:
                battles.append(battle)
    
    if battles:
        dates['battles_mentioned'] = battles
    
    return dates


def extract_state_info(text: str) -> Dict[str, Optional[str]]:
    """Extract state of service and residence."""
    
    states = {
        'state_of_service': None,
        'state_of_residence': None
    }
    
    # Look for state of service
    for state in STATES:
        service_patterns = [
            rf'State of\s+{state}\s+that\s+(?:payments for )?military\s+services?',  # "State of North Carolina that payments for military services"
            rf'{state}\s+(?:Line|Regiment|Militia|Troops)',
            rf'(?:served in|service in)\s+{state}',
        ]
        for pattern in service_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                states['state_of_service'] = state
                break
        if states['state_of_service']:
            break
    
    # Look for state of residence
    for state in STATES:
        residence_patterns = [
            rf'(?:resident of|residing in|residence)\s+{state}',
            rf'State of\s+{state}',
        ]
        for pattern in residence_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                states['state_of_residence'] = state
                break
        if states['state_of_residence']:
            break
    
    return states


def extract_metadata_from_text(text: str, filename: str) -> Dict[str, Any]:
    """Extract all metadata from pension file text."""
    
    # Extract basic info from header
    record_id = None
    nara_url = None
    page_count = None
    
    header_match = re.search(r'Record ID:\s*(\d+)', text)
    if header_match:
        record_id = header_match.group(1)
    
    url_match = re.search(r'naraURL:\s*(https?://[^\s]+)', text)
    if url_match:
        nara_url = url_match.group(1)
    
    pages_match = re.search(r'numberOfPages:\s*(\d+)', text)
    if pages_match:
        page_count = int(pages_match.group(1))
    
    # Extract veteran information
    veteran_name = extract_veteran_name(text, filename)
    rank = extract_rank(text)
    unit = extract_unit(text)
    pension_number = extract_pension_number(text)
    pension_info = extract_pension_amount(text)
    pension_issue = extract_pension_issue_info(text)
    military_pay = extract_military_pay(text)
    service_dates = extract_service_dates(text)
    state_info = extract_state_info(text)
    property_value = extract_property_value(text)
    death_info = extract_death_info(text)
    family_info = extract_family_info(text)
    
    # Build metadata
    metadata = {
        'document_id': record_id or filename.replace('.txt', ''),
        'file_name': filename,
        'collection': 'pension_files',
        'source_system': 'NARA',
        'nara_url': nara_url,
        'page_count': page_count,
        
        # Enriched veteran information
        'veteran_name': veteran_name,
        'rank': rank,
        'unit': unit,
        'pension_number': pension_number,
        'pension_amount': pension_info['pension_amount'],
        'pension_amount_units': pension_info['pension_amount_units'],
        'pension_issue_date': pension_issue['pension_issue_date'],
        'pension_issue_location': pension_issue['pension_issue_location'],
        'military_pay_amount': military_pay['military_pay_amount'],
        'military_pay_units': military_pay['military_pay_units'],
        'service_start': service_dates['service_start'],
        'service_end': service_dates['service_end'],
        'service_duration_months': service_dates['service_duration_months'],
        'battles_mentioned': service_dates['battles_mentioned'],
        'state_of_service': state_info['state_of_service'],
        'state_of_residence': state_info['state_of_residence'],
        'property_value': property_value,
        'death_date': death_info['death_date'],
        'death_context': death_info['death_context'],
        'wife_mentioned': family_info['wife_mentioned'],
        'wife_name': family_info['wife_name'],
        'wife_age': family_info['wife_age'],
        'wife_age_units': family_info['wife_age_units'],
        'marriage_date': family_info['marriage_date'],
        'children_count': family_info['children_count'],
        'children_names': family_info['children_names'],
        'children_details': family_info['children_details'],
        'family_record_data': family_info['family_record_data'],
        
        # Document metadata
        'document_type': 'pension_application',
        'language': 'en',
        'enrichment_date': datetime.now().isoformat(),
        'enrichment_version': '1.0'
    }
    
    return metadata


def enrich_pension_files(input_dir: Path, limit: int = 0, dry_run: bool = False):
    """Process pension files and create enriched metadata."""
    
    print("\n" + "=" * 70)
    print("📜 PENSION FILE ENRICHMENT")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Dry run: {dry_run}")
    
    # Find all text files
    text_files = list(input_dir.glob("*.txt"))
    
    if not text_files:
        print(f"\n❌ No text files found in {input_dir}")
        return
    
    print(f"\n✓  Found {len(text_files)} text files")
    
    # Limit if specified
    if limit > 0:
        text_files = text_files[:limit]
        print(f"✓  Processing first {limit} files")
    
    # Statistics
    stats = {
        'total': len(text_files),
        'processed': 0,
        'with_name': 0,
        'with_rank': 0,
        'with_unit': 0,
        'with_dates': 0,
        'with_pension_amount': 0,
        'with_military_pay': 0,
        'with_death_info': 0,
        'with_family': 0,
        'errors': 0
    }
    
    # Process each file
    print(f"\n⏳ Processing files...")
    for i, txt_file in enumerate(text_files, 1):
        try:
            # Read text
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            
            # Extract metadata
            metadata = extract_metadata_from_text(text, txt_file.name)
            
            # Update statistics
            stats['processed'] += 1
            if metadata['veteran_name']:
                stats['with_name'] += 1
            if metadata['rank']:
                stats['with_rank'] += 1
            if metadata['unit']:
                stats['with_unit'] += 1
            if metadata['service_start'] or metadata['service_end']:
                stats['with_dates'] += 1
            if metadata['pension_amount'] or metadata['pension_amount_units']:
                stats['with_pension_amount'] += 1
            if metadata['military_pay_amount'] or metadata['military_pay_units']:
                stats['with_military_pay'] += 1
            if metadata['death_date'] or metadata['death_context']:
                stats['with_death_info'] += 1
            if metadata['wife_mentioned'] or metadata['children_count']:
                stats['with_family'] += 1
            
            # Save metadata (unless dry run)
            if not dry_run:
                metadata_file = txt_file.with_name(txt_file.stem + '_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
            
            # Show progress
            if i % 100 == 0 or i == len(text_files):
                print(f"  Processed {i}/{len(text_files)} files...")
            
            # Show first few examples in dry run
            if dry_run and i <= 5:
                print(f"\n--- Example {i}: {txt_file.name} ---")
                print(f"  Veteran: {metadata['veteran_name']}")
                print(f"  Rank: {metadata['rank']}")
                print(f"  Unit: {metadata['unit']}")
                print(f"  Service: {metadata['service_start']} - {metadata['service_end']}")
                print(f"  Pension #: {metadata['pension_number']}")
                print(f"  Pension Amount: {metadata['pension_amount']} ({metadata['pension_amount_units']})")
                print(f"  Military Pay: {metadata['military_pay_amount']} ({metadata['military_pay_units']})")
                print(f"  Issue: {metadata['pension_issue_date']} at {metadata['pension_issue_location']}")
                print(f"  Property: {metadata['property_value']}")
                print(f"  Death: {metadata['death_date']} ({metadata['death_context']})")
                print(f"  Wife: {metadata['wife_name']} (age {metadata['wife_age']} {metadata['wife_age_units']})")
                print(f"  Children: {metadata['children_count']} - {metadata['children_names']}")
        
        except Exception as e:
            stats['errors'] += 1
            print(f"\n⚠️  Error processing {txt_file.name}: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal files: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['processed'] > 0:
        print(f"\nExtraction Success Rates:")
        print(f"  Veteran names: {stats['with_name']} ({stats['with_name']/stats['processed']*100:.1f}%)")
        print(f"  Ranks: {stats['with_rank']} ({stats['with_rank']/stats['processed']*100:.1f}%)")
        print(f"  Units: {stats['with_unit']} ({stats['with_unit']/stats['processed']*100:.1f}%)")
        print(f"  Service dates: {stats['with_dates']} ({stats['with_dates']/stats['processed']*100:.1f}%)")
        print(f"  Pension amounts: {stats['with_pension_amount']} ({stats['with_pension_amount']/stats['processed']*100:.1f}%)")
        print(f"  Military pay: {stats['with_military_pay']} ({stats['with_military_pay']/stats['processed']*100:.1f}%)")
        print(f"  Death info: {stats['with_death_info']} ({stats['with_death_info']/stats['processed']*100:.1f}%)")
        print(f"  Family info: {stats['with_family']} ({stats['with_family']/stats['processed']*100:.1f}%)")
    else:
        print(f"\n⚠️  No files were successfully processed")
    
    if not dry_run:
        print(f"\n✓  Saved {stats['processed']} metadata files to: {input_dir}")
    else:
        print(f"\n✓  Dry run complete (no files saved)")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich pension files with veteran metadata"
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing text files (default: data/documents/smithsonian/pension_files/text_files)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Number of files to process (default: 0 = all files)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be extracted without saving files'
    )
    
    args = parser.parse_args()
    
    # Set input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        # Go up from scripts/ to project root
        project_root = Path(__file__).parent.parent
        input_dir = project_root / "data" / "documents" / "smithsonian" / "pension_files" / "text_files"
    
    if not input_dir.exists():
        print(f"\n❌ Error: Directory not found: {input_dir}")
        return
    
    enrich_pension_files(input_dir, args.limit, args.dry_run)


if __name__ == '__main__':
    main()
