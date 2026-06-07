"""
Update Book Catalog Schema

Adds timestamp and file size fields to all books in catalog.

Usage:
    python scripts/update_book_catalog_schema.py
"""

import json
from pathlib import Path


def update_catalog_schema():
    """Add timestamp fields to all books."""
    
    base_dir = Path(__file__).parent.parent
    catalog_path = base_dir / "data" / "documents" / "books" / "book_catalog.json"
    
    # Load catalog
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    print(f"\n{'=' * 70}")
    print("🔄 UPDATING BOOK CATALOG SCHEMA")
    print(f"{'=' * 70}\n")
    print(f"Total books: {len(catalog['books'])}\n")
    
    updated = 0
    
    for book in catalog['books']:
        # Add new fields if they don't exist
        if 'downloaded_date' not in book:
            book['downloaded_date'] = None
            updated += 1
        
        if 'processed_date' not in book:
            book['processed_date'] = None
            updated += 1
        
        if 'file_size_bytes' not in book:
            book['file_size_bytes'] = None
            updated += 1
    
    # Save updated catalog
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated {updated} fields across {len(catalog['books'])} books")
    print(f"✓ Catalog saved to: {catalog_path}\n")
    
    # Show new schema
    print("New fields added:")
    print("  - downloaded_date: ISO 8601 timestamp when book was downloaded")
    print("  - processed_date: ISO 8601 timestamp when book was added to vector DB")
    print("  - file_size_bytes: Actual file size in bytes")
    print()


if __name__ == "__main__":
    update_catalog_schema()
