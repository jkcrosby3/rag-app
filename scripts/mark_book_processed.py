"""
Mark Book as Processed

Updates book catalog to mark books as processed after adding to vector database.

Usage:
    python scripts/mark_book_processed.py --id 1
    python scripts/mark_book_processed.py --filename american_prisoners_revolution.txt
    python scripts/mark_book_processed.py --all
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict


def load_catalog(catalog_path: Path) -> Dict:
    """Load book catalog from JSON."""
    with open(catalog_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_catalog(catalog: Dict, catalog_path: Path):
    """Save updated catalog to JSON."""
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def mark_processed(catalog: Dict, book_id: int = None, filename: str = None) -> bool:
    """Mark a book as processed."""
    
    for book in catalog['books']:
        if (book_id and book['id'] == book_id) or (filename and book['filename'] == filename):
            if not book.get('downloaded', False):
                print(f"⚠️  Warning: Book not downloaded yet: {book['title']}")
                print(f"   Download first with: python scripts/download_books.py --flag {book['id']}")
                return False
            
            book['processed'] = True
            book['processed_date'] = datetime.now().isoformat()
            
            print(f"✓ Marked as processed: {book['title']}")
            print(f"  Author: {book['author']}")
            print(f"  Downloaded: {book.get('downloaded_date', 'Unknown')}")
            print(f"  Processed: {book['processed_date']}")
            return True
    
    if book_id:
        print(f"❌ Book ID {book_id} not found")
    elif filename:
        print(f"❌ Book filename '{filename}' not found")
    return False


def mark_all_downloaded_as_processed(catalog: Dict):
    """Mark all downloaded books as processed."""
    
    downloaded_books = [b for b in catalog['books'] if b.get('downloaded', False) and not b.get('processed', False)]
    
    if not downloaded_books:
        print("ℹ️  No downloaded books to process")
        return
    
    print(f"\n{'=' * 70}")
    print(f"📊 MARKING {len(downloaded_books)} BOOKS AS PROCESSED")
    print(f"{'=' * 70}\n")
    
    processed_date = datetime.now().isoformat()
    
    for book in downloaded_books:
        book['processed'] = True
        book['processed_date'] = processed_date
        print(f"✓ {book['title']}")
    
    print(f"\n{'=' * 70}")
    print(f"✓ Marked {len(downloaded_books)} books as processed")
    print(f"  Timestamp: {processed_date}")
    print(f"{'=' * 70}\n")


def show_processing_status(catalog: Dict):
    """Show processing status of all books."""
    
    print(f"\n{'=' * 70}")
    print(f"📊 PROCESSING STATUS")
    print(f"{'=' * 70}\n")
    
    downloaded = [b for b in catalog['books'] if b.get('downloaded', False)]
    processed = [b for b in catalog['books'] if b.get('processed', False)]
    pending = [b for b in downloaded if not b.get('processed', False)]
    
    print(f"Total books: {len(catalog['books'])}")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Processed: {len(processed)}")
    print(f"Pending processing: {len(pending)}")
    
    if pending:
        print(f"\n⏳ Pending Processing:")
        for book in pending:
            print(f"  - [{book['id']}] {book['title']}")
            print(f"    Downloaded: {book.get('downloaded_date', 'Unknown')}")
    
    if processed:
        print(f"\n✓ Recently Processed:")
        # Show last 5 processed
        recent = sorted(processed, 
                       key=lambda x: x.get('processed_date', ''), 
                       reverse=True)[:5]
        for book in recent:
            print(f"  - [{book['id']}] {book['title']}")
            print(f"    Processed: {book.get('processed_date', 'Unknown')}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Mark books as processed in catalog')
    parser.add_argument('--id', type=int, help='Book ID to mark as processed')
    parser.add_argument('--filename', help='Book filename to mark as processed')
    parser.add_argument('--all', action='store_true', 
                       help='Mark all downloaded books as processed')
    parser.add_argument('--status', action='store_true',
                       help='Show processing status')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    catalog_path = base_dir / "data" / "documents" / "books" / "book_catalog.json"
    
    # Load catalog
    if not catalog_path.exists():
        print(f"❌ Catalog not found: {catalog_path}")
        return
    
    catalog = load_catalog(catalog_path)
    
    # Execute command
    if args.status:
        show_processing_status(catalog)
    
    elif args.all:
        mark_all_downloaded_as_processed(catalog)
        save_catalog(catalog, catalog_path)
        print("✓ Catalog updated")
    
    elif args.id or args.filename:
        if mark_processed(catalog, book_id=args.id, filename=args.filename):
            save_catalog(catalog, catalog_path)
            print("✓ Catalog updated")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
