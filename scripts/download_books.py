"""
Download and Manage Gutenberg Books

Downloads books from the catalog based on boolean flags and presets.

Usage:
    python scripts/download_books.py --preset minimal
    python scripts/download_books.py --preset standard
    python scripts/download_books.py --preset comprehensive
    python scripts/download_books.py --download-all
    python scripts/download_books.py --status
"""

import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import urllib3

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_catalog(catalog_path: Path) -> Dict:
    """Load book catalog from JSON."""
    with open(catalog_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_catalog(catalog: Dict, catalog_path: Path):
    """Save updated catalog to JSON."""
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def download_book(book: Dict, output_dir: Path) -> bool:
    """Download a single book."""
    url = book['url']
    filename = book['filename']
    output_path = output_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        print(f"  ✓ Already exists: {filename}")
        return True
    
    try:
        print(f"  📥 Downloading: {book['title']}")
        print(f"     Author: {book['author']}")
        print(f"     URL: {url}")
        
        # Disable SSL verification for corporate networks
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Update book metadata with download info
        file_size = output_path.stat().st_size
        download_date = datetime.now().isoformat()
        
        book['file_size_bytes'] = file_size
        book['downloaded_date'] = download_date
        
        print(f"  ✓ Downloaded: {filename} ({file_size:,} bytes)")
        print(f"     Downloaded at: {download_date}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error downloading {filename}: {e}")
        return False


def download_by_preset(catalog: Dict, preset_name: str, output_dir: Path):
    """Download books based on preset."""
    
    if preset_name not in catalog['download_presets']:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(catalog['download_presets'].keys())}")
        return
    
    preset = catalog['download_presets'][preset_name]
    book_ids = preset['book_ids']
    
    print(f"\n{'=' * 70}")
    print(f"📚 DOWNLOADING PRESET: {preset_name.upper()}")
    print(f"{'=' * 70}\n")
    print(f"Description: {preset['description']}")
    print(f"Total books: {preset['total_books']}")
    print(f"Estimated size: {preset['estimated_size_mb']} MB\n")
    
    # Filter books by IDs
    books_to_download = [b for b in catalog['books'] if b['id'] in book_ids]
    
    downloaded = 0
    failed = 0
    
    for book in books_to_download:
        if download_book(book, output_dir):
            book['downloaded'] = True
            downloaded += 1
        else:
            failed += 1
        print()
    
    print(f"{'=' * 70}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successfully downloaded: {downloaded}/{len(books_to_download)}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print()


def download_all_flagged(catalog: Dict, output_dir: Path):
    """Download all books where download=true."""
    
    books_to_download = [b for b in catalog['books'] if b.get('download', False)]
    
    print(f"\n{'=' * 70}")
    print(f"📚 DOWNLOADING ALL FLAGGED BOOKS")
    print(f"{'=' * 70}\n")
    print(f"Total books flagged for download: {len(books_to_download)}\n")
    
    downloaded = 0
    failed = 0
    
    for book in books_to_download:
        if download_book(book, output_dir):
            book['downloaded'] = True
            downloaded += 1
        else:
            failed += 1
        print()
    
    print(f"{'=' * 70}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successfully downloaded: {downloaded}/{len(books_to_download)}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print()


def show_status(catalog: Dict, output_dir: Path):
    """Show download status of all books."""
    
    print(f"\n{'=' * 70}")
    print(f"📊 BOOK CATALOG STATUS")
    print(f"{'=' * 70}\n")
    
    # Count by tier
    tier_counts = {}
    for book in catalog['books']:
        tier = book['tier']
        if tier not in tier_counts:
            tier_counts[tier] = {'total': 0, 'flagged': 0, 'downloaded': 0}
        tier_counts[tier]['total'] += 1
        if book.get('download', False):
            tier_counts[tier]['flagged'] += 1
        if book.get('downloaded', False):
            tier_counts[tier]['downloaded'] += 1
    
    print("By Tier:")
    for tier in sorted(tier_counts.keys()):
        counts = tier_counts[tier]
        print(f"  Tier {tier}: {counts['total']} books, "
              f"{counts['flagged']} flagged, {counts['downloaded']} downloaded")
    
    print(f"\nFlagged for Download:")
    flagged = [b for b in catalog['books'] if b.get('download', False)]
    for book in flagged:
        status = "✓" if book.get('downloaded', False) else "⏳"
        print(f"  {status} [{book['tier']}] {book['title']} by {book['author']}")
    
    print(f"\nTotal books in catalog: {len(catalog['books'])}")
    print(f"Flagged for download: {len(flagged)}")
    print(f"Downloaded: {sum(1 for b in catalog['books'] if b.get('downloaded', False))}")
    print(f"Output directory: {output_dir}")
    print()


def toggle_download_flag(catalog: Dict, book_id: int, value: bool):
    """Toggle download flag for a specific book."""
    for book in catalog['books']:
        if book['id'] == book_id:
            book['download'] = value
            print(f"✓ Set download={value} for: {book['title']}")
            return True
    print(f"❌ Book ID {book_id} not found")
    return False


def main():
    parser = argparse.ArgumentParser(description='Download Gutenberg books from catalog')
    parser.add_argument('--preset', choices=['minimal', 'standard', 'comprehensive'],
                       help='Download books by preset')
    parser.add_argument('--download-all', action='store_true',
                       help='Download all books flagged with download=true')
    parser.add_argument('--status', action='store_true',
                       help='Show download status')
    parser.add_argument('--flag', type=int, metavar='BOOK_ID',
                       help='Set download=true for book ID')
    parser.add_argument('--unflag', type=int, metavar='BOOK_ID',
                       help='Set download=false for book ID')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    catalog_path = base_dir / "data" / "documents" / "books" / "book_catalog.json"
    output_dir = base_dir / "data" / "documents" / "books"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    if not catalog_path.exists():
        print(f"❌ Catalog not found: {catalog_path}")
        return
    
    catalog = load_catalog(catalog_path)
    
    # Execute command
    if args.status:
        show_status(catalog, output_dir)
    
    elif args.preset:
        download_by_preset(catalog, args.preset, output_dir)
        save_catalog(catalog, catalog_path)
    
    elif args.download_all:
        download_all_flagged(catalog, output_dir)
        save_catalog(catalog, catalog_path)
    
    elif args.flag:
        if toggle_download_flag(catalog, args.flag, True):
            save_catalog(catalog, catalog_path)
    
    elif args.unflag:
        if toggle_download_flag(catalog, args.unflag, False):
            save_catalog(catalog, catalog_path)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
