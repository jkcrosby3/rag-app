#!/usr/bin/env python3
"""
Download Chronicling America Newspapers via Hugging Face API

Purpose: Efficiently download newspaper data using HF Dataset Server API

Usage:
    python download_newspapers_api.py --method rows --limit 100
    python download_newspapers_api.py --method parquet --limit 500
    python download_newspapers_api.py --explore  # Just explore structure

Author: Smithsonian Hackathon Team
Date: May 7, 2026
"""

import argparse
import requests
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any


DATASET_NAME = "RevolutionCrossroads/loc_chronicling_america_1770-1810_issues"
BASE_URL = "https://datasets-server.huggingface.co"


def explore_dataset():
    """Explore dataset structure and metadata."""
    print("\n🔍 Exploring dataset structure...")
    
    # Get splits info
    splits_url = f"{BASE_URL}/splits"
    params = {"dataset": DATASET_NAME}
    
    response = requests.get(splits_url, params=params)
    if response.status_code == 200:
        splits_data = response.json()
        print("\n📊 Dataset Splits:")
        print(json.dumps(splits_data, indent=2))
    
    # Get first few rows to see structure
    print("\n📄 Sample Data (first 3 rows):")
    sample = fetch_rows_api(offset=0, length=3)
    
    if sample and 'rows' in sample:
        for i, row in enumerate(sample['rows']):
            print(f"\n--- Row {i+1} ---")
            print(json.dumps(row['row'], indent=2))
    
    return True


def fetch_rows_api(offset: int = 0, length: int = 100) -> Dict[str, Any]:
    """
    Fetch rows using the /rows API endpoint.
    
    Args:
        offset: Starting row index
        length: Number of rows to fetch
        
    Returns:
        JSON response with rows
    """
    url = f"{BASE_URL}/rows"
    params = {
        "dataset": DATASET_NAME,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching rows: {e}")
        return {}


def download_via_rows_api(limit: int, output_dir: Path):
    """
    Download data using paginated /rows API.
    Good for small to medium datasets.
    
    Args:
        limit: Total number of rows to download
        output_dir: Directory to save files
    """
    print(f"\n📥 Downloading {limit} rows via /rows API...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = 100  # API limit per request
    total_downloaded = 0
    all_rows = []
    
    with tqdm(total=limit, desc="Downloading rows") as pbar:
        while total_downloaded < limit:
            remaining = limit - total_downloaded
            fetch_size = min(batch_size, remaining)
            
            data = fetch_rows_api(offset=total_downloaded, length=fetch_size)
            
            if not data or 'rows' not in data:
                print(f"\n⚠️  No more data available at offset {total_downloaded}")
                break
            
            rows = data['rows']
            if not rows:
                break
            
            all_rows.extend([row['row'] for row in rows])
            total_downloaded += len(rows)
            pbar.update(len(rows))
    
    # Save as JSON
    json_path = output_dir / "newspapers_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓  Downloaded {total_downloaded} rows")
    print(f"✓  Saved to: {json_path}")
    
    # Also save individual text files for RAG processing
    save_as_text_files(all_rows, output_dir)
    
    return total_downloaded


def download_via_parquet(limit: int, output_dir: Path):
    """
    Download data using Parquet API.
    More efficient for larger datasets.
    
    Args:
        limit: Maximum number of rows to keep
        output_dir: Directory to save files
    """
    print(f"\n📥 Downloading via Parquet API (limit: {limit})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parquet file URLs
    parquet_url = f"https://huggingface.co/api/datasets/{DATASET_NAME}/parquet/default/train"
    
    try:
        response = requests.get(parquet_url, timeout=30)
        response.raise_for_status()
        parquet_files = response.json()
        
        if not parquet_files:
            print("❌ No Parquet files found")
            return 0
        
        print(f"✓  Found {len(parquet_files)} Parquet file(s)")
        
        # Download first parquet file
        print(f"\n⏳ Downloading Parquet file...")
        df = pd.read_parquet(parquet_files[0])
        
        print(f"✓  Loaded {len(df)} rows from Parquet")
        
        # Take subset
        df_subset = df.head(limit)
        print(f"✓  Keeping first {len(df_subset)} rows")
        
        # Save as Parquet
        parquet_path = output_dir / "newspapers_data.parquet"
        df_subset.to_parquet(parquet_path, index=False)
        print(f"✓  Saved to: {parquet_path}")
        
        # Save as JSON for inspection
        json_path = output_dir / "newspapers_data.json"
        df_subset.to_json(json_path, orient='records', indent=2)
        print(f"✓  Saved JSON to: {json_path}")
        
        # Save as text files for RAG
        rows = df_subset.to_dict('records')
        save_as_text_files(rows, output_dir)
        
        return len(df_subset)
        
    except Exception as e:
        print(f"❌ Error downloading Parquet: {e}")
        return 0


def save_as_text_files(rows: List[Dict], output_dir: Path):
    """
    Save newspaper data as individual text files for RAG processing.
    
    Args:
        rows: List of newspaper records
        output_dir: Directory to save text files
    """
    text_dir = output_dir / "text_files"
    text_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving {len(rows)} text files for RAG processing...")
    
    for i, row in enumerate(tqdm(rows, desc="Saving text files")):
        try:
            # Extract text content (adjust field names based on actual data)
            text_fields = ['text', 'ocr_text', 'content', 'full_text', 'article_text']
            text_content = None
            
            for field in text_fields:
                if field in row and row[field]:
                    text_content = row[field]
                    break
            
            if not text_content:
                # Fallback: concatenate all string fields
                text_content = "\n\n".join([
                    f"{k}: {v}" for k, v in row.items() 
                    if isinstance(v, str) and len(str(v)) > 10
                ])
            
            if text_content:
                # Create filename from metadata
                issue_id = row.get('id', row.get('issue_id', f'issue_{i:05d}'))
                filename = f"{issue_id}.txt"
                
                # Clean filename
                filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
                
                # Save text file
                text_path = text_dir / filename
                with open(text_path, 'w', encoding='utf-8') as f:
                    # Add metadata header
                    f.write(f"=== Newspaper Issue ===\n")
                    f.write(f"ID: {issue_id}\n")
                    if 'date' in row:
                        f.write(f"Date: {row['date']}\n")
                    if 'title' in row:
                        f.write(f"Title: {row['title']}\n")
                    f.write(f"\n{'='*50}\n\n")
                    f.write(text_content)
                
                # Save metadata separately
                metadata_path = text_dir / f"{issue_id}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(row, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"\n⚠️  Error saving row {i}: {e}")
    
    print(f"✓  Saved text files to: {text_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Chronicling America newspapers via HF API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore dataset structure
  python download_newspapers_api.py --explore
  
  # Download 100 rows via /rows API (good for testing)
  python download_newspapers_api.py --method rows --limit 100
  
  # Download 500 rows via Parquet API (more efficient)
  python download_newspapers_api.py --method parquet --limit 500
  
  # Download to custom directory
  python download_newspapers_api.py --method parquet --limit 200 --output-dir custom/path
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['rows', 'parquet'],
        default='parquet',
        help='Download method (default: parquet)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of rows to download (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/documents/newspapers)'
    )
    
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Just explore dataset structure without downloading'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("📰 CHRONICLING AMERICA NEWSPAPER DOWNLOADER")
    print("=" * 60)
    
    if args.explore:
        explore_dataset()
        return 0
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "data" / "documents" / "newspapers"
    
    # Download based on method
    if args.method == 'rows':
        count = download_via_rows_api(args.limit, output_dir)
    else:
        count = download_via_parquet(args.limit, output_dir)
    
    if count > 0:
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"\n✓  Downloaded {count} newspaper issues")
        print(f"✓  Files saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Review text files in:", output_dir / "text_files")
        print("  2. Run: .\\start_web_app.bat")
        print("  3. Query your newspaper data!")
        print("\n" + "=" * 60)
    else:
        print("\n❌ Download failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
