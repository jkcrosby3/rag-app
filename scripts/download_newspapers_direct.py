#!/usr/bin/env python3
"""
Download Chronicling America Newspapers - Direct Download Method

This version downloads the parquet file directly using requests,
bypassing SSL verification issues on corporate networks.

Usage:
    python download_newspapers_direct.py --limit 50
"""

import os
import ssl
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_parquet_file(url, output_file):
    """Download a parquet file with SSL verification disabled."""
    print(f"\n⏳ Downloading parquet file...")
    print(f"   From: {url}")
    print(f"   To: {output_file}")
    
    try:
        # Download with SSL verification disabled
        response = requests.get(url, stream=True, verify=False, timeout=300)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓  Download complete: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def process_parquet_file(parquet_file, limit, output_dir):
    """Load and process the downloaded parquet file."""
    print(f"\n⏳ Loading parquet file...")
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        print(f"✓  Loaded {len(df):,} total rows")
        print(f"✓  Columns: {list(df.columns)}")
        
        # Take subset or all records
        if limit is None or limit <= 0:
            df_subset = df
            print(f"✓  Keeping ALL {len(df_subset):,} rows")
        else:
            df_subset = df.head(limit)
            print(f"✓  Keeping first {len(df_subset):,} rows")
        
        # Save subset as parquet
        subset_parquet = output_dir / "newspapers_data.parquet"
        df_subset.to_parquet(subset_parquet, index=False)
        print(f"\n💾 Saved subset: {subset_parquet}")
        
        # Save as JSON
        json_file = output_dir / "newspapers_data.json"
        df_subset.to_json(json_file, orient='records', indent=2)
        print(f"💾 Saved JSON: {json_file}")
        
        # Save as text files
        text_dir = output_dir / "text_files"
        text_dir.mkdir(exist_ok=True)
        
        print(f"\n📝 Creating text files for RAG processing...")
        save_as_text_files(df_subset, text_dir)
        
        return df_subset
        
    except Exception as e:
        print(f"❌ Error processing parquet: {e}")
        return None


def save_as_text_files(df, text_dir):
    """Save records as individual text files."""
    text_fields = ['text', 'ocr_text', 'full_text', 'content', 'article_text']
    id_fields = ['id', 'issue_id', 'lccn', 'document_id']
    
    saved_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving text files"):
        try:
            # Find text content
            text_content = None
            for field in text_fields:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    text_content = str(row[field])
                    break
            
            if not text_content or len(text_content) < 50:
                continue
            
            # Generate filename
            record_id = None
            for field in id_fields:
                if field in row and pd.notna(row[field]):
                    record_id = str(row[field])
                    break
            
            if not record_id:
                record_id = f"record_{idx:06d}"
            
            # Clean filename
            filename = "".join(c for c in record_id if c.isalnum() or c in ('_', '-'))
            filename = f"{filename}.txt"
            
            # Save text file
            text_path = text_dir / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"Document ID: {record_id}\n")
                
                # Add metadata
                for field in ['date', 'title', 'newspaper', 'state', 'city']:
                    if field in row and pd.notna(row[field]):
                        f.write(f"{field.title()}: {row[field]}\n")
                
                f.write("=" * 70 + "\n\n")
                f.write(text_content)
            
            # Save metadata
            metadata_path = text_dir / f"{filename.replace('.txt', '_metadata.json')}"
            metadata = {k: str(v) if pd.notna(v) else None for k, v in row.to_dict().items()}
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            saved_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error saving record {idx}: {e}")
    
    print(f"✓  Saved {saved_count} text files to: {text_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Chronicling America newspapers (Direct Method)"
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Number of records to keep (default: 50, use 0 or -1 for ALL records)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/documents/smithsonian/newspapers)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "data" / "documents" / "newspapers"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📰 CHRONICLING AMERICA DOWNLOADER (Direct Method)")
    print("=" * 70)
    print("\n⚠️  SSL verification disabled for corporate network")
    
    # Direct URL to parquet file (you may need to update this)
    # This is a direct link to the parquet file on Hugging Face
    # parquet_url = "https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810/resolve/main/chronam_ocr_1770-1810.parquet"
    parquet_url = "https://huggingface.co/datasets/RevolutionCrossroads/loc_chronam_textract_ocr_bah/chronam_ocr_1770-1810.with_ocr_text_new.parquet"
    # Download to temp file
    temp_parquet = output_dir / "temp_download.parquet"
    
    if download_parquet_file(parquet_url, temp_parquet):
        df = process_parquet_file(temp_parquet, args.limit, output_dir)
        
        if df is not None:
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            print(f"\n✓  Downloaded {len(df):,} newspaper records")
            print(f"✓  Files saved to: {output_dir}")
            print(f"✓  Text files in: {output_dir / 'text_files'}")
            print("\nNext steps:")
            print("  1. Review text files")
            print("  2. Run: .\\start_web_app.bat")
            print("  3. Query your newspaper data!")
            print("\n" + "=" * 70)
            
            # Clean up temp file
            if temp_parquet.exists():
                temp_parquet.unlink()
            
            return 0
    
    print("\n❌ Download failed")
    return 1


if __name__ == "__main__":
    exit(main())
