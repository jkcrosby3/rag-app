#!/usr/bin/env python3
"""
Download Smithsonian Revolutionary Era Collections - Direct Download Method

Downloads Revolutionary Era museum collections from Smithsonian datasets on Hugging Face.
Two datasets available:
1. Museum collections (si_us_revolutionary_era_collections) - 12,667 object records
2. Image text labeling (si_images_textlabeling_bah) - Image dataset with text labels

This version downloads the parquet file directly using requests,
bypassing SSL verification issues on corporate networks.

Usage Examples:
    # Download 50 museum collection records (default)
    python scripts/download_rev_era_collections.py
    
    # Download 100 museum collection records
    python scripts/download_rev_era_collections.py --limit 100
    
    # Download 50 image text labeling records
    python scripts/download_rev_era_collections.py --dataset images --limit 50
    
    # Download ALL museum collection records (12,667 total)
    python scripts/download_rev_era_collections.py --dataset collections --limit 0
    
    # Download ALL image text labeling records
    python scripts/download_rev_era_collections.py --dataset images --limit 0
    
    # Custom output directory
    python scripts/download_rev_era_collections.py --output-dir "C:\custom\path"
    
    # Combine options
    python scripts/download_rev_era_collections.py --dataset collections --limit 200 --output-dir "data/collections"

Arguments:
    --dataset    Choose 'collections' (12,667 museum objects, default) or 'images' (image text labels)
    --limit      Number of records to download (default: 50, use 0 for all)
    --output-dir Custom output directory (default: data/documents/smithsonian/rev_era_collections)

Output:
    - rev_era_[type].parquet        # Efficient data format
    - rev_era_[type].json           # Human-readable format
    - text_files/*.txt              # Individual text files for RAG
    - text_files/*_metadata.json    # Metadata for each file

Next Steps:
    1. Review downloaded files in data/documents/smithsonian/rev_era_collections/text_files/
    2. Run: python scripts/process_documents.py
    3. Run: .\start_web_app.bat
    4. Query your Revolutionary Era collection data!
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


DATASETS = {
    'collections': {
        'name': 'Smithsonian Revolutionary Era Collections',
        'dataset_id': 'RevolutionCrossroads/si_us_revolutionary_era_collections',
        'url': 'https://huggingface.co/datasets/RevolutionCrossroads/si_us_revolutionary_era_collections/resolve/main/data/train-00000-of-00001.parquet',
        'description': '12,667 museum object records from NMAH, NPM, SAAM, and NPG (1770-1810)'
    },
    'images': {
        'name': 'Smithsonian Images with Text Labeling',
        'dataset_id': 'RevolutionCrossroads/si_images_textlabeling_bah',
        'url': 'https://huggingface.co/datasets/RevolutionCrossroads/si_images_textlabeling_bah/resolve/main/dataset.parquet',
        'description': 'Image dataset with text labeling for Revolutionary Era objects'
    }
}


def get_parquet_urls(dataset_id):
    """Get parquet file URLs from HuggingFace Datasets API."""
    api_url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    
    try:
        print(f"\n⏳ Fetching parquet file list from API...")
        response = requests.get(api_url, verify=False, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract parquet URLs from the response
        parquet_urls = []
        
        if isinstance(data, dict):
            for config_name, splits in data.items():
                if isinstance(splits, dict):
                    for split_name, urls in splits.items():
                        if isinstance(urls, list):
                            parquet_urls.extend(urls)
        
        if parquet_urls:
            print(f"✓  Found {len(parquet_urls)} parquet file(s)")
            return parquet_urls
        else:
            print(f"⚠️  No parquet files found, using fallback URL")
            return None
            
    except Exception as e:
        print(f"⚠️  API request failed: {e}, using fallback URL")
        return None


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


def process_parquet_file(parquet_file, limit, output_dir, dataset_type):
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
        
        # # Save subset as parquet (disabled - only saving text files for RAG)
        # subset_parquet = output_dir / f"rev_era_{dataset_type}.parquet"
        # df_subset.to_parquet(subset_parquet, index=False)
        # print(f"\n💾 Saved subset: {subset_parquet}")
        
        # # Save as JSON (disabled - only saving text files for RAG)
        # json_file = output_dir / f"rev_era_{dataset_type}.json"
        # df_subset.to_json(json_file, orient='records', indent=2)
        # print(f"💾 Saved JSON: {json_file}")
        
        # Save as text files
        text_dir = output_dir / "text_files"
        text_dir.mkdir(exist_ok=True)
        
        print(f"\n📝 Creating text files for RAG processing...")
        save_as_text_files(df_subset, text_dir, dataset_type)
        
        return df_subset
        
    except Exception as e:
        print(f"❌ Error processing parquet: {e}")
        return None


def save_as_text_files(df, text_dir, dataset_type):
    """Save records as individual text files."""
    # Different fields for different dataset types
    if dataset_type == 'collections':
        # Museum collection fields
        text_fields = ['title', 'creditLine', 'notes', 'physicalDescription', 'topic']
        id_fields = ['EDANid', 'guid', 'identifier']
        metadata_fields = ['unitCode', 'dataSource', 'date', 'name', 'objectType', 
                          'place', 'publisher', 'mediaCount', 'collectionsURL']
    else:  # images
        # Image dataset fields (adjust based on actual schema)
        text_fields = ['text', 'label', 'caption', 'description', 'content']
        id_fields = ['id', 'image_id', 'filename']
        metadata_fields = ['source', 'date', 'type']
    
    saved_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving text files"):
        try:
            # Build text content from multiple fields
            text_parts = []
            
            # Add title/main content first
            for field in text_fields:
                if field in row:
                    value = row[field]
                    if value is None:
                        continue
                    
                    # Handle list/array fields
                    if hasattr(value, '__len__') and hasattr(value, '__iter__') and not isinstance(value, str):
                        if len(value) > 0:
                            value = '\n'.join(str(item) for item in value if item is not None and str(item).strip())
                    else:
                        value = str(value)
                    
                    if value and value.strip():
                        text_parts.append(f"{field.upper()}: {value}")
            
            text_content = '\n\n'.join(text_parts)
            
            # Skip if no meaningful content
            if not text_content or len(text_content) < 50:
                continue
            
            # Generate filename
            record_id = None
            for field in id_fields:
                if field in row:
                    value = row[field]
                    # Handle arrays/lists
                    if hasattr(value, '__len__') and hasattr(value, '__iter__') and not isinstance(value, str):
                        if len(value) > 0 and value[0] is not None:
                            record_id = str(value[0])
                    # Handle scalars
                    elif value is not None:
                        record_id = str(value)
                    
                    if record_id:
                        # Clean up EDAN IDs
                        if 'edan' in record_id.lower():
                            record_id = record_id.split(':')[-1]
                        break
            
            if not record_id:
                record_id = f"rev_era_{idx:06d}"
            
            # Clean filename
            filename = "".join(c for c in record_id if c.isalnum() or c in ('_', '-'))
            filename = f"{filename}.txt"
            
            # Save text file
            text_path = text_dir / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"Smithsonian Revolutionary Era Collection\n")
                f.write(f"Record ID: {record_id}\n")
                
                # Add key metadata
                for field in metadata_fields:
                    if field in row:
                        value = row[field]
                        if value is not None:
                            if hasattr(value, '__len__') and hasattr(value, '__iter__') and not isinstance(value, str):
                                if len(value) > 0:
                                    value = ', '.join(str(item) for item in value[:3] if item is not None)
                                    f.write(f"{field}: {value}\n")
                            else:
                                f.write(f"{field}: {value}\n")
                
                f.write("=" * 70 + "\n\n")
                f.write(text_content)
            
            # Save metadata
            metadata_path = text_dir / f"{filename.replace('.txt', '_metadata.json')}"
            metadata = {}
            for k, v in row.to_dict().items():
                if v is None:
                    metadata[k] = None
                # Handle list/array types
                elif hasattr(v, '__len__') and hasattr(v, '__iter__') and not isinstance(v, str):
                    metadata[k] = [str(item) if item is not None else None for item in v]
                else:
                    metadata[k] = str(v)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            saved_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error saving record {idx}: {e}")
    
    print(f"✓  Saved {saved_count} text files to: {text_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Smithsonian Revolutionary Era Collections (Direct Method)"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['collections', 'images'],
        default='collections',
        help='Which dataset to download: collections (12,667 museum objects, default) or images (image text labels)'
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
        help='Output directory (default: smithsonian\rag-app\data\documents\smithsonian\american_revolutionary_era_collections)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Go up from scripts/ to project root
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "documents" / "smithsonian" / "american_revolutionary_era_collections"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_info = DATASETS[args.dataset]
    
    print("\n" + "=" * 70)
    print("🏛️  SMITHSONIAN REVOLUTIONARY ERA COLLECTIONS DOWNLOADER")
    print("=" * 70)
    print(f"\nDataset: {dataset_info['name']}")
    print(f"Description: {dataset_info['description']}")
    print("\n⚠️  SSL verification disabled for corporate network")
    
    # Try to get parquet URLs from API, fallback to direct URL
    parquet_urls = get_parquet_urls(dataset_info['dataset_id'])
    
    if parquet_urls:
        parquet_url = parquet_urls[0]
        print(f"\n📦 Using parquet file from API: {parquet_url}")
    else:
        parquet_url = dataset_info['url']
        print(f"\n📦 Using fallback URL: {parquet_url}")
    
    # Download to temp file (required to extract text content)
    # This temp file will be deleted after processing
    temp_parquet = output_dir / "temp_download.parquet"
    
    if download_parquet_file(parquet_url, temp_parquet):
        df = process_parquet_file(temp_parquet, args.limit, output_dir, args.dataset)
        
        if df is not None:
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            print(f"\n✓  Downloaded {len(df):,} Revolutionary Era collection records")
            print(f"✓  Files saved to: {output_dir}")
            print(f"✓  Text files in: {output_dir / 'text_files'}")
            print("\nNext steps:")
            print("  1. Review text files")
            print("  2. Run: python scripts/process_documents.py")
            print("  3. Run: .\\start_web_app.bat")
            print("  4. Query your Revolutionary Era collection data!")
            print("\n" + "=" * 70)
            
            # Clean up temp file
            if temp_parquet.exists():
                temp_parquet.unlink()
            
            return 0
    
    print("\n❌ Download failed")
    return 1


if __name__ == "__main__":
    exit(main())
