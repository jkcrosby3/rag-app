#!/usr/bin/env python3
"""
Download Chronicling America Newspapers - Simple Parquet Method

Purpose: Download newspaper data using Hugging Face's simple parquet loading

Usage:
    python download_newspapers_simple.py --limit 100
    python download_newspapers_simple.py --limit 500 --output-dir custom/path
    python download_newspapers_simple.py --explore  # Just show first 5 rows

Author: Smithsonian Hackathon Team
Date: May 7, 2026
"""

import os
import ssl

# Disable SSL verification FIRST, before any other imports
# This is needed when behind corporate SSL inspection proxies (like Booz Allen)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HTTPX_VERIFY'] = '0'  # For httpx library
ssl._create_default_https_context = ssl._create_unverified_context

# Disable warnings about insecure requests
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Now import everything else
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from huggingface_hub import login
from dotenv import load_dotenv

# Import pandas and configure for SSL bypass
import pandas as pd
import fsspec

# Configure fsspec to skip SSL verification for HTTP filesystems
fsspec.config.conf['https'] = {'client_kwargs': {'verify': False}}


# Hugging Face dataset paths
DATASETS = {
    'newspapers': 'hf://datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810/chronam_ocr_1770-1810.parquet',
    'newspapers_issues': 'hf://datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810_issues',
    'pension_files': 'hf://datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files',
    'smithsonian': 'hf://datasets/RevolutionCrossroads/si_us_revolutionary_era_collections'
}


def authenticate_huggingface():
    """
    Authenticate with Hugging Face.
    
    Tries multiple methods in order:
    1. Existing token file (~/.huggingface/token)
    2. HF_TOKEN from .env file
    3. HF_TOKEN environment variable
    4. Interactive login prompt
    """
    try:
        # Check if already logged in (token file exists)
        token_path = Path.home() / ".huggingface" / "token"
        if token_path.exists():
            print("✓  Using existing Hugging Face authentication")
            return True
        
        # Load .env file from parent directory (smithsonian/.env)
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            print(f"✓  Loading .env file from: {env_path}")
            load_dotenv(env_path)
        
        # Try environment variable (from .env or system)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("✓  Logging in with HF_TOKEN environment variable...")
            login(token=hf_token)
            return True
        
        # Interactive login
        print("\n🔑 Hugging Face authentication required")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        print("\n   Choose authentication method:")
        print("   1. Enter token now (will be saved)")
        print("   2. Run 'huggingface-cli login' in terminal first")
        
        choice = input("\n   Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            login()  # Interactive prompt
            return True
        else:
            print("\n   Please run: huggingface-cli login")
            print("   Then run this script again.")
            return False
            
    except Exception as e:
        print(f"⚠️  Authentication warning: {e}")
        print("   Some datasets may not be accessible without authentication.")
        return False


def explore_dataset(dataset_key='newspapers'):
    """Load and display first few rows to explore structure."""
    print(f"\n🔍 Exploring {dataset_key} dataset...")
    print(f"   Loading from: {DATASETS[dataset_key]}")
    
    try:
        # Load just first 5 rows
        print("\n⏳ Loading sample data...")
        df = pd.read_parquet(DATASETS[dataset_key])
        
        print(f"\n✓  Dataset loaded: {len(df):,} total rows")
        print(f"✓  Columns: {list(df.columns)}")
        print(f"\n📊 Dataset Info:")
        print(df.info())
        
        print(f"\n📄 First 5 rows:")
        print(df.head())
        
        print(f"\n📝 Sample record (first row):")
        print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str))
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTip: Make sure you have 'huggingface_hub' installed:")
        print("     pip install huggingface_hub")
        return None


def download_dataset(dataset_key='newspapers', limit=100, output_dir=None):
    """
    Download dataset and save as text files for RAG processing.
    
    Args:
        dataset_key: Which dataset to download
        limit: Maximum number of records to download
        output_dir: Where to save files
    """
    print(f"\n📥 Downloading {dataset_key} dataset...")
    print(f"   Source: {DATASETS[dataset_key]}")
    print(f"   Limit: {limit:,} records")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "documents" / dataset_key
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Output: {output_dir}")
    
    try:
        # Load dataset
        print("\n⏳ Loading dataset from Hugging Face...")
        df = pd.read_parquet(DATASETS[dataset_key])
        
        print(f"✓  Loaded {len(df):,} total rows")
        print(f"✓  Columns: {list(df.columns)}")
        
        # Take subset
        df_subset = df.head(limit)
        print(f"✓  Keeping first {len(df_subset):,} rows")
        
        # Save full data as parquet
        parquet_path = output_dir / f"{dataset_key}_data.parquet"
        df_subset.to_parquet(parquet_path, index=False)
        print(f"\n💾 Saved Parquet: {parquet_path}")
        
        # Save as JSON for inspection
        json_path = output_dir / f"{dataset_key}_data.json"
        df_subset.to_json(json_path, orient='records', indent=2)
        print(f"💾 Saved JSON: {json_path}")
        
        # Save as individual text files for RAG
        text_dir = output_dir / "text_files"
        text_dir.mkdir(exist_ok=True)
        
        print(f"\n📝 Creating text files for RAG processing...")
        save_as_text_files(df_subset, text_dir, dataset_key)
        
        # Save summary
        summary = {
            'dataset': dataset_key,
            'source': DATASETS[dataset_key],
            'total_records': len(df),
            'downloaded_records': len(df_subset),
            'columns': list(df.columns),
            'output_directory': str(output_dir),
            'text_files_directory': str(text_dir)
        }
        
        summary_path = output_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Download complete!")
        print(f"   Records: {len(df_subset):,}")
        print(f"   Text files: {text_dir}")
        print(f"   Summary: {summary_path}")
        
        return df_subset
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Install required package: pip install huggingface_hub")
        print("  2. Check internet connection")
        print("  3. Verify dataset path is correct")
        return None


def save_as_text_files(df, text_dir, dataset_key):
    """
    Save records as individual text files for RAG processing.
    
    Args:
        df: DataFrame with records
        text_dir: Directory to save text files
        dataset_key: Type of dataset (for field detection)
    """
    # Common text field names across datasets
    text_fields = [
        'text', 'ocr_text', 'full_text', 'content', 'article_text',
        'description', 'abstract', 'body', 'transcription'
    ]
    
    # ID field names
    id_fields = ['id', 'issue_id', 'document_id', 'record_id', 'lccn']
    
    saved_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving text files"):
        try:
            # Find text content
            text_content = None
            for field in text_fields:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    text_content = str(row[field])
                    break
            
            # If no text field found, concatenate all string fields
            if not text_content:
                text_parts = []
                for col, val in row.items():
                    if pd.notna(val) and isinstance(val, str) and len(val) > 20:
                        text_parts.append(f"{col}: {val}")
                text_content = "\n\n".join(text_parts)
            
            if not text_content or len(text_content) < 50:
                continue  # Skip records with insufficient text
            
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
            
            # Create text file with metadata header
            text_path = text_dir / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                # Write metadata header
                f.write("=" * 70 + "\n")
                f.write(f"Document ID: {record_id}\n")
                
                # Add other metadata fields
                metadata_fields = ['date', 'title', 'newspaper', 'location', 'state', 'city']
                for field in metadata_fields:
                    if field in row and pd.notna(row[field]):
                        f.write(f"{field.title()}: {row[field]}\n")
                
                f.write("=" * 70 + "\n\n")
                
                # Write main content
                f.write(text_content)
            
            # Save metadata as separate JSON
            metadata_path = text_dir / f"{filename.replace('.txt', '_metadata.json')}"
            metadata = row.to_dict()
            # Convert non-serializable types
            metadata = {k: str(v) if pd.notna(v) else None for k, v in metadata.items()}
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            saved_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error saving record {idx}: {e}")
    
    print(f"✓  Saved {saved_count} text files")


def main():
    parser = argparse.ArgumentParser(
        description="Download Smithsonian Revolutionary War datasets (Simple Method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  newspapers          - Chronicling America 1770-1810 (with OCR)
  newspapers_issues   - Chronicling America Issues
  pension_files       - Revolutionary War Pension Files
  smithsonian         - Smithsonian Revolutionary-era Collections

Examples:
  # Explore dataset structure
  python download_newspapers_simple.py --explore
  
  # Download 100 newspapers
  python download_newspapers_simple.py --limit 100
  
  # Download 500 pension files
  python download_newspapers_simple.py --dataset pension_files --limit 500
  
  # Download to custom directory
  python download_newspapers_simple.py --limit 200 --output-dir my_data
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=DATASETS.keys(),
        default='newspapers',
        help='Dataset to download (default: newspapers)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of records to download (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/documents/<dataset_name>)'
    )
    
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Just explore dataset structure without downloading'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("📚 SMITHSONIAN HACKATHON DATA DOWNLOADER (Simple Method)")
    print("=" * 70)
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface():
        print("\n⚠️  Continuing without authentication...")
        print("   Public datasets will work, but some may require login.")
    
    if args.explore:
        explore_dataset(args.dataset)
        print("\n💡 Tip: Run without --explore to download data")
        return 0
    
    df = download_dataset(
        dataset_key=args.dataset,
        limit=args.limit,
        output_dir=args.output_dir
    )
    
    if df is not None:
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review text files in: data/documents/<dataset>/text_files/")
        print("  2. Run: .\\start_web_app.bat")
        print("  3. Start querying your Revolutionary War data!")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n❌ Download failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
