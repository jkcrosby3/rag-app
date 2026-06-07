#!/usr/bin/env python3
"""
Download Smithsonian Hackathon Data from Hugging Face

Purpose: Download a manageable subset of Revolutionary War data for RAG processing

Usage:
    python download_smithsonian_data.py --collection pension_pdfs --limit 100
    python download_smithsonian_data.py --collection newspapers --limit 500
    python download_smithsonian_data.py --collection smithsonian --limit 200

Author: Smithsonian Hackathon Team
Date: May 7, 2026
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json


COLLECTIONS = {
    'pension_pdfs': 'RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs',
    'pension_text': 'RevolutionCrossroads/nara_revolutionary_war_pension_files',
    'newspapers_issues': 'RevolutionCrossroads/loc_chronicling_america_1770-1810_issues',
    'newspapers_ocr': 'RevolutionCrossroads/loc_chronam_textract_ocr_bah',
    'newspapers_full': 'RevolutionCrossroads/loc_chronicling_america_1770-1810',
    'smithsonian': 'RevolutionCrossroads/si_us_revolutionary_era_collections',
    'smithsonian_images': 'RevolutionCrossroads/si_images_textlabeling_bah'
}


def download_collection(collection_name: str, limit: int, output_dir: Path):
    """
    Download a subset of a Hugging Face dataset.
    
    Args:
        collection_name: Name of collection from COLLECTIONS dict
        limit: Maximum number of records to download
        output_dir: Directory to save downloaded files
    """
    if collection_name not in COLLECTIONS:
        print(f"❌ Unknown collection: {collection_name}")
        print(f"   Available: {', '.join(COLLECTIONS.keys())}")
        return False
    
    dataset_path = COLLECTIONS[collection_name]
    print(f"\n📥 Downloading from: {dataset_path}")
    print(f"   Limit: {limit} records")
    print(f"   Output: {output_dir}")
    
    try:
        # Load dataset with limit
        print("\n⏳ Loading dataset...")
        dataset = load_dataset(
            dataset_path,
            split=f"train[:{limit}]",
            trust_remote_code=True
        )
        
        print(f"✓  Loaded {len(dataset)} records")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process records based on collection type
        print("\n💾 Saving files...")
        
        if 'pdf' in collection_name.lower():
            save_pdfs(dataset, output_dir)
        elif 'image' in collection_name.lower():
            save_images(dataset, output_dir)
        else:
            save_text(dataset, output_dir)
        
        print(f"\n✅ Download complete!")
        print(f"   Files saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return False


def save_pdfs(dataset, output_dir: Path):
    """Save PDF files from dataset."""
    for i, record in enumerate(tqdm(dataset, desc="Saving PDFs")):
        try:
            # Adjust field names based on actual dataset structure
            if 'pdf' in record:
                pdf_data = record['pdf']
                filename = record.get('filename', f"document_{i:04d}.pdf")
                
                output_path = output_dir / filename
                with open(output_path, 'wb') as f:
                    f.write(pdf_data)
            
            # Also save metadata
            metadata = {k: v for k, v in record.items() if k != 'pdf'}
            metadata_path = output_dir / f"metadata_{i:04d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"\n⚠️  Error saving record {i}: {e}")


def save_text(dataset, output_dir: Path):
    """Save text records from dataset."""
    for i, record in enumerate(tqdm(dataset, desc="Saving text files")):
        try:
            # Try common text field names
            text_fields = ['text', 'content', 'ocr_text', 'full_text', 'description']
            text_content = None
            
            for field in text_fields:
                if field in record and record[field]:
                    text_content = record[field]
                    break
            
            if text_content:
                # Save text file
                filename = record.get('id', record.get('filename', f"document_{i:04d}"))
                output_path = output_dir / f"{filename}.txt"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            
            # Save full metadata as JSON
            metadata_path = output_dir / f"metadata_{i:04d}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"\n⚠️  Error saving record {i}: {e}")


def save_images(dataset, output_dir: Path):
    """Save image files from dataset."""
    for i, record in enumerate(tqdm(dataset, desc="Saving images")):
        try:
            if 'image' in record:
                image = record['image']
                filename = record.get('filename', f"image_{i:04d}.jpg")
                
                output_path = output_dir / filename
                image.save(output_path)
            
            # Save metadata
            metadata = {k: v for k, v in record.items() if k != 'image'}
            metadata_path = output_dir / f"metadata_{i:04d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"\n⚠️  Error saving record {i}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Smithsonian Revolutionary War data from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Collections:
  pension_pdfs        - Revolutionary War Pension Files (PDFs)
  pension_text        - Revolutionary War Pension Files (Text)
  newspapers_issues   - Chronicling America Newspaper Issues
  newspapers_ocr      - Chronicling America with OCR
  newspapers_full     - Chronicling America Full Dataset
  smithsonian         - Smithsonian Revolutionary-era Collections
  smithsonian_images  - Smithsonian Images with Text Labels

Examples:
  # Download 100 pension PDFs
  python download_smithsonian_data.py --collection pension_pdfs --limit 100
  
  # Download 500 newspaper issues
  python download_smithsonian_data.py --collection newspapers_issues --limit 500
  
  # Download 200 Smithsonian records
  python download_smithsonian_data.py --collection smithsonian --limit 200
        """
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        choices=COLLECTIONS.keys(),
        help='Collection to download'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of records to download (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/documents/<collection_name>)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).parent / "data" / "documents"
        output_dir = base_dir / args.collection
    
    print("\n" + "=" * 60)
    print("📚 SMITHSONIAN HACKATHON DATA DOWNLOADER")
    print("=" * 60)
    
    success = download_collection(args.collection, args.limit, output_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review downloaded files in:", output_dir)
        print("  2. Run: .\\start_web_app.bat")
        print("  3. The RAG system will process your documents")
        print("\n" + "=" * 60)
    else:
        print("\n❌ Download failed. Check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
