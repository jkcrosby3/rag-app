"""
Script to process documents for the RAG system.

AGNOSTIC DOCUMENT PROCESSOR:
This script is dataset-agnostic and can process any collection of documents.
It recursively searches the input directory for files matching the specified
extensions (default: .txt files) and extracts text and metadata from each file.

WHAT "PROCESS ALL DOCUMENTS" MEANS:
- Recursively scans the input directory and all subdirectories
- Finds all files matching the specified extensions (default: .txt)
- Extracts text content and metadata from each file
- Saves processed results to the output directory
- Example: If input-dir is "data/documents", it will find and process:
  * data/documents/smithsonian/pension_files/text_files/*.txt
  * data/documents/smithsonian/rev_era_collections/text_files/*.txt
  * data/documents/any_other_folder/**/*.txt (all nested .txt files)

Usage Examples:
    # Process all .txt files in data/documents (default behavior)
    # Recursively processes ALL subdirectories
    python scripts/process_documents.py
    
    # Process only Smithsonian documents
    python scripts/process_documents.py --input-dir data/documents/smithsonian
    
    # Process specific subdirectory (e.g., only pension files)
    python scripts/process_documents.py --input-dir data/documents/smithsonian/pension_files
    
    # Custom output directory
    python scripts/process_documents.py --output-dir data/my_processed
    
    # Process multiple file types (.txt and .pdf)
    python scripts/process_documents.py --extensions .txt .pdf
    
    # Combine options
    python scripts/process_documents.py --input-dir data/my_docs --output-dir data/output --extensions .txt .md

Arguments:
    --input-dir     Input directory to scan (default: data/documents)
    --output-dir    Where to save processed documents (default: data/processed)
    --extensions    File types to process (default: .txt only)
    --recursive     Process subdirectories (default: True)

Note: The script automatically skips non-text files like _metadata.json
      unless you explicitly include .json in --extensions
"""
import logging
import sys
import os
import argparse
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing.batch_processor import BatchProcessor


def main():
    """Process documents and save results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process documents for the RAG system"
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/documents',
        help='Input directory containing documents to process (default: data/documents)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed documents (default: data/processed)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.txt'],
        help='File extensions to process (default: .txt)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Process subdirectories recursively (default: True)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting document processing")
    
    # Define directories
    base_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing documents from: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process documents
    try:
        processor = BatchProcessor(output_dir=output_dir)
        results = processor.process_directory(
            directory=base_dir,
            recursive=args.recursive,
            file_extensions=args.extensions
        )
        logger.info(f"Successfully processed {len(results)} documents")
        
        # Print summary of processed documents
        topics = {}
        for result in results:
            topic = result['metadata'].get('topic', 'unknown')
            if topic not in topics:
                topics[topic] = 0
            topics[topic] += 1
        
        logger.info("Documents processed by topic:")
        for topic, count in topics.items():
            logger.info(f"  - {topic}: {count} documents")
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
