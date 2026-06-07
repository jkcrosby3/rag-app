"""
PDF processor for the RAG system.

This script extracts text from PDF files and prepares them
for ingestion into the RAG system.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
    sys.exit(1)


class PDFProcessor:
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""


def process_pdf(pdf_path: str, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Process a PDF file and save the extracted text.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the extracted text
        metadata: Optional metadata to include
        
    Returns:
        Path to the saved text file or None if processing failed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    # Extract text
    logger.info(f"Extracting text from {pdf_path}")
    text = extract_text_from_pdf(str(pdf_path))
    
    if not text.strip():
        logger.warning(f"No text extracted from {pdf_path}")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text
    output_path = output_dir / f"{pdf_path.stem}.txt"
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    # Save metadata
    if metadata:
        metadata_path = output_dir / f"{pdf_path.stem}.metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2)
    
    logger.info(f"Saved extracted text to {output_path}")
    return str(output_path)


def process_directory(input_dir: str, output_dir: str) -> List[str]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save extracted text
        
    Returns:
        List of paths to saved text files
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Process each PDF file
    output_paths = []
    for pdf_path in pdf_files:
        # Create basic metadata
        metadata = {
            "title": pdf_path.stem,
            "source": str(pdf_path),
            "topic": "great_depression"
        }
        
        output_path = process_pdf(pdf_path, output_dir, metadata)
        if output_path:
            output_paths.append(output_path)
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("--input", type=str, required=True, help="Path to PDF file or directory")
    parser.add_argument("--output-dir", type=str, default="data/documents/great_depression", help="Directory to save extracted text")
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return
    
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Process single PDF file
        metadata = {
            "title": input_path.stem,
            "source": str(input_path),
            "topic": "great_depression"
        }
        output_path = process_pdf(input_path, args.output_dir, metadata)
        if output_path:
            logger.info(f"Successfully processed PDF: {output_path}")
        else:
            logger.error(f"Failed to process PDF: {input_path}")
    elif input_path.is_dir():
        # Process all PDF files in directory
        output_paths = process_directory(input_path, args.output_dir)
        if output_paths:
            logger.info(f"Successfully processed {len(output_paths)} PDF files")
            for path in output_paths:
                logger.info(f"  - {path}")
        else:
            logger.error(f"Failed to process any PDF files in {input_path}")
    else:
        logger.error(f"Input must be a PDF file or directory: {input_path}")


if __name__ == "__main__":
    main()
