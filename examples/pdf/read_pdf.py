#!/usr/bin/env python3
"""Example script demonstrating PDF reading capabilities."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.document_processing.pdf_reader import PDFReader


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the PDF reader example."""
    parser = argparse.ArgumentParser(description="Demonstrate PDF reading capabilities")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to PDF file (default: data/books/dracula.pdf)"
    )
    args = parser.parse_args()

    # Initialize reader
    reader = PDFReader(min_file_size=100)  # 100 bytes minimum

    # Get path to PDF file
    if args.file:
        pdf_path = Path(args.file)
    else:
        pdf_path = Path(project_root) / "data" / "books" / "dracula.pdf"

    try:
        # Validate the PDF
        logger.info("Validating PDF file: %s", pdf_path)
        reader.validate_file(pdf_path)
        logger.info("✅ PDF is valid")

        # Extract text and metadata
        logger.info("\nExtracting content...")
        result = reader.extract_text(pdf_path)

        # Show metadata
        logger.info("\nMetadata:")
        for key, value in result["metadata"].items():
            logger.info("  %s: %s", key, value)

        # Show text preview
        text = result["text"]
        preview_length = 500
        logger.info("\nText preview (first %d chars):", preview_length)
        logger.info("-" * 40)
        logger.info(text[:preview_length] + "...")
        logger.info("-" * 40)
        logger.info("Total text length: %d characters", len(text))

    except Exception as e:
        logger.error("Error processing PDF: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()