#!/usr/bin/env python3
"""
Script to convert PDF documents to Markdown format.
Particularly useful for converting historical policy documents.
"""
import argparse
import logging
import re
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


class PDFToMarkdownConverter:
    """Converts PDF documents to Markdown format."""

    def __init__(self):
        """Initialize the converter with a PDFReader instance."""
        self.pdf_reader = PDFReader(min_file_size=100)  # 100 bytes minimum
    
    def _clean_text(self, text):
        """Clean and normalize text from PDF."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove any BOM characters that might cause encoding issues
        text = text.replace('\ufeff', '')
        
        # Replace non-breaking spaces with regular spaces
        text = text.replace('\xa0', ' ')
        
        return text
    
    def _detect_headings(self, text):
        """
        Attempt to detect headings in the text and convert them to Markdown headings.
        This is a heuristic approach and may need adjustment for different PDFs.
        """
        # Look for potential section titles (all caps lines or numbered sections)
        lines = text.split('\n')
        markdown_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
                
            # Check if line is all caps and not too long (likely a heading)
            if line.isupper() and len(line) < 100:
                markdown_lines.append(f'## {line.title()}')
            
            # Check for numbered sections (e.g., "1. Introduction")
            elif re.match(r'^\d+\.\s+\w+', line) and len(line) < 100:
                markdown_lines.append(f'## {line}')
            
            # Check for Roman numeral sections
            elif re.match(r'^[IVX]+\.\s+\w+', line) and len(line) < 100:
                markdown_lines.append(f'## {line}')
            
            # Otherwise, keep the line as is
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)
    
    def _detect_lists(self, text):
        """
        Detect and format lists in the text.
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for i, line in enumerate(lines):
            # Check for bullet points or numbered lists
            if re.match(r'^\s*[•●◦○*-]\s+', line):
                # Convert to Markdown bullet points
                indented_line = re.sub(r'^\s*[•●◦○*-]\s+', '- ', line)
                markdown_lines.append(indented_line)
            elif re.match(r'^\s*\d+\.\s+', line):
                # Keep numbered lists as they are (already Markdown compatible)
                markdown_lines.append(line)
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)
    
    def _format_paragraphs(self, text):
        """
        Format paragraphs properly for Markdown.
        """
        # Ensure paragraphs are separated by blank lines
        paragraphs = re.split(r'\n\s*\n', text)
        formatted_paragraphs = []
        
        for p in paragraphs:
            if p.strip():
                # Join lines within paragraphs
                p = re.sub(r'\n(?!\s*\n)', ' ', p)
                formatted_paragraphs.append(p.strip())
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _add_metadata(self, text, title, source=None, date=None):
        """
        Add YAML frontmatter with metadata.
        """
        frontmatter = ["---"]
        frontmatter.append(f"title: {title}")
        
        if source:
            frontmatter.append(f"source: {source}")
        
        if date:
            frontmatter.append(f"date: {date}")
        
        frontmatter.append("---\n")
        
        return '\n'.join(frontmatter) + text
    
    def convert_pdf_to_markdown(self, pdf_path, output_path=None, title=None, source=None, date=None):
        """
        Convert a PDF file to Markdown format.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the Markdown file (if None, derived from PDF path)
            title: Document title for metadata
            source: Document source for metadata
            date: Document date for metadata
            
        Returns:
            Path to the created Markdown file
        """
        pdf_path = Path(pdf_path)
        
        if not title:
            title = pdf_path.stem.replace('_', ' ').title()
        
        if not output_path:
            output_path = pdf_path.with_suffix('.md')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting {pdf_path} to Markdown")
        
        try:
            # Validate and extract text from PDF
            self.pdf_reader.validate_file(pdf_path)
            result = self.pdf_reader.extract_text(pdf_path)
            text = result["text"]
            
            # Process the text
            text = self._clean_text(text)
            text = self._detect_headings(text)
            text = self._detect_lists(text)
            text = self._format_paragraphs(text)
            
            # Add metadata if provided
            text = self._add_metadata(text, title, source, date)
            
            # Write to output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Successfully converted to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting PDF to Markdown: {e}")
            raise


def main():
    """Run the PDF to Markdown converter."""
    parser = argparse.ArgumentParser(description="Convert PDF documents to Markdown format")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to PDF file to convert"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the Markdown file (default: same as PDF with .md extension)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Document title for metadata"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Document source for metadata"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Document date for metadata"
    )
    args = parser.parse_args()

    converter = PDFToMarkdownConverter()
    
    try:
        output_path = converter.convert_pdf_to_markdown(
            args.file,
            args.output,
            args.title,
            args.source,
            args.date
        )
        logger.info(f"Conversion complete. Markdown file saved to: {output_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
