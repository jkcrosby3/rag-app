"""PDF reader implementation for the RAG system.

This module provides functionality to read and validate PDF files,
implementing the PDF reader component of the RAG system.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from PyPDF2 import PdfReader
import fitz

logger = logging.getLogger(__name__)


class PDFError(Exception):
    """Base exception for PDF reader errors."""
    pass


class PDFNotFoundError(PDFError):
    """Raised when the input PDF file does not exist."""
    pass


class PDFCorruptError(PDFError):
    """Raised when the PDF file is corrupt or invalid."""
    pass


class PDFEncryptedError(PDFError):
    """Raised when the PDF file is password-protected."""
    pass


class PDFReader:
    """Handles reading and basic validation of PDF files."""

    def __init__(self, min_file_size: int = 1):
        """Initialize the PDFReader.

        Args:
            min_file_size: Minimum acceptable file size in bytes
        """
        self.min_file_size = min_file_size

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate that the file exists and is a valid PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            bool: True if file is valid

        Raises:
            PDFNotFoundError: If file does not exist
            PDFCorruptError: If file is not a valid PDF
            PDFEncryptedError: If file is password-protected
        """
        path = Path(file_path)
        if not path.exists():
            raise PDFNotFoundError(f"File not found: {file_path}")

        if path.stat().st_size < self.min_file_size:
            raise PDFCorruptError(f"File is empty or too small: {file_path}")

        try:
            with PdfReader(file_path) as pdf_reader:
                if doc.needs_pass:
                    raise PDFEncryptedError(
                        f"PDF is password-protected: {file_path}"
                    )
                # Try to access first page to verify PDF is valid
                _ = doc[0]
        except Exception as e:
            raise PDFCorruptError(
                f"Invalid or corrupt PDF: {file_path}"
            ) from e

        return True

    def extract_text(
        self,
        file_path: Union[str, Path],
        password: Optional[str] = None
    ) -> Dict[str, Union[str, Dict]]:
        """Extract text content and metadata from a PDF file.

        Args:
            file_path: Path to the PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Dict containing:
                - text: The extracted text content
                - metadata: Dict with file info (size, path, title, author, etc.)

        Raises:
            PDFNotFoundError: If file does not exist
            PDFCorruptError: If file is not a valid PDF
            PDFEncryptedError: If file is password-protected and no
                password provided
        """
        path = Path(file_path)
        self.validate_file(path)

        try:
            with PdfReader(file_path) as pdf_reader:
                if doc.needs_pass:
                    if not password:
                        raise PDFEncryptedError(
                            f"Password required for: {file_path}"
                        )
                    if not doc.authenticate(password):
                        raise PDFEncryptedError(
                            f"Invalid password for: {file_path}"
                        )

                # Extract text from all pages
                text = ""
                for page in doc:
                    text += page.get_text()

                # Get metadata
                metadata = {
                    "file_path": str(path.absolute()),
                    "file_size": path.stat().st_size,
                    "file_name": path.name,
                    "page_count": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                }

                return {"text": text, "metadata": metadata}

        except Exception as e:
            raise PDFCorruptError(f"Error reading PDF: {file_path}") from e
