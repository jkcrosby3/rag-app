"""Unit tests for the PDF reader module."""
from pathlib import Path

import pytest

from src.document_processing.pdf_reader import (
    PDFCorruptError,
    PDFNotFoundError,
    PDFReader,
)


@pytest.fixture
def pdf_reader():
    """Create a PDFReader instance for testing."""
    return PDFReader()


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF file for testing.

    Note: This is a placeholder. In a real test, we would need
    actual PDF content. For now, we'll simulate by checking if
    the file exists and has minimum size.
    """
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    return file_path


@pytest.fixture
def corrupt_pdf(tmp_path):
    """Create a corrupt PDF file for testing."""
    file_path = tmp_path / "corrupt.pdf"
    file_path.write_text("This is not a PDF file")
    return file_path


@pytest.fixture
def encrypted_pdf(tmp_path):
    """Create an encrypted PDF for testing.

    Note: This is a placeholder. In a real test, we would need a real
    encrypted PDF file. For now, we'll simulate by checking if the
    file path ends with 'encrypted.pdf'.
    """
    file_path = tmp_path / "encrypted.pdf"
    file_path.touch()
    return file_path


def test_validate_valid_pdf(pdf_reader, sample_pdf):
    """Should return True for a valid PDF."""
    assert pdf_reader.validate_file(sample_pdf) is True


def test_validate_nonexistent_pdf(pdf_reader, tmp_path):
    """Should raise PDFNotFoundError for non-existent file."""
    with pytest.raises(PDFNotFoundError):
        pdf_reader.validate_file(tmp_path / "nonexistent.pdf")


def test_validate_corrupt_pdf(pdf_reader, corrupt_pdf):
    """Should raise PDFCorruptError for corrupt file."""
    with pytest.raises(PDFCorruptError):
        pdf_reader.validate_file(corrupt_pdf)


def test_validate_empty_pdf(pdf_reader, tmp_path):
    """Should raise PDFCorruptError for empty file."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()
    with pytest.raises(PDFCorruptError):
        pdf_reader.validate_file(empty_pdf)


def test_extract_text_valid_pdf(pdf_reader, sample_pdf):
    """Should extract text and metadata from valid PDF."""
    result = pdf_reader.extract_text(sample_pdf)
    assert isinstance(result, dict)
    assert "text" in result
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["file_name"] == "sample.pdf"


def test_extract_text_with_custom_size(tmp_path):
    """Should respect custom minimum file size."""
    reader = PDFReader(min_file_size=100)
    small_pdf = tmp_path / "small.pdf"
    small_pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(PDFCorruptError):
        reader.extract_text(small_pdf)


def test_extract_text_nonexistent_pdf(pdf_reader, tmp_path):
    """Should raise PDFNotFoundError for non-existent file."""
    with pytest.raises(PDFNotFoundError):
        pdf_reader.extract_text(tmp_path / "nonexistent.pdf")


def test_extract_text_corrupt_pdf(pdf_reader, corrupt_pdf):
    """Should raise PDFCorruptError for corrupt file."""
    with pytest.raises(PDFCorruptError):
        pdf_reader.extract_text(corrupt_pdf)
