"""Unit tests for the text reader module."""
import pytest
from pathlib import Path
from src.document_processing.reader import (
    TextReader, FileNotFoundError, EmptyFileError
)


@pytest.fixture
def text_reader():
    """Create a TextReader instance for testing."""
    return TextReader()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test file.\nIt has multiple lines.\n")
    return file_path


@pytest.fixture
def empty_text_file(tmp_path):
    """Create an empty text file for testing."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()
    return file_path


def test_read_valid_file(text_reader, sample_text_file):
    """Test reading a valid text file."""
    result = text_reader.read_file(sample_text_file)

    assert 'text' in result
    assert 'metadata' in result
    assert result['text'] == "This is a test file.\nIt has multiple lines.\n"
    assert result['metadata']['file_name'] == "test.txt"
    assert result['metadata']['file_size'] > 0


def test_read_nonexistent_file(text_reader, tmp_path):
    """Test reading a file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        text_reader.read_file(tmp_path / "nonexistent.txt")


def test_read_empty_file(text_reader, empty_text_file):
    """Test reading an empty file."""
    with pytest.raises(EmptyFileError):
        text_reader.read_file(empty_text_file)


def test_validate_file(text_reader, sample_text_file):
    """Test file validation."""
    assert text_reader.validate_file(sample_text_file) is True


def test_custom_min_file_size():
    """Test custom minimum file size validation."""
    reader = TextReader(min_file_size=100)  # Require at least 100 bytes

    # Create a small file
    file_path = Path("test_small.txt")
    file_path.write_text("Small")

    try:
        with pytest.raises(EmptyFileError):
            reader.validate_file(file_path)
    finally:
        # Clean up
        file_path.unlink()
