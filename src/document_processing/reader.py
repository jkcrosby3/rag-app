"""Text file reader implementation for the RAG system.

This module provides functionality to read and validate text files,
implementing the basic text reader component of the RAG system.
"""
from pathlib import Path
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)

class TextReaderError(Exception):
    """Base exception for text reader errors."""
    pass

class FileNotFoundError(TextReaderError):
    """Raised when the input file does not exist."""
    pass

class EmptyFileError(TextReaderError):
    """Raised when the input file is empty."""
    pass

class TextReader:
    """Handles reading and basic validation of text files."""
    
    def __init__(self, min_file_size: int = 1):
        """Initialize the TextReader.
        
        Args:
            min_file_size: Minimum acceptable file size in bytes
        """
        self.min_file_size = min_file_size
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate that the file exists and is not empty.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            bool: True if file is valid
            
        Raises:
            FileNotFoundError: If file does not exist
            EmptyFileError: If file is empty
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.stat().st_size < self.min_file_size:
            raise EmptyFileError(f"File is empty: {file_path}")
        
        return True
    
    def read_file(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """Read a text file and return its contents with metadata.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dict containing:
                - text: The file contents as a string
                - metadata: Dict with file info (size, path)
                
        Raises:
            FileNotFoundError: If file does not exist
            EmptyFileError: If file is empty
        """
        path = Path(file_path)
        self.validate_file(path)
        
        logger.info(f"Reading file: {path}")
        text = path.read_text(encoding='utf-8')
        
        metadata = {
            'file_path': str(path.absolute()),
            'file_size': path.stat().st_size,
            'file_name': path.name
        }
        
        return {
            'text': text,
            'metadata': metadata
        }
