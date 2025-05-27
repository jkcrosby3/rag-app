"""Batch document processor for the RAG system.

This module provides functionality to process multiple documents in batch,
extracting text and metadata from various file formats.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.document_processing.reader import TextReader
from src.document_processing.pdf_reader import PDFReader

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes multiple documents in batch, extracting text and metadata."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the BatchProcessor.

        Args:
            output_dir: Directory to save processed documents.
                If None, processed data will only be returned, not saved.
        """
        self.text_reader = TextReader()
        self.pdf_reader = PDFReader()
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def _get_reader_for_file(self, file_path: Path):
        """Get the appropriate reader for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Appropriate reader object for the file type
        """
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return self.pdf_reader
        else:
            # Default to text reader for .txt and other text files
            return self.text_reader

    def _extract_text_from_file(self, file_path: Path) -> Dict:
        """Extract text and metadata from a single file.

        Args:
            file_path: Path to the file

        Returns:
            Dict containing extracted text and metadata

        Raises:
            Various exceptions from the underlying readers
        """
        reader = self._get_reader_for_file(file_path)
        
        if isinstance(reader, PDFReader):
            return reader.extract_text(file_path)
        else:
            return reader.read_file(file_path)

    def process_directory(
        self, 
        directory: Union[str, Path], 
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict]:
        """Process all files in a directory.

        Args:
            directory: Directory containing files to process
            recursive: Whether to process subdirectories
            file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
                If None, process all files

        Returns:
            List of dicts containing extracted text and metadata for each file
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        logger.info(f"Processing directory: {directory}")
        
        results = []
        
        # Standardize file extensions format
        if file_extensions:
            file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                              for ext in file_extensions]
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            
            # Skip if not recursive and not the top directory
            if not recursive and root_path != directory:
                continue
                
            for file in files:
                file_path = root_path / file
                
                # Skip if not in allowed extensions
                if file_extensions and file_path.suffix.lower() not in file_extensions:
                    continue
                    
                try:
                    logger.info(f"Processing file: {file_path}")
                    result = self._extract_text_from_file(file_path)
                    
                    # Add relative path to metadata for context
                    result['metadata']['relative_path'] = str(file_path.relative_to(directory))
                    result['metadata']['topic'] = self._extract_topic_from_path(file_path, directory)
                    
                    # Save to output directory if specified
                    if self.output_dir:
                        self._save_processed_result(result)
                        
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    
        return results
    
    def _extract_topic_from_path(self, file_path: Path, base_dir: Path) -> str:
        """Extract topic information from file path.
        
        For our Great Depression documents, the topic is the parent directory name.
        
        Args:
            file_path: Path to the file
            base_dir: Base directory for processing
            
        Returns:
            Topic string extracted from path
        """
        try:
            rel_path = file_path.relative_to(base_dir)
            parts = rel_path.parts
            
            # For our structure, the topic is the first directory
            if len(parts) > 1:
                return parts[0]
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _save_processed_result(self, result: Dict):
        """Save processed result to output directory.
        
        Args:
            result: Dict containing text and metadata
        """
        if not self.output_dir:
            return
            
        # Create a unique filename based on the original path
        rel_path = result['metadata'].get('relative_path', 'unknown')
        safe_name = rel_path.replace('/', '_').replace('\\', '_')
        output_path = self.output_dir / f"{safe_name}.processed.json"
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved processed result to {output_path}")


def process_great_depression_documents(
    base_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\great_depression",
    output_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\processed"
) -> List[Dict]:
    """Process all Great Depression documents.
    
    Args:
        base_dir: Base directory containing Great Depression documents
        output_dir: Directory to save processed results
        
    Returns:
        List of processed document results
    """
    processor = BatchProcessor(output_dir=output_dir)
    
    # Process all text and PDF files
    results = processor.process_directory(
        base_dir,
        recursive=True,
        file_extensions=['.txt', '.pdf']
    )
    
    logger.info(f"Processed {len(results)} documents")
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process documents
    process_great_depression_documents()
