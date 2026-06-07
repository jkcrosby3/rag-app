"""Batch document processor for the RAG system.

This module provides functionality to process multiple documents in batch,
extracting text and metadata from various file formats.
"""
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.document_processing.reader import TextReader

# Try to import PDF reader, but make it optional
try:
    from src.document_processing.pdf_reader import PDFReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger = logging.getLogger(__name__)
    logger.warning("PDF support not available (fitz/PyMuPDF not installed)")

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
        self.pdf_reader = PDFReader() if PDF_SUPPORT else None
        
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
            if not self.pdf_reader:
                raise ValueError("PDF support not available. Install PyMuPDF (fitz) to process PDF files.")
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
        
        if PDF_SUPPORT and isinstance(reader, PDFReader):
            result = reader.extract_text(file_path)
        else:
            result = reader.read_file(file_path)
        
        # Check for companion metadata JSON file
        metadata_json = self._find_metadata_json(file_path)
        if metadata_json:
            result = self._merge_metadata(result, metadata_json)
        
        return result

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
    
    def _find_metadata_json(self, file_path: Path) -> Optional[Path]:
        """Find companion metadata JSON file for a text file.
        
        Looks for patterns like:
        - filename.txt -> filename_metadata.json
        - filename.txt -> filename-metadata.json
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Path to metadata JSON if found, None otherwise
        """
        # Try common metadata file patterns
        base_name = file_path.stem  # filename without extension
        parent_dir = file_path.parent
        
        patterns = [
            f"{base_name}_metadata.json",
            f"{base_name}-metadata.json",
            f"{base_name}.metadata.json"
        ]
        
        for pattern in patterns:
            metadata_path = parent_dir / pattern
            if metadata_path.exists():
                logger.info(f"Found metadata file: {metadata_path}")
                return metadata_path
        
        return None
    
    def _merge_metadata(self, result: Dict, metadata_path: Path) -> Dict:
        """Merge enriched metadata from JSON file into result.
        
        Args:
            result: Dict with text and basic metadata
            metadata_path: Path to metadata JSON file
            
        Returns:
            Updated result dict with merged metadata
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                enriched_metadata = json.load(f)
            
            # Merge enriched metadata into result's metadata
            # Preserve original metadata, add enriched fields
            result['metadata']['enriched'] = enriched_metadata
            
            # Also add key fields to top-level metadata for easy access
            if 'people_mentioned' in enriched_metadata:
                result['metadata']['people_mentioned'] = enriched_metadata['people_mentioned']
            if 'battles_mentioned' in enriched_metadata:
                result['metadata']['battles_mentioned'] = enriched_metadata['battles_mentioned']
            if 'places_mentioned' in enriched_metadata:
                result['metadata']['places_mentioned'] = enriched_metadata['places_mentioned']
            if 'military_units_mentioned' in enriched_metadata:
                result['metadata']['military_units_mentioned'] = enriched_metadata['military_units_mentioned']
            if 'war_relevance_score' in enriched_metadata:
                result['metadata']['war_relevance_score'] = enriched_metadata['war_relevance_score']
            if 'subjects' in enriched_metadata:
                result['metadata']['subjects'] = enriched_metadata['subjects']
            
            logger.info(f"Merged enriched metadata from {metadata_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to merge metadata from {metadata_path}: {e}")
        
        return result
    
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
