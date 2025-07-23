#!/usr/bin/env python
"""
Script to update the document processing pipeline to handle the new business plan
document organization structure.

This script extends the existing document processing capabilities to extract
metadata from the hierarchical business plan documents.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import RAG system components
try:
    from src.document_processing.batch_processor import BatchProcessor
    from src.document_processing.chunker import Chunker
except ImportError as e:
    logger.error(f"Error importing RAG system components: {str(e)}")
    sys.exit(1)

class BusinessPlanProcessor(BatchProcessor):
    """
    Extended BatchProcessor for business plan documents.
    Handles hierarchical document relationships and enhanced metadata extraction.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the BusinessPlanProcessor.
        
        Args:
            output_dir: Directory to save processed documents
        """
        super().__init__(output_dir)
    
    def _extract_plan_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract plan metadata from filename and path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        # Extract plan number
        plan_number_match = re.search(r'plan_(\d+)', file_path.name)
        if plan_number_match:
            metadata["plan_number"] = plan_number_match.group(1)
        
        # Extract document type
        if "-main" in file_path.name:
            metadata["document_type"] = "main"
            metadata["hierarchy_level"] = 1
        elif "-annex" in file_path.name and "-appendix" in file_path.name:
            # Extract annex and appendix letters
            annex_match = re.search(r'-annex([A-Z])', file_path.name)
            appendix_match = re.search(r'-appendix([A-Z])', file_path.name)
            
            if annex_match and appendix_match:
                metadata["document_type"] = f"annex{annex_match.group(1)}-appendix{appendix_match.group(1)}"
                metadata["annex"] = annex_match.group(1)
                metadata["appendix"] = appendix_match.group(1)
                metadata["hierarchy_level"] = 3
        elif "-annex" in file_path.name:
            # Extract annex letter
            annex_match = re.search(r'-annex([A-Z])', file_path.name)
            
            if annex_match:
                metadata["document_type"] = f"annex{annex_match.group(1)}"
                metadata["annex"] = annex_match.group(1)
                metadata["hierarchy_level"] = 2
        elif "-supporting" in file_path.name:
            metadata["document_type"] = "supporting"
            metadata["hierarchy_level"] = 0
            
            # Extract supporting document type
            support_match = re.search(r'-supporting-([a-z_]+)', file_path.name)
            if support_match:
                metadata["supporting_type"] = support_match.group(1)
        
        # Extract status if present
        status_match = re.search(r'-(DRAFT|APPROVED|EXECUTED)', file_path.name)
        if status_match:
            metadata["status"] = status_match.group(1)
        
        return metadata
    
    def _load_relationship_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load relationship data for a plan if available.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing relationship data or None if not found
        """
        # Extract plan number
        plan_number_match = re.search(r'plan_(\d+)', file_path.name)
        if not plan_number_match:
            return None
            
        plan_number = plan_number_match.group(1)
        
        # Look for relationship file
        plan_dir = file_path.parent
        relationship_file = plan_dir / f"plan_{plan_number}-relationships.json"
        
        if relationship_file.exists():
            try:
                with open(relationship_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading relationship file {relationship_file}: {str(e)}")
        
        return None
    
    def _extract_document_relationships(self, file_path: Path, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract document relationships from relationship data.
        
        Args:
            file_path: Path to the document file
            relationship_data: Relationship data for the plan
            
        Returns:
            Dictionary containing relationship information
        """
        relationships = {}
        
        # Get the filename
        filename = file_path.name
        
        # Check if this is the main document
        if relationship_data.get("main_document") == filename:
            relationships["document_role"] = "main"
            relationships["related_documents"] = [
                annex["file"] for annex in relationship_data.get("annexes", [])
            ]
            return relationships
        
        # Check if this is an annex
        for annex in relationship_data.get("annexes", []):
            if annex.get("file") == filename:
                relationships["document_role"] = "annex"
                relationships["annex_id"] = annex.get("id")
                relationships["department"] = annex.get("department")
                relationships["related_documents"] = [
                    appendix["file"] for appendix in annex.get("appendices", [])
                ]
                relationships["parent_document"] = relationship_data.get("main_document")
                return relationships
            
            # Check if this is an appendix
            for appendix in annex.get("appendices", []):
                if appendix.get("file") == filename:
                    relationships["document_role"] = "appendix"
                    relationships["appendix_id"] = appendix.get("id")
                    relationships["annex_id"] = annex.get("id")
                    relationships["answers_questions"] = appendix.get("answers_questions", [])
                    relationships["parent_document"] = annex.get("file")
                    relationships["root_document"] = relationship_data.get("main_document")
                    return relationships
        
        return relationships
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file, extracting text and enhanced metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing extracted text and metadata
        """
        # Use the parent class to extract basic text and metadata
        result = super().process_file(file_path)
        
        # Extract plan-specific metadata
        plan_metadata = self._extract_plan_metadata(file_path)
        result["metadata"].update(plan_metadata)
        
        # Load relationship data if available
        relationship_data = self._load_relationship_data(file_path)
        if relationship_data:
            # Extract document relationships
            relationships = self._extract_document_relationships(file_path, relationship_data)
            result["metadata"]["relationships"] = relationships
        
        # Check for external metadata file
        metadata_file = file_path.with_suffix(".metadata.json")
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    external_metadata = json.load(f)
                    # Add external metadata, but don't overwrite existing metadata
                    for key, value in external_metadata.items():
                        if key not in result["metadata"]:
                            result["metadata"][key] = value
            except Exception as e:
                logger.error(f"Error loading metadata file {metadata_file}: {str(e)}")
        
        return result

def process_business_plans(
    input_dir: Union[str, Path] = "data/documents/plans",
    output_dir: Union[str, Path] = "data/documents/processed_plans",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> None:
    """
    Process business plan documents and prepare them for embedding.
    
    Args:
        input_dir: Directory containing business plan documents
        output_dir: Directory to save processed documents
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = BusinessPlanProcessor(output_dir)
    
    # Process all files in the input directory
    processed_docs = processor.process_directory(input_dir, recursive=True)
    
    logger.info(f"Processed {len(processed_docs)} documents")
    
    # Initialize chunker
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Chunk processed documents
    chunked_dir = output_dir.parent / "chunked_plans"
    chunked_dir.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    
    # Process each document
    for doc in processed_docs:
        try:
            # Create chunks
            chunks = chunker.create_chunks(doc["text"])
            
            # Get base filename
            base_name = Path(doc["metadata"].get("source", "unknown")).stem
            
            # Save each chunk
            for i, chunk_text in enumerate(chunks):
                # Create chunk with metadata
                chunk = {
                    "text": chunk_text,
                    "metadata": doc["metadata"].copy()
                }
                
                # Add chunk-specific metadata
                chunk["metadata"]["chunk_index"] = i
                chunk["metadata"]["total_chunks"] = len(chunks)
                
                # Create a unique filename
                chunk_file = chunked_dir / f"{base_name}.chunk{i}.json"
                
                # Save the chunk
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                
                total_chunks += 1
                
        except Exception as e:
            logger.error(f"Error chunking document {doc.get('metadata', {}).get('source')}: {str(e)}")
    
    logger.info(f"Created {total_chunks} chunks from {len(processed_docs)} documents")

def main():
    """Main function to process business plan documents."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process business plan documents")
    parser.add_argument("--input-dir", type=str, default="data/documents/plans",
                       help="Directory containing business plan documents")
    parser.add_argument("--output-dir", type=str, default="data/documents/processed_plans",
                       help="Directory to save processed documents")
    parser.add_argument("--chunk-size", type=int, default=500,
                       help="Size of each chunk in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                       help="Overlap between chunks in tokens")
    
    args = parser.parse_args()
    
    # Process business plan documents
    process_business_plans(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

if __name__ == "__main__":
    main()
