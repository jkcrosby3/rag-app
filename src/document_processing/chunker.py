"""Text chunking module for the RAG system.

This module provides functionality to split documents into smaller,
semantically meaningful chunks for embedding and retrieval.
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Callable

logger = logging.getLogger(__name__)


class Chunker:
    """Splits documents into smaller chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n"
    ):
        """Initialize the Chunker.

        Args:
            chunk_size: Target size of each chunk in tokens (approximately words)
            chunk_overlap: Number of tokens to overlap between chunks
            separator: String used to split text into smaller units
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def _estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text.
        
        This is a simple approximation based on whitespace splitting.
        For production, consider using a tokenizer from the model you'll use.
        
        Args:
            text: Text to estimate token count for
            
        Returns:
            Estimated token count
        """
        # Simple approximation: split on whitespace
        return len(text.split())
    
    def _split_text_by_separator(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Split text into paragraphs with classification.
        
        Args:
            text: Text to split
            
        Returns:
            List of dictionaries containing paragraph text and classification
        """
        # Split text into paragraphs and extract classification
        paragraphs = []
        current_paragraph = []
        current_classification = None
        
        # Split by lines and process each line
        lines = text.split(self.separator)
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with classification
            if line.startswith('(') and ')' in line:
                current_classification = line.split(')')[0] + ')'
                if current_paragraph:  # Save previous paragraph
                    paragraphs.append({
                        'text': self.separator.join(current_paragraph),
                        'classification': current_classification
                    })
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Add last paragraph if exists
        if current_paragraph:
            paragraphs.append({
                'text': self.separator.join(current_paragraph),
                'classification': current_classification or 'U'
            })
            
        return paragraphs
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
            
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined = current_chunk + self.separator + next_chunk
            if self._estimate_token_count(combined) <= self.chunk_size:
                current_chunk = combined
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
                
        # Add the last chunk
        merged_chunks.append(current_chunk)
        return merged_chunks
        
    def process_document(self, doc: Dict[str, Union[str, Dict]]) -> List[Dict[str, Union[str, int]]]:
        """Process a single document and create chunks with classification.
        
        Args:
            doc: Dictionary containing document text and metadata
            
        Returns:
            List of chunk dictionaries with text, metadata, and classification
        """
        text = doc['text']
        metadata = doc.get('metadata', {})
        
        # Split text into paragraphs with classification
        paragraphs = self._split_text_by_separator(text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        current_classification = None
        
        for paragraph in paragraphs:
            paragraph_text = paragraph['text']
            paragraph_classification = paragraph['classification']
            unit_token_count = self._estimate_token_count(paragraph_text)
            
            # If adding this paragraph would exceed chunk size, create new chunk
            if current_token_count + unit_token_count > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': self.separator.join(current_chunk),
                        'metadata': {
                            **metadata,
                            'classification': current_classification
                        },
                        'token_count': current_token_count
                    })
                current_chunk = []
                current_token_count = 0
            
            current_chunk.append(paragraph_text)
            current_token_count += unit_token_count
            current_classification = paragraph_classification
        
        # Add last chunk if exists
        if current_chunk:
            chunks.append({
                'text': self.separator.join(current_chunk),
                'metadata': {
                    **metadata,
                    'classification': current_classification
                },
                'token_count': current_token_count
            })
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text:
            return []

        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        current_metadata = metadata or {}
        
        # Process each paragraph
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue

            # Extract classification if present
            if "unclassified" in paragraph.lower():
                current_metadata["classification"] = "U"
            elif "confidential" in paragraph.lower():
                current_metadata["classification"] = "C"
            elif "secret" in paragraph.lower():
                current_metadata["classification"] = "S"
            elif "top secret" in paragraph.lower():
                current_metadata["classification"] = "TS"

            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph

            # Add chunk immediately if it contains classification information
            if 'classification' in current_metadata:
                chunks.append({
                    'text': current_chunk,
                    'metadata': current_metadata.copy()  # Copy metadata to avoid overwriting
                })
                current_chunk = ""
                current_metadata = metadata or {}  # Reset metadata

        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk,
                'metadata': current_metadata.copy()  # Copy metadata to avoid overwriting
            })

        return chunks
        overlapped_chunks.append(chunk_dict)
        
        return overlapped_chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """Process a document dict and split its text into chunks.
        
        Args:
            document: Dict containing text and metadata
            
        Returns:
            List of chunk dicts with text and metadata
        """
        if "text" not in document:
            logger.warning("Document has no 'text' field")
            return []
            
        text = document["text"]
        chunks = self.chunk_text(text)
        
        # Add document metadata to each chunk
        for i, chunk in enumerate(chunks):
            # Copy document metadata
            if "metadata" in document:
                chunk["metadata"] = document["metadata"].copy()
            else:
                chunk["metadata"] = {}
                
            # Add chunk-specific metadata
            chunk["metadata"]["chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks


def chunk_processed_documents(
    input_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\processed",
    output_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\chunked",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, int]:
    """Process all documents in the input directory and create chunks.
    
    Args:
        input_dir: Directory containing processed documents
        output_dir: Directory to save chunked documents
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        Dict with statistics about the chunking process
    """
    import json
    import os
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    stats = {
        "total_documents": 0,
        "total_chunks": 0,
        "documents_by_topic": {}
    }
    
    # Process each JSON file in the input directory
    for file_path in input_dir.glob("*.processed.json"):
        try:
            # Load the processed document
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Get document topic
            topic = document.get("metadata", {}).get("topic", "unknown")
            
            # Update stats
            stats["total_documents"] += 1
            if topic not in stats["documents_by_topic"]:
                stats["documents_by_topic"][topic] = {
                    "document_count": 0,
                    "chunk_count": 0
                }
            stats["documents_by_topic"][topic]["document_count"] += 1
            
            # Chunk the document
            chunks = chunker.chunk_document(document)
            stats["total_chunks"] += len(chunks)
            stats["documents_by_topic"][topic]["chunk_count"] += len(chunks)
            
            # Save each chunk as a separate file
            for i, chunk in enumerate(chunks):
                # Create a unique filename
                base_name = file_path.stem.replace(".processed", "")
                chunk_file = output_dir / f"{base_name}.chunk{i}.json"
                
                # Save the chunk
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
            logger.info(f"Processed {file_path.name} into {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return stats


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Chunk documents
    stats = chunk_processed_documents()
    
    logger.info(f"Chunking complete. Processed {stats['total_documents']} documents into {stats['total_chunks']} chunks.")
    for topic, topic_stats in stats["documents_by_topic"].items():
        logger.info(f"Topic '{topic}': {topic_stats['document_count']} documents, {topic_stats['chunk_count']} chunks")
