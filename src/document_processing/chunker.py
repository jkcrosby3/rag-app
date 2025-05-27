"""Text chunking module for the RAG system.

This module provides functionality to split documents into smaller,
semantically meaningful chunks for embedding and retrieval.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

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
    
    def _split_text_by_separator(self, text: str) -> List[str]:
        """Split text into smaller units using the separator.
        
        Args:
            text: Text to split
            
        Returns:
            List of text units
        """
        # Handle case where separator isn't found
        if self.separator not in text:
            return [text]
            
        # Split by separator and filter out empty strings
        return [unit for unit in text.split(self.separator) if unit.strip()]
    
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
        
    def chunk_text(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Split text into chunks with metadata.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of dicts containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
            
        # Split text into smaller units
        text_units = self._split_text_by_separator(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for i, unit in enumerate(text_units):
            unit_size = self._estimate_token_count(unit)
            
            # If a single unit is larger than chunk_size, we need to split it further
            if unit_size > self.chunk_size:
                # If we have accumulated text, add it as a chunk first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_size = 0
                
                # Split large unit into smaller pieces
                words = unit.split()
                sub_chunk = ""
                sub_size = 0
                
                for word in words:
                    word_size = 1  # Approximate a word as one token
                    if sub_size + word_size <= self.chunk_size:
                        sub_chunk += " " + word if sub_chunk else word
                        sub_size += word_size
                    else:
                        chunks.append(sub_chunk)
                        sub_chunk = word
                        sub_size = word_size
                
                # Add the last sub-chunk if it exists
                if sub_chunk:
                    chunks.append(sub_chunk)
                
            # Normal case: unit fits in a chunk
            elif current_size + unit_size <= self.chunk_size:
                separator = self.separator if current_chunk else ""
                current_chunk += separator + unit
                current_size += unit_size
            
            # Unit doesn't fit in current chunk
            else:
                # Add current chunk to the list
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start a new chunk with this unit
                current_chunk = unit
                current_size = unit_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        # Create chunks with overlap
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            chunk_text = chunks[i]
            
            # If not the last chunk and overlap is specified, add overlap
            if i < len(chunks) - 1 and self.chunk_overlap > 0:
                next_chunk = chunks[i + 1]
                next_words = next_chunk.split()
                overlap_word_count = min(self.chunk_overlap, len(next_words))
                
                if overlap_word_count > 0:
                    overlap_text = " ".join(next_words[:overlap_word_count])
                    chunk_text += self.separator + overlap_text
            
            # Create chunk with metadata
            chunk_dict = {
                "text": chunk_text,
                "chunk_index": i,
                "token_count": self._estimate_token_count(chunk_text)
            }
            
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
