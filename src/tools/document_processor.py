"""
Document processor for the RAG system.

This script processes text documents and prepares them for embedding generation
and ingestion into the RAG system.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, fixing encoding issues, etc.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€"', '–')
    text = text.replace('â€"', '—')
    
    return text.strip()


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of a maximum size with overlap.
    
    Args:
        text: Text to split into chunks
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + max_chunk_size
        
        # If we're not at the end of the text, try to find a natural break point
        if end < len(text):
            # Try to find a paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + max_chunk_size // 2:
                end = paragraph_break + 2  # Include the paragraph break
            else:
                # Try to find a sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + max_chunk_size // 2:
                    end = sentence_break + 2  # Include the period and space
                else:
                    # Fall back to a word break
                    word_break = text.rfind(' ', start, end)
                    if word_break != -1:
                        end = word_break + 1  # Include the space
        
        # Add the chunk
        chunks.append(text[start:end].strip())
        
        # Move the start position for the next chunk, accounting for overlap
        start = end - overlap
        
        # Make sure we're making progress
        if start >= len(text):
            break
    
    return chunks


def process_document(
    file_path: str, 
    output_dir: str, 
    topic: str = "great_depression",
    max_chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Process a document and prepare it for embedding generation.
    
    Args:
        file_path: Path to the document file
        output_dir: Directory to save the processed document
        topic: Topic to assign to the document
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of paths to the processed document chunks
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Document file not found: {file_path}")
        return []
    
    # Read the document
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading document {file_path}: {str(e)}")
        return []
    
    # Clean the text
    text = clean_text(text)
    
    # Split into chunks
    chunks = chunk_text(text, max_chunk_size, overlap)
    logger.info(f"Split document into {len(chunks)} chunks")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    output_paths = []
    for i, chunk in enumerate(chunks):
        # Create metadata
        metadata = {
            "title": file_path.stem,
            "source": str(file_path),
            "topic": topic,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        
        # Create output filename
        output_filename = f"{file_path.stem}_chunk_{i+1:03d}.json"
        output_path = output_dir / output_filename
        
        # Create document with text and metadata
        document = {
            "text": chunk,
            "metadata": metadata
        }
        
        # Save document
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(document, file, indent=2)
        
        output_paths.append(str(output_path))
    
    logger.info(f"Saved {len(output_paths)} document chunks to {output_dir}")
    return output_paths


def process_directory(
    input_dir: str, 
    output_dir: str, 
    topic: str = "great_depression",
    max_chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Process all documents in a directory.
    
    Args:
        input_dir: Directory containing document files
        output_dir: Directory to save the processed documents
        topic: Topic to assign to the documents
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of paths to the processed document chunks
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    # Find all text files
    text_files = list(input_dir.glob("*.txt"))
    if not text_files:
        logger.warning(f"No text files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(text_files)} text files in {input_dir}")
    
    # Process each document
    output_paths = []
    for file_path in text_files:
        paths = process_document(file_path, output_dir, topic, max_chunk_size, overlap)
        output_paths.extend(paths)
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Process documents for the RAG system")
    parser.add_argument("--input", type=str, required=True, help="Path to document file or directory")
    parser.add_argument("--output-dir", type=str, default="data/documents/processed", help="Directory to save processed documents")
    parser.add_argument("--topic", type=str, default="great_depression", help="Topic to assign to the documents")
    parser.add_argument("--max-chunk-size", type=int, default=1000, help="Maximum size of each chunk in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Number of characters to overlap between chunks")
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return
    
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        # Process single document
        output_paths = process_document(
            input_path, 
            args.output_dir, 
            args.topic,
            args.max_chunk_size,
            args.overlap
        )
        if output_paths:
            logger.info(f"Successfully processed document: {input_path}")
        else:
            logger.error(f"Failed to process document: {input_path}")
    elif input_path.is_dir():
        # Process all documents in directory
        output_paths = process_directory(
            input_path, 
            args.output_dir, 
            args.topic,
            args.max_chunk_size,
            args.overlap
        )
        if output_paths:
            logger.info(f"Successfully processed {len(output_paths)} document chunks")
        else:
            logger.error(f"Failed to process any documents in {input_path}")
    else:
        logger.error(f"Input must be a text file or directory: {input_path}")


if __name__ == "__main__":
    main()
