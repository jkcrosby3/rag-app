#!/usr/bin/env python3
"""
Process newspaper text files directly for the RAG system.

This script processes the downloaded newspaper text files,
chunks them, generates embeddings, and builds the vector database.
"""
import os
import ssl

# Disable SSL verification for corporate networks (Booz Allen)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HTTPX_VERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL warnings
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import logging
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.document_processing.chunker import Chunker
from src.embeddings.generator import EmbeddingGenerator
from src.vector_db.faiss_db import FAISSVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_newspaper_files():
    """Process newspaper text files through the RAG pipeline."""
    
    # Define directories
    text_files_dir = project_root / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    chunked_dir = project_root / "data" / "chunked"
    embedded_dir = project_root / "data" / "embedded"
    vector_db_dir = project_root / "data" / "vector_db"
    
    # Create output directories
    chunked_dir.mkdir(parents=True, exist_ok=True)
    embedded_dir.mkdir(parents=True, exist_ok=True)
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files (not metadata)
    text_files = sorted([f for f in text_files_dir.glob("*.txt") if not f.name.endswith("_metadata.json")])
    
    if not text_files:
        logger.error(f"No text files found in {text_files_dir}")
        return
    
    logger.info(f"Found {len(text_files)} newspaper text files to process")
    
    # Initialize components
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    embedding_generator = EmbeddingGenerator()
    
    all_chunks = []
    
    # Step 1: Chunk all documents
    logger.info("Step 1: Chunking documents...")
    for text_file in tqdm(text_files, desc="Chunking"):
        try:
            # Read text file
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Read metadata if available
            metadata_file = text_file.parent / f"{text_file.stem}_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Create unique document ID from content hash
            doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            
            # Create document structure
            document = {
                "text": text,
                "metadata": {
                    "source": text_file.name,
                    "file_path": str(text_file),
                    "doc_id": f"{text_file.stem}_{doc_hash}",  # Unique ID
                    **metadata
                }
            }
            
            # Chunk the document
            chunks = chunker.chunk_document(document)
            
            # Add to all chunks
            all_chunks.extend(chunks)
            
            logger.debug(f"Chunked {text_file.name}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {e}")
            continue
    
    logger.info(f"Created {len(all_chunks)} total chunks from {len(text_files)} documents")
    
    # Save chunked documents
    chunked_file = chunked_dir / "newspapers_chunks.json"
    with open(chunked_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunks to {chunked_file}")
    
    # Step 2: Generate embeddings
    logger.info("Step 2: Generating embeddings...")
    embedded_chunks = []
    
    for chunk in tqdm(all_chunks, desc="Embedding"):
        try:
            # Generate embedding
            embedding = embedding_generator.generate_embedding(chunk['text'])
            
            # Add embedding to chunk
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding  # Already a list
            embedded_chunks.append(chunk_with_embedding)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            continue
    
    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
    
    # Save embedded documents
    embedded_file = embedded_dir / "newspapers_embedded.json"
    with open(embedded_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved embedded chunks to {embedded_file}")
    
    # Step 3: Build vector database
    logger.info("Step 3: Building vector database...")
    try:
        vector_db = FAISSVectorDB(index_path=str(vector_db_dir / "faiss.index"))
        
        # Get existing document IDs to avoid duplicates
        existing_doc_ids = set()
        for doc_data in vector_db.document_lookup.values():
            if 'metadata' in doc_data and 'doc_id' in doc_data['metadata']:
                existing_doc_ids.add(doc_data['metadata']['doc_id'])
        
        logger.info(f"Found {len(existing_doc_ids)} existing documents in database")
        
        # Add documents to vector database (skip duplicates)
        added_count = 0
        skipped_count = 0
        for chunk in tqdm(embedded_chunks, desc="Building DB"):
            doc_id = chunk['metadata'].get('doc_id', '')
            
            # Skip if this document already exists (based on content hash)
            if doc_id in existing_doc_ids:
                skipped_count += 1
                continue
            
            # Prepare document dict for FAISS
            doc = {
                'text': chunk['text'],
                'embedding': chunk['embedding'],
                'metadata': chunk['metadata']
            }
            vector_db.add_document(doc)
            added_count += 1
        
        logger.info(f"Added {added_count} new documents, skipped {skipped_count} duplicates")
        
        # Save the vector database
        vector_db.save(str(vector_db_dir / "faiss.index"))
        logger.info(f"Vector database saved to {vector_db_dir}")
        
        # Print statistics
        logger.info(f"Vector database statistics:")
        logger.info(f"  Total documents: {len(embedded_chunks)}")
        logger.info(f"  Index type: {vector_db.index_type}")
        
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        raise
    
    logger.info("✅ Processing complete! Your RAG system is ready to use.")
    logger.info(f"   - Processed {len(text_files)} newspaper files")
    logger.info(f"   - Created {len(all_chunks)} chunks")
    logger.info(f"   - Generated {len(embedded_chunks)} embeddings")
    logger.info(f"   - Built vector database with {len(embedded_chunks)} documents")


if __name__ == "__main__":
    try:
        process_newspaper_files()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
