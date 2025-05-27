"""FAISS vector database implementation for the RAG system.

This module provides a vector database implementation using Facebook AI
Similarity Search (FAISS) for efficient similarity search and retrieval.
"""
import logging
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorDB:
    """Vector database implementation using FAISS."""

    def __init__(
        self,
        dimension: int = 384,  # Default dimension for all-MiniLM-L6-v2 model
        index_type: str = "L2",
        index_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the FAISS vector database.

        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use ('L2', 'IP', 'Cosine')
                L2: Euclidean distance (smaller is more similar)
                IP: Inner Product (larger is more similar)
                Cosine: Cosine similarity (larger is more similar)
            index_path: Path to load an existing index from
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.document_lookup = {}  # Maps index id to document data
        
        # Create or load index
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            self._create_index()
    
    def _create_index(self, vector_count=None):
        """Create a new FAISS index.
        
        Dynamically selects between flat and IVF index based on vector count.
        For small datasets, we use a flat index for simplicity and accuracy.
        For larger datasets, an IVF index is used for better performance.
        
        Args:
            vector_count: Optional count of vectors to be indexed. If provided,
                          will be used to determine the appropriate index type.
                          If None, defaults to flat index.
        
        Flat Index vs. IVF Index Guidelines:
        - Flat Index: Performs exhaustive search, comparing query to every vector
          - Pros: Most accurate results, simple implementation, no training needed
          - Cons: Search time scales linearly with dataset size O(n)
          - Used for: Datasets with fewer than 1,000,000 vectors
        
        - IVF Index: Uses clustering to partition vectors into cells for faster search
          - Pros: Much faster search times for large datasets O(log n)
          - Cons: Requires training, slightly reduced accuracy, more complex
          - Used when: Dataset exceeds 1,000,000 vectors
        """
        # Constants for index selection
        VECTOR_COUNT_THRESHOLD = 1_000_000  # Switch to IVF when exceeding this count
        
        # Determine if we should use IVF index based on vector count
        use_ivf = False
        if vector_count is not None and vector_count >= VECTOR_COUNT_THRESHOLD:
            use_ivf = True
            logger.info(f"Vector count {vector_count} exceeds threshold {VECTOR_COUNT_THRESHOLD}, using IVF index")
        
        # Create the appropriate index based on similarity metric and vector count
        if self.index_type == "L2":
            if use_ivf:
                # For IVF index, nlist (number of clusters) should be approximately sqrt(n)
                nlist = max(int(vector_count ** 0.5), 100)  # At least 100 clusters
                # Create IVF index with flat quantizer
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                logger.info(f"Created FAISS IVF index with {nlist} clusters, dimension {self.dimension}, and type {self.index_type}")
                self.trained = False  # IVF indexes need training
            else:
                # L2 distance (Euclidean) with flat index
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created FAISS flat index with dimension {self.dimension} and type {self.index_type}")
                self.trained = True  # Flat indexes don't need training
        
        elif self.index_type == "IP":
            if use_ivf:
                # For IVF index with Inner Product similarity
                nlist = max(int(vector_count ** 0.5), 100)
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                logger.info(f"Created FAISS IVF index with {nlist} clusters, dimension {self.dimension}, and type {self.index_type}")
                self.trained = False
            else:
                # Inner Product (dot product) with flat index
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created FAISS flat index with dimension {self.dimension} and type {self.index_type}")
                self.trained = True
        
        elif self.index_type == "Cosine":
            if use_ivf:
                # For IVF index with Cosine similarity (normalized inner product)
                nlist = max(int(vector_count ** 0.5), 100)
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                logger.info(f"Created FAISS IVF index with {nlist} clusters, dimension {self.dimension}, and type {self.index_type}")
                self.trained = False
            else:
                # Cosine similarity (normalized inner product) with flat index
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created FAISS flat index with dimension {self.dimension} and type {self.index_type}")
                self.trained = True
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_document(self, document: Dict[str, Any]) -> int:
        """Add a document with embedding to the index.
        
        Args:
            document: Document dict containing 'embedding' and other metadata
            
        Returns:
            ID of the document in the index
        """
        if 'embedding' not in document:
            raise ValueError("Document must contain an 'embedding' field")
            
        embedding = document['embedding']
        
        # Convert to numpy array and ensure correct shape
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.index_type == "Cosine":
            faiss.normalize_L2(embedding_array)
        
        # Get the next ID
        doc_id = len(self.document_lookup)
        
        # Add to index
        self.index.add(embedding_array)
        
        # Store document data without the embedding to save memory
        doc_data = {k: v for k, v in document.items() if k != 'embedding'}
        doc_data['id'] = doc_id
        self.document_lookup[doc_id] = doc_data
        
        return doc_id
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """Add multiple documents with embeddings to the index.
        
        Args:
            documents: List of document dicts, each containing 'embedding' and other metadata
            
        Returns:
            List of document IDs in the index
        """
        if not documents:
            return []
        
        # Extract embeddings for batch processing
        embeddings = []
        for doc in documents:
            if 'embedding' not in doc:
                raise ValueError("All documents must contain an 'embedding' field")
            embeddings.append(doc['embedding'])
        
        # Convert to numpy array
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize if using cosine similarity
        if self.index_type == "Cosine":
            faiss.normalize_L2(embedding_array)
        
        # Calculate total vector count (existing + new)
        total_vector_count = len(self.document_lookup) + len(documents)
        
        # Check if we need to recreate the index based on vector count
        # This is only done when we're transitioning from flat to IVF
        current_index_is_flat = isinstance(self.index, faiss.IndexFlat) or \
                              isinstance(self.index, faiss.IndexFlatL2) or \
                              isinstance(self.index, faiss.IndexFlatIP)
        
        # If we have a flat index but now have enough vectors for IVF, recreate the index
        if current_index_is_flat and total_vector_count >= 1_000_000:
            logger.info(f"Vector count {total_vector_count} exceeds threshold for flat index, recreating as IVF index")
            
            # Save existing vectors if we have any
            existing_vectors = None
            if len(self.document_lookup) > 0:
                # Extract existing vectors
                logger.info(f"Extracting {len(self.document_lookup)} existing vectors for index recreation")
                existing_vectors = self.index.reconstruct_n(0, len(self.document_lookup))
            
            # Recreate the index with the appropriate type
            self._create_index(vector_count=total_vector_count)
            
            # Add back existing vectors if we had any
            if existing_vectors is not None:
                logger.info(f"Adding {len(existing_vectors)} existing vectors to new index")
                if self.index_type == "Cosine":
                    faiss.normalize_L2(existing_vectors)
                self.index.add(existing_vectors)
        
        # Train index if not trained and we have enough vectors
        if not getattr(self, 'trained', False) and len(documents) >= 16:
            try:
                logger.info(f"Training IVF index with {len(embedding_array)} vectors")
                self.index.train(embedding_array)
                self.trained = True
            except Exception as e:
                logger.warning(f"Failed to train index: {str(e)}. Will continue with untrained index.")
        
        # Get starting ID
        start_id = len(self.document_lookup)
        
        # Add embeddings to index in batch
        self.index.add(embedding_array)
        
        # Store document data without embeddings to save memory
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            doc_data = {k: v for k, v in doc.items() if k != 'embedding'}
            doc_data['id'] = doc_id
            self.document_lookup[doc_id] = doc_data
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of document dicts with similarity scores
        """
        if not self.index:
            raise ValueError("Index not initialized")
            
        if len(self.document_lookup) == 0:
            logger.warning("Search called on empty index")
            return []
        
        # Skip search if index is not trained and is an IVF index
        if not getattr(self, 'trained', True) and isinstance(self.index, faiss.IndexIVF):
            logger.warning("IVF index not trained, returning random documents")
            # Return random documents as fallback
            indices = np.random.choice(len(self.document_lookup), min(k, len(self.document_lookup)), replace=False)
            results = []
            for idx in indices:
                doc_data = self.document_lookup[idx].copy()
                doc_data['similarity'] = 0.0  # No meaningful similarity score
                results.append(doc_data)
            return results
            
        # Convert to numpy array and ensure correct shape
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.index_type == "Cosine":
            faiss.normalize_L2(query_array)
        
        # Adjust nprobe based on index size for better recall
        if isinstance(self.index, faiss.IndexIVF):
            # Increase nprobe for small indices to improve recall
            if len(self.document_lookup) < 100:
                self.index.nprobe = min(16, self.index.nlist)  # More exhaustive search for small indices
            else:
                self.index.nprobe = 4  # Default for larger indices
            
        # Perform the search
        distances, indices = self.index.search(query_array, min(k, len(self.document_lookup)))
        
        # Flatten results
        distances = distances[0]
        indices = indices[0]
        
        # Convert to similarity score if needed
        if self.index_type == "L2":
            # Convert L2 distance to similarity score (smaller distance = higher similarity)
            # Add small epsilon to avoid division by zero
            similarities = 1.0 / (distances + 1e-10)
        else:
            # For IP and Cosine, higher is already more similar
            similarities = distances
            
        # Build result list
        results = []
        for i, (idx, similarity) in enumerate(zip(indices, similarities)):
            # Skip invalid indices (can happen with small indices)
            if idx == -1 or idx >= len(self.document_lookup):
                continue
                
            # Get document data
            doc_data = self.document_lookup[idx].copy()
            
            # Add similarity score
            doc_data['similarity'] = float(similarity)
            
            results.append(doc_data)
            
        return results
    
    def save(self, path: Union[str, Path]):
        """Save the index and document lookup to disk.
        
        Args:
            path: Directory path to save the index and metadata
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save the document lookup
        lookup_path = path / "document_lookup.pkl"
        with open(lookup_path, 'wb') as f:
            pickle.dump(self.document_lookup, f)
            
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "document_count": len(self.document_lookup)
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved FAISS index with {len(self.document_lookup)} documents to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load the index and document lookup from disk.
        
        Args:
            path: Directory path containing the index and metadata
        """
        path = Path(path)
        
        # Load the FAISS index
        index_path = path / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
            
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        
        # Load the document lookup
        lookup_path = path / "document_lookup.pkl"
        if not lookup_path.exists():
            raise FileNotFoundError(f"Document lookup not found at {lookup_path}")
            
        with open(lookup_path, 'rb') as f:
            self.document_lookup = pickle.load(f)
            
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.index_type = metadata.get("index_type", "L2")
                
        logger.info(f"Loaded FAISS index with {len(self.document_lookup)} documents from {path}")
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dict or None if not found
        """
        return self.document_lookup.get(doc_id)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index.
        
        Returns:
            Number of documents
        """
        return len(self.document_lookup)
    
    def clear(self):
        """Clear the index and document lookup."""
        self._create_index()
        self.document_lookup = {}
        logger.info("Cleared FAISS index and document lookup")


def build_vector_db_from_embedded_docs(
    input_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\embedded",
    output_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\vector_db",
    index_type: str = "Cosine"
) -> Dict[str, Any]:
    """Build a vector database from embedded documents.
    
    Args:
        input_dir: Directory containing embedded documents
        output_dir: Directory to save the vector database
        index_type: Type of FAISS index to use ('L2', 'IP', 'Cosine')
        
    Returns:
        Dict with statistics about the vector database
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, count the total number of documents to determine appropriate index type
    document_files = list(input_dir.glob("*.embedded*.json"))
    total_document_count = len(document_files)
    logger.info(f"Found {total_document_count} embedded documents in {input_dir}")
    
    if not document_files:
        raise FileNotFoundError(f"No embedded documents found in {input_dir}")
    
    # Get the dimension from the first document
    with open(document_files[0], 'r', encoding='utf-8') as f:
        first_doc = json.load(f)
        dimension = len(first_doc.get('embedding', []))
    
    # Initialize vector database with appropriate index type based on document count
    vector_db = FAISSVectorDB(dimension=dimension, index_type=index_type)
    
    # Recreate the index with the appropriate type based on vector count
    vector_db._create_index(vector_count=total_document_count)
    
    # Prepare to collect documents in batches for more efficient processing
    batch_size = 1000  # Process documents in batches of 1000 for efficiency
    current_batch = []
    
    stats = {
        "total_documents": 0,
        "documents_by_topic": {}
    }
    
    # Process each embedded document
    for file_path in document_files:
        try:
            # Load the embedded document
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Get document topic
            topic = document.get("metadata", {}).get("topic", "unknown")
            
            # Add to current batch
            current_batch.append(document)
            
            # Update stats
            stats["total_documents"] += 1
            if topic not in stats["documents_by_topic"]:
                stats["documents_by_topic"][topic] = 0
            stats["documents_by_topic"][topic] += 1
            
            # Process batch if it reaches the batch size
            if len(current_batch) >= batch_size:
                doc_ids = vector_db.add_documents(current_batch)
                logger.info(f"Added batch of {len(current_batch)} documents to vector database")
                current_batch = []
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Process any remaining documents in the last batch
    if current_batch:
        doc_ids = vector_db.add_documents(current_batch)
        logger.info(f"Added final batch of {len(current_batch)} documents to vector database")
    
    # Save the vector database
    vector_db.save(output_dir)
    
    return stats


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build vector database
    stats = build_vector_db_from_embedded_docs()
    
    logger.info(f"Vector database build complete. Added {stats['total_documents']} documents.")
    for topic, count in stats["documents_by_topic"].items():
        logger.info(f"Topic '{topic}': {count} documents")
