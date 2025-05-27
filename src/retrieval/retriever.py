"""Document retriever for the RAG system.

This module provides functionality to retrieve relevant documents
based on query similarity and optional topic filtering.
"""
import logging
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from src.embeddings.generator import EmbeddingGenerator
from src.vector_db.faiss_db import FAISSVectorDB

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant documents based on query similarity."""

    def __init__(
        self,
        vector_db_path: Union[str, Path],
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the Retriever.

        Args:
            vector_db_path: Path to the vector database
            embedding_model_name: Name of the embedding model to use
        """
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
        
        # Load vector database
        self.vector_db = FAISSVectorDB(index_path=vector_db_path)
        
        logger.info(f"Initialized Retriever with {self.vector_db.get_document_count()} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_topics: Optional list of topics to filter results by
            
        Returns:
            List of document dicts with similarity scores
        """
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit embedding generation task
            embedding_future = executor.submit(
                self.embedding_generator.generate_embedding, query
            )
            
            # Wait for embedding generation to complete
            query_embedding = embedding_future.result()
            
            # Search vector database (this is I/O bound, so can be parallelized)
            search_future = executor.submit(
                self.vector_db.search, query_embedding, top_k * 2  # Get more results for filtering
            )
            
            # Get search results
            results = search_future.result()
        
        # Filter by topic if specified
        if filter_topics:
            # Use list comprehension for faster filtering
            filtered_results = [
                result for result in results
                if result.get("metadata", {}).get("topic") in filter_topics
            ]
            results = filtered_results
        
        # Limit to top_k results
        results = results[:top_k]
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time:.2f} seconds")
        
        return results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector database.
        
        Returns:
            Number of documents
        """
        return self.vector_db.get_document_count()


def search_documents(
    query: str,
    vector_db_path: Union[str, Path] = "D:\\Development\\rag-app\\data\\vector_db",
    top_k: int = 5,
    filter_topics: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Search for relevant documents based on query similarity.
    
    Args:
        query: Query text
        vector_db_path: Path to the vector database
        top_k: Number of results to return
        filter_topics: Optional list of topics to filter results by
        
    Returns:
        List of document dicts with similarity scores
    """
    # Initialize retriever
    retriever = Retriever(vector_db_path=vector_db_path)
    
    # Retrieve relevant documents
    results = retriever.retrieve(query, top_k=top_k, filter_topics=filter_topics)
    
    return results


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if query was provided
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query> [top_k] [filter_topics]")
        sys.exit(1)
    
    # Parse arguments
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    filter_topics = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    
    # Search documents
    results = search_documents(query, top_k=top_k, filter_topics=filter_topics)
    
    # Print results
    print(f"Top {len(results)} results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Similarity: {result['similarity']:.4f}) ---")
        print(f"Topic: {result.get('metadata', {}).get('topic', 'unknown')}")
        print(f"Source: {result.get('metadata', {}).get('file_name', 'unknown')}")
        print(f"Text: {result.get('text', '')[:200]}...")
