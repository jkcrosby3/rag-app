"""
Test script for the retrieval component of the RAG system with Great Depression documents.

This script tests just the retrieval part of the RAG system with queries related to the Great Depression,
without requiring an API key for the LLM.
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import RAG system components
try:
    from src.embeddings.quantized_generator import QuantizedEmbeddingGenerator
    from src.vector_db.faiss_db import FAISSVectorDB
except ImportError as e:
    logger.error(f"Error importing RAG system components: {str(e)}")
    sys.exit(1)


def test_document_retrieval(
    queries: List[str],
    vector_db_path: str = "data/vector_db",
    k: int = 5
) -> Dict[str, Any]:
    """
    Test the document retrieval component with a list of queries.
    
    Args:
        queries: List of queries to test
        vector_db_path: Path to the vector database
        k: Number of documents to retrieve
        
    Returns:
        Dict with test results
    """
    # Initialize embedding generator
    embedding_generator = QuantizedEmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        quantization_type="int8"
    )
    
    # Initialize vector database
    vector_db = FAISSVectorDB(index_path=vector_db_path)
    logger.info(f"Loaded FAISS vector database with {vector_db.get_document_count()} documents")
    
    results = {
        "queries": [],
        "total_time": 0,
        "avg_time": 0,
        "total_docs_retrieved": 0
    }
    
    # Test each query
    total_time = 0
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        # Measure retrieval time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Search for relevant documents
        retrieved_docs = vector_db.search(query_embedding, k=k)
        
        # Calculate time
        query_time = time.time() - start_time
        total_time += query_time
        
        # Store results
        query_result = {
            "query": query,
            "time": query_time,
            "retrieved_docs": retrieved_docs
        }
        
        results["queries"].append(query_result)
        results["total_docs_retrieved"] += len(retrieved_docs)
        
        logger.info(f"Retrieval time: {query_time:.2f} seconds")
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Display top document content
        if retrieved_docs:
            top_doc = retrieved_docs[0]
            similarity = top_doc.get('similarity', 0)
            topic = top_doc.get('metadata', {}).get('topic', 'unknown')
            logger.info(f"Top document (similarity: {similarity:.4f}, topic: {topic}):")
            logger.info(f"Content: {top_doc.get('text', '')[:150]}...")
        
        logger.info("-" * 50)
    
    # Calculate averages
    results["total_time"] = total_time
    results["avg_time"] = total_time / len(queries) if queries else 0
    results["avg_docs_per_query"] = results["total_docs_retrieved"] / len(queries) if queries else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test the retrieval component with Great Depression queries")
    parser.add_argument("--vector-db-path", type=str, default="data/vector_db", 
                        help="Path to the vector database")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Define test queries
    test_queries = [
        "What were the main causes of the Great Depression?",
        "How did the stock market crash of 1929 contribute to the Great Depression?",
        "What role did the Federal Reserve play during the Great Depression?",
        "What were the effects of the Great Depression on banking?",
        "How did President Hoover respond to the Great Depression?",
        "What was the New Deal and how did it address the Great Depression?",
        "How did the Great Depression affect international trade?",
        "What ended the Great Depression?",
        "How did monetary policy change after the Great Depression?",
        "What lessons can we learn from the Great Depression for modern economic policy?"
    ]
    
    # Run tests
    results = test_document_retrieval(
        queries=test_queries,
        vector_db_path=args.vector_db_path,
        k=args.k
    )
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    logger.info(f"Total queries: {len(results['queries'])}")
    logger.info(f"Total time: {results['total_time']:.2f} seconds")
    logger.info(f"Average time per query: {results['avg_time']:.2f} seconds")
    logger.info(f"Average documents per query: {results['avg_docs_per_query']:.2f}")
    logger.info(f"Total documents retrieved: {results['total_docs_retrieved']}")


if __name__ == "__main__":
    main()
