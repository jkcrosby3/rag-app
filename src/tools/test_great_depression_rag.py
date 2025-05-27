"""
Test script for the RAG system with Great Depression documents.

This script tests the RAG system with queries related to the Great Depression.
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

# Import RAG system
try:
    from dotenv import load_dotenv
    from src.rag_system import RAGSystem
    from src.llm.semantic_cache import get_stats as get_semantic_cache_stats
except ImportError as e:
    logger.error(f"Error importing RAG system: {str(e)}")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()


def test_rag_system(
    queries: List[str],
    vector_db_type: str = "faiss",
    vector_db_path: str = "data/vector_db",
    semantic_similarity_threshold: float = 0.75,
    k: int = 5
) -> Dict[str, Any]:
    """
    Test the RAG system with a list of queries.
    
    Args:
        queries: List of queries to test
        vector_db_type: Type of vector database to use
        vector_db_path: Path to the vector database
        semantic_cache_threshold: Threshold for semantic cache
        k: Number of documents to retrieve
        
    Returns:
        Dict with test results
    """
    # Initialize RAG system
    rag_system = RAGSystem(
        vector_db_type=vector_db_type,
        vector_db_path=vector_db_path,
        semantic_similarity_threshold=semantic_similarity_threshold
    )
    
    results = {
        "queries": [],
        "total_time": 0,
        "avg_time": 0,
        "cache_hits": 0,
        "cache_misses": 0
    }
    
    # Test each query
    total_time = 0
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        # Measure response time
        start_time = time.time()
        
        # Generate response
        result = rag_system.process_query(query, top_k=k)
        
        # Calculate time
        query_time = time.time() - start_time
        total_time += query_time
        
        # Get cache stats
        cache_stats = get_semantic_cache_stats()
        
        # Store results
        query_result = {
            "query": query,
            "response": result["response"],
            "time": query_time,
            "retrieved_docs": result["retrieved_documents"],
            "cache_hit": result.get("cache_info", {}).get("semantic_cache", {}).get("hit", False)
        }
        
        # Update cache stats
        if query_result["cache_hit"]:
            results["cache_hits"] += 1
        else:
            results["cache_misses"] += 1
        
        results["queries"].append(query_result)
        
        logger.info(f"Response time: {query_time:.2f} seconds")
        logger.info(f"Cache hit: {query_result['cache_hit']}")
        logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
        logger.info(f"Response: {result['response'][:100]}...")
        logger.info("-" * 50)
    
    # Calculate averages
    results["total_time"] = total_time
    results["avg_time"] = total_time / len(queries) if queries else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test the RAG system with Great Depression queries")
    parser.add_argument("--vector-db-type", type=str, default="faiss", choices=["faiss", "elasticsearch"], 
                        help="Vector database type")
    parser.add_argument("--vector-db-path", type=str, default="data/vector_db", 
                        help="Path to the vector database")
    parser.add_argument("--semantic-similarity-threshold", type=float, default=0.75, 
                        help="Threshold for semantic cache")
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
    results = test_rag_system(
        queries=test_queries,
        vector_db_type=args.vector_db_type,
        vector_db_path=args.vector_db_path,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        k=args.k
    )
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    logger.info(f"Total queries: {len(results['queries'])}")
    logger.info(f"Total time: {results['total_time']:.2f} seconds")
    logger.info(f"Average time per query: {results['avg_time']:.2f} seconds")
    logger.info(f"Cache hits: {results['cache_hits']}")
    logger.info(f"Cache misses: {results['cache_misses']}")
    logger.info(f"Cache hit rate: {results['cache_hits'] / len(results['queries']) * 100:.2f}%")


if __name__ == "__main__":
    main()
