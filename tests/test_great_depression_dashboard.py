"""
Test script for the RAG system with Great Depression documents that saves results for the dashboard.

This script tests the RAG system with queries related to the Great Depression
and saves the results for visualization in the performance dashboard.
"""
import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
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
    # Import the RAG system components
    from src.rag_system import RAGSystem
    # Import cache stats functions if available
    try:
        from src.llm.semantic_cache import get_stats as get_semantic_cache_stats
    except ImportError:
        logger.warning("Could not import semantic cache stats")
        def get_semantic_cache_stats():
            return {"hits": 0, "misses": 0}
    
    try:
        from src.embeddings.model_cache import get_stats as get_embedding_cache_stats
    except ImportError:
        logger.warning("Could not import embedding cache stats")
        def get_embedding_cache_stats():
            return {"hits": 0, "misses": 0}
            
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
    k: int = 5,
    use_quantized: bool = True,
    results_dir: str = "data/test_results"
) -> Dict[str, Any]:
    """
    Test the RAG system with a list of queries and save results for the dashboard.
    
    Args:
        queries: List of queries to test
        vector_db_type: Type of vector database to use
        vector_db_path: Path to the vector database
        semantic_similarity_threshold: Threshold for semantic cache
        k: Number of documents to retrieve
        use_quantized: Whether to use quantized embeddings
        results_dir: Directory to save test results
        
    Returns:
        Dict with test results
    """
    # Initialize RAG system
    rag_system = RAGSystem(
        vector_db_type=vector_db_type,
        vector_db_path=vector_db_path,
        semantic_similarity_threshold=semantic_similarity_threshold,
        use_quantized_embeddings=use_quantized
    )
    
    # Prepare results structure
    test_results = {
        "system_config": {
            "vector_db_type": vector_db_type,
            "vector_db_path": vector_db_path,
            "semantic_similarity_threshold": semantic_similarity_threshold,
            "k": k,
            "use_quantized": use_quantized
        },
        "queries": [],
        "summary": {
            "total_queries": len(queries),
            "cache_hits": 0,
            "semantic_cache_hits": 0,
            "avg_processing_time": 0.0,
            "avg_original_query_time": 0.0,
            "avg_similar_query_time": 0.0,
            "time_improvement_percentage": 0.0,
            "cache_hit_rate": 0.0,
            "semantic_cache_hit_rate": 0.0
        }
    }
    
    # Test each query
    total_time = 0
    original_query_times = []
    similar_query_times = []
    
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
        embedding_stats = get_embedding_cache_stats()
        
        # Determine if this was a cache hit
        is_cache_hit = result.get("cache_info", {}).get("semantic_cache", {}).get("hit", False)
        is_semantic_hit = result.get("cache_info", {}).get("semantic_cache", {}).get("semantic_hit", False)
        
        # Update cache hit counts
        if is_cache_hit:
            test_results["summary"]["cache_hits"] += 1
        if is_semantic_hit:
            test_results["summary"]["semantic_cache_hits"] += 1
        
        # Store query result
        query_result = {
            "query": query,
            "time": query_time,
            "cache_hit": is_cache_hit,
            "semantic_hit": is_semantic_hit,
            "retrieved_docs_count": len(result["retrieved_documents"]),
            "embedding_time": result.get("embedding_time", 0.0),
            "retrieval_time": result.get("retrieval_time", 0.0),
            "generation_time": result.get("generation_time", 0.0)
        }
        
        # Track times for original vs similar queries
        if i < len(queries) // 2:  # First half are original queries
            original_query_times.append(query_time)
        else:  # Second half are similar queries
            similar_query_times.append(query_time)
        
        test_results["queries"].append(query_result)
        
        logger.info(f"Response time: {query_time:.2f} seconds")
        logger.info(f"Cache hit: {is_cache_hit}")
        logger.info(f"Semantic hit: {is_semantic_hit}")
        logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
        logger.info(f"Response: {result['response'][:100]}...")
        logger.info("-" * 50)
    
    # Calculate summary statistics
    test_results["summary"]["avg_processing_time"] = total_time / len(queries) if queries else 0
    
    if original_query_times:
        test_results["summary"]["avg_original_query_time"] = sum(original_query_times) / len(original_query_times)
    
    if similar_query_times:
        test_results["summary"]["avg_similar_query_time"] = sum(similar_query_times) / len(similar_query_times)
        
        # Calculate time improvement
        if test_results["summary"]["avg_original_query_time"] > 0:
            time_improvement = (
                (test_results["summary"]["avg_original_query_time"] - test_results["summary"]["avg_similar_query_time"]) / 
                test_results["summary"]["avg_original_query_time"] * 100
            )
            test_results["summary"]["time_improvement_percentage"] = max(0, time_improvement)
    
    # Calculate hit rates
    test_results["summary"]["cache_hit_rate"] = (
        test_results["summary"]["cache_hits"] / len(queries) if queries else 0
    )
    test_results["summary"]["semantic_cache_hit_rate"] = (
        test_results["summary"]["semantic_cache_hits"] / len(queries) if queries else 0
    )
    
    # Save test results
    save_test_results(test_results, results_dir)
    
    return test_results


def save_test_results(results: Dict[str, Any], results_dir: str) -> str:
    """
    Save test results to a JSON file.
    
    Args:
        results: Test results dictionary
        results_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    # Create results directory if it doesn't exist
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp and configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = results["system_config"]
    threshold = config["semantic_similarity_threshold"]
    model_type = "quantized" if config["use_quantized"] else "standard"
    
    filename = f"great_depression_{model_type}_threshold_{threshold:.2f}_{timestamp}.json"
    file_path = results_path / filename
    
    # Save results
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {file_path}")
    return str(file_path)


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
    parser.add_argument("--use-quantized", action="store_true", default=True,
                        help="Use quantized embeddings")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                        help="Directory to save test results")
    
    args = parser.parse_args()
    
    # Define test queries
    # First half are original queries, second half are similar queries
    test_queries = [
        # Original queries
        "What were the main causes of the Great Depression?",
        "How did the stock market crash of 1929 contribute to the Great Depression?",
        "What role did the Federal Reserve play during the Great Depression?",
        "What were the effects of the Great Depression on banking?",
        "How did President Hoover respond to the Great Depression?",
        
        # Similar queries (for testing semantic cache)
        "What factors led to the Great Depression?",
        "How did the 1929 market crash affect the Great Depression?",
        "What actions did the Fed take during the Great Depression?",
        "How did the Great Depression impact the banking system?",
        "What policies did Herbert Hoover implement during the Great Depression?"
    ]
    
    # Run tests
    results = test_rag_system(
        queries=test_queries,
        vector_db_type=args.vector_db_type,
        vector_db_path=args.vector_db_path,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        k=args.k,
        use_quantized=args.use_quantized,
        results_dir=args.results_dir
    )
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    logger.info(f"Total queries: {results['summary']['total_queries']}")
    logger.info(f"Average processing time: {results['summary']['avg_processing_time']:.2f} seconds")
    logger.info(f"Cache hits: {results['summary']['cache_hits']}")
    logger.info(f"Semantic cache hits: {results['summary']['semantic_cache_hits']}")
    logger.info(f"Cache hit rate: {results['summary']['cache_hit_rate'] * 100:.2f}%")
    logger.info(f"Semantic cache hit rate: {results['summary']['semantic_cache_hit_rate'] * 100:.2f}%")
    logger.info(f"Time improvement: {results['summary']['time_improvement_percentage']:.2f}%")
    
    # Suggest running the dashboard
    logger.info("\nTo view the performance dashboard, run:")
    logger.info("python src/dashboard/performance_monitor.py")


if __name__ == "__main__":
    main()
