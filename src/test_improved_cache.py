"""
Test script to evaluate the performance of the improved semantic cache and quantized embedding model.

This script runs a series of queries to test the performance of the RAG system with:
1. Standard embedding model vs. quantized embedding model
2. Different semantic cache similarity thresholds
3. Cache hit rates for similar queries
"""
import argparse
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from statistics import mean, median, stdev

from dotenv import load_dotenv

import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test queries with expected similar queries
TEST_QUERIES = [
    {
        "original": "What were the key provisions of the Glass-Steagall Act?",
        "similar": [
            "What were the main components of the Glass-Steagall Banking Act?",
            "What did the Glass-Steagall Act do?",
            "Explain the major provisions in the Banking Act of 1933."
        ],
        "topics": ["glass_steagall"]
    },
    {
        "original": "Explain the impact of the repeal of Glass-Steagall on the 2008 financial crisis",
        "similar": [
            "How did the Gramm-Leach-Bliley Act contribute to the 2008 financial crisis?",
            "What role did the Glass-Steagall repeal play in the Great Recession?",
            "Did removing the separation between commercial and investment banking lead to the financial crisis?"
        ],
        "topics": ["glass_steagall"]
    }
]

def run_test(
    use_quantized: bool = False,
    semantic_similarity_threshold: float = 0.75,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run performance tests on the RAG system.
    
    Args:
        use_quantized: Whether to use the quantized embedding model
        similarity_threshold: Similarity threshold for the semantic cache
        output_file: Path to save test results
        
    Returns:
        Dict with test results
    """
    # Initialize RAG system
    start_time = time.time()
    rag_system = RAGSystem(
        vector_db_type="faiss",
        vector_db_path="D:\\Development\\rag-app\\data\\vector_db",
        semantic_similarity_threshold=semantic_similarity_threshold,
        use_quantized_embeddings=use_quantized,
        quantization_type="int8" if use_quantized else None
    )
    init_time = time.time() - start_time
    logger.info(f"RAG system initialized in {init_time:.2f} seconds")
    
    results = {
        "system_config": {
            "use_quantized": use_quantized,
            "semantic_similarity_threshold": semantic_similarity_threshold,
            "quantization_type": "int8" if use_quantized else None
        },
        "initialization_time": init_time,
        "query_results": [],
        "summary": {}
    }
    
    # Run original queries first
    for test_case in TEST_QUERIES:
        original_query = test_case["original"]
        topics = test_case["topics"]
        
        logger.info(f"Processing original query: {original_query}")
        
        # Process original query
        start_time = time.time()
        result = rag_system.process_query(
            query=original_query,
            top_k=3,
            filter_topics=topics,
            return_cache_stats=True
        )
        query_time = time.time() - start_time
        
        # Extract performance metrics
        metrics = {
            "query": original_query,
            "is_original": True,
            "processing_time": query_time,
            "embedding_time": result.get("embedding_time", 0),
            "retrieval_time": result.get("retrieval_time", 0),
            "generation_time": result.get("generation_time", 0),
            "cache_hit": result.get("cache_hit", False),
            "semantic_cache_hit": result.get("semantic_cache_hit", False),
            "semantic_cache_hit_rate": result.get("cache_stats", {}).get("semantic_cache", {}).get("hit_rate", 0),
            "embedding_cache_hit_rate": result.get("cache_stats", {}).get("embedding_cache", {}).get("hit_rate", 0),
            "retrieved_documents_count": len(result.get("retrieved_documents", []))
        }
        
        results["query_results"].append(metrics)
        
        # Wait a bit to avoid rate limiting
        time.sleep(1)
    
    # Now run similar queries to test cache hits
    for test_case in TEST_QUERIES:
        original_query = test_case["original"]
        similar_queries = test_case["similar"]
        topics = test_case["topics"]
        
        for similar_query in similar_queries:
            logger.info(f"Processing similar query: {similar_query}")
            
            # Process similar query
            start_time = time.time()
            result = rag_system.process_query(
                query=similar_query,
                top_k=3,
                filter_topics=topics,
                return_cache_stats=True
            )
            query_time = time.time() - start_time
            
            # Extract performance metrics
            metrics = {
                "query": similar_query,
                "similar_to": original_query,
                "is_original": False,
                "processing_time": query_time,
                "embedding_time": result.get("embedding_time", 0),
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "cache_hit": result.get("cache_hit", False),
                "semantic_cache_hit": result.get("semantic_cache_hit", False),
                "semantic_cache_hit_rate": result.get("cache_stats", {}).get("semantic_cache", {}).get("hit_rate", 0),
                "embedding_cache_hit_rate": result.get("cache_stats", {}).get("embedding_cache", {}).get("hit_rate", 0),
                "retrieved_documents_count": len(result.get("retrieved_documents", []))
            }
            
            results["query_results"].append(metrics)
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
    
    # Calculate summary statistics
    all_times = [r["processing_time"] for r in results["query_results"]]
    original_times = [r["processing_time"] for r in results["query_results"] if r["is_original"]]
    similar_times = [r["processing_time"] for r in results["query_results"] if not r["is_original"]]
    
    cache_hits = sum(1 for r in results["query_results"] if r["cache_hit"])
    semantic_cache_hits = sum(1 for r in results["query_results"] if r["semantic_cache_hit"])
    
    results["summary"] = {
        "total_queries": len(results["query_results"]),
        "original_queries": len(original_times),
        "similar_queries": len(similar_times),
        "cache_hits": cache_hits,
        "semantic_cache_hits": semantic_cache_hits,
        "cache_hit_rate": cache_hits / len(results["query_results"]) if results["query_results"] else 0,
        "semantic_cache_hit_rate": semantic_cache_hits / len(similar_times) if similar_times else 0,
        "avg_processing_time": mean(all_times) if all_times else 0,
        "median_processing_time": median(all_times) if all_times else 0,
        "std_processing_time": stdev(all_times) if len(all_times) > 1 else 0,
        "avg_original_query_time": mean(original_times) if original_times else 0,
        "avg_similar_query_time": mean(similar_times) if similar_times else 0,
        "time_improvement_percentage": (
            (mean(original_times) - mean(similar_times)) / mean(original_times) * 100
            if original_times and similar_times and mean(original_times) > 0 else 0
        )
    }
    
    # Log summary
    logger.info(f"Test completed with {len(results['query_results'])} queries")
    logger.info(f"Cache hit rate: {results['summary']['cache_hit_rate']:.2f}")
    logger.info(f"Semantic cache hit rate: {results['summary']['semantic_cache_hit_rate']:.2f}")
    logger.info(f"Average processing time: {results['summary']['avg_processing_time']:.2f} seconds")
    logger.info(f"Time improvement for similar queries: {results['summary']['time_improvement_percentage']:.2f}%")
    
    # Save results to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test the RAG system with improved caching")
    parser.add_argument("--quantized", action="store_true", help="Use quantized embedding model")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for semantic cache")
    parser.add_argument("--output", type=str, help="Path to save test results")
    
    args = parser.parse_args()
    
    # Run the test
    results = run_test(
        use_quantized=args.quantized,
        semantic_similarity_threshold=args.threshold,
        output_file=args.output
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Configuration: {'Quantized' if args.quantized else 'Standard'} model, Semantic Similarity Threshold: {args.threshold}")
    print(f"Total queries: {results['summary']['total_queries']}")
    print(f"Cache hit rate: {results['summary']['cache_hit_rate']:.2f}")
    print(f"Semantic cache hit rate: {results['summary']['semantic_cache_hit_rate']:.2f}")
    print(f"Average processing time: {results['summary']['avg_processing_time']:.2f} seconds")
    print(f"Average time for original queries: {results['summary']['avg_original_query_time']:.2f} seconds")
    print(f"Average time for similar queries: {results['summary']['avg_similar_query_time']:.2f} seconds")
    print(f"Time improvement: {results['summary']['time_improvement_percentage']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
