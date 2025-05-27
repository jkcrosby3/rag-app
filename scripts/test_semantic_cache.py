#!/usr/bin/env python
"""Test script for the semantic cache for LLM responses.

This script tests the performance of the semantic cache for LLM responses
implemented in the RAG system, including semantic matching for similar queries.
"""
import sys
import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG system components
from src.rag_system import RAGSystem
from src.llm.semantic_cache import semantic_cache, get_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_queries() -> List[Dict[str, Any]]:
    """Generate test queries for the RAG system.
    
    Returns:
        List of test query dictionaries with query text and topics
    """
    # Define base queries
    base_queries = [
        {
            "query": "What were the key provisions of the Glass-Steagall Act?",
            "topics": ["glass_steagall"],
            "description": "Factual query about Glass-Steagall"
        },
        {
            "query": "How did the Glass-Steagall Act separate commercial and investment banking?",
            "topics": ["glass_steagall"],
            "description": "Mechanism query about Glass-Steagall"
        },
        {
            "query": "What was the purpose of the SEC during the Great Depression?",
            "topics": ["sec"],
            "description": "Purpose query about SEC"
        },
        {
            "query": "Compare the New Deal programs with Glass-Steagall regulations.",
            "topics": ["new_deal", "glass_steagall"],
            "description": "Comparative query across multiple topics"
        },
        {
            "query": "What were the major financial reforms after the Great Depression?",
            "topics": None,  # All topics
            "description": "Broad query across all topics"
        }
    ]
    
    # Generate semantic variations for each base query
    test_queries = []
    
    for base_query in base_queries:
        # Add the original query
        test_queries.append(base_query.copy())
        
        # Add semantic variations
        if "Glass-Steagall" in base_query["query"]:
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("Glass-Steagall", "Banking Act of 1933")
            variation["is_variation"] = True
            test_queries.append(variation)
            
        if "key provisions" in base_query["query"]:
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("key provisions", "main components")
            variation["is_variation"] = True
            test_queries.append(variation)
            
        if "purpose" in base_query["query"]:
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("purpose", "role")
            variation["is_variation"] = True
            test_queries.append(variation)
            
        if "Compare" in base_query["query"]:
            variation = base_query.copy()
            variation["query"] = "What are the similarities and differences between " + \
                               base_query["query"].split("Compare ")[1].rstrip(".")
            variation["is_variation"] = True
            test_queries.append(variation)
            
        if "major financial reforms" in base_query["query"]:
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("major financial reforms", 
                                                          "significant banking regulations")
            variation["is_variation"] = True
            test_queries.append(variation)
    
    return test_queries


def test_semantic_cache(rag_system: RAGSystem, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test the semantic cache performance.
    
    Args:
        rag_system: RAG system instance
        test_queries: List of test queries
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing semantic cache performance...")
    
    # Clear the cache before testing
    semantic_cache.clear()
    
    results = {
        "exact_hits": 0,
        "semantic_hits": 0,
        "misses": 0,
        "response_times": [],
        "cache_hit_times": [],
        "cache_miss_times": [],
        "query_results": []
    }
    
    # First pass: Run all base queries (non-variations)
    logger.info("First pass: Running base queries to populate cache...")
    for query_data in test_queries:
        if query_data.get("is_variation", False):
            continue
            
        query = query_data["query"]
        topics = query_data["topics"]
        
        start_time = time.time()
        response = rag_system.process_query(query, filter_topics=topics)
        elapsed_time = time.time() - start_time
        
        results["response_times"].append(elapsed_time)
        results["cache_miss_times"].append(elapsed_time)
        results["misses"] += 1
        
        results["query_results"].append({
            "query": query,
            "topics": topics,
            "response_time": elapsed_time,
            "cache_hit": False,
            "hit_type": "miss"
        })
        
        logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Cache miss)")
    
    # Get cache stats after first pass
    first_pass_stats = get_stats()
    
    # Second pass: Run exact same queries to test exact cache hits
    logger.info("Second pass: Testing exact cache hits...")
    for query_data in test_queries:
        if query_data.get("is_variation", False):
            continue
            
        query = query_data["query"]
        topics = query_data["topics"]
        
        start_time = time.time()
        response = rag_system.process_query(query, filter_topics=topics)
        elapsed_time = time.time() - start_time
        
        results["response_times"].append(elapsed_time)
        results["cache_hit_times"].append(elapsed_time)
        results["exact_hits"] += 1
        
        results["query_results"].append({
            "query": query,
            "topics": topics,
            "response_time": elapsed_time,
            "cache_hit": True,
            "hit_type": "exact"
        })
        
        logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Exact cache hit)")
    
    # Third pass: Run semantic variations to test semantic cache hits
    logger.info("Third pass: Testing semantic cache hits...")
    for query_data in test_queries:
        if not query_data.get("is_variation", False):
            continue
            
        query = query_data["query"]
        topics = query_data["topics"]
        
        start_time = time.time()
        response = rag_system.process_query(query, filter_topics=topics)
        elapsed_time = time.time() - start_time
        
        results["response_times"].append(elapsed_time)
        
        # Check if it was a semantic hit or miss
        cache_stats = get_stats()
        if cache_stats["semantic_hits"] > first_pass_stats["semantic_hits"]:
            results["semantic_hits"] += 1
            results["cache_hit_times"].append(elapsed_time)
            hit_type = "semantic"
            cache_hit = True
            logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Semantic cache hit)")
        else:
            results["misses"] += 1
            results["cache_miss_times"].append(elapsed_time)
            hit_type = "miss"
            cache_hit = False
            logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Cache miss)")
        
        results["query_results"].append({
            "query": query,
            "topics": topics,
            "response_time": elapsed_time,
            "cache_hit": cache_hit,
            "hit_type": hit_type
        })
        
        # Update first_pass_stats for the next iteration
        first_pass_stats = cache_stats
    
    # Calculate statistics
    total_queries = results["exact_hits"] + results["semantic_hits"] + results["misses"]
    results["total_queries"] = total_queries
    results["hit_rate"] = (results["exact_hits"] + results["semantic_hits"]) / total_queries if total_queries > 0 else 0
    results["semantic_hit_rate"] = results["semantic_hits"] / len([q for q in test_queries if q.get("is_variation", False)]) if len([q for q in test_queries if q.get("is_variation", False)]) > 0 else 0
    
    results["avg_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
    results["avg_cache_hit_time"] = statistics.mean(results["cache_hit_times"]) if results["cache_hit_times"] else 0
    results["avg_cache_miss_time"] = statistics.mean(results["cache_miss_times"]) if results["cache_miss_times"] else 0
    
    results["speedup"] = results["avg_cache_miss_time"] / results["avg_cache_hit_time"] if results["avg_cache_hit_time"] > 0 else 0
    
    # Get final cache stats
    results["cache_stats"] = get_stats()
    
    return results


def main():
    """Main function to run the semantic cache test."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY environment variable not set. Using mock mode.")
        os.environ["ANTHROPIC_API_KEY"] = "mock_key_for_testing"
    
    # Initialize the RAG system
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem(
        vector_db_type="faiss",
        use_quantized_embeddings=False
    )
    
    # Generate test queries
    test_queries = generate_test_queries()
    
    # Test semantic cache
    semantic_results = test_semantic_cache(rag_system, test_queries)
    
    # Save results
    results_dir = project_root / "data" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    semantic_results_file = results_dir / "semantic_cache_results.json"
    with open(semantic_results_file, 'w', encoding='utf-8') as f:
        # Convert non-serializable objects to strings
        serializable_results = json.dumps(semantic_results, default=str)
        json.dump(json.loads(serializable_results), f, indent=2)
        
    logger.info(f"Semantic cache test results saved to {semantic_results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SEMANTIC CACHE TEST SUMMARY")
    print("="*80)
    print(f"Total queries: {semantic_results['total_queries']}")
    print(f"Exact hits: {semantic_results['exact_hits']}")
    print(f"Semantic hits: {semantic_results['semantic_hits']}")
    print(f"Misses: {semantic_results['misses']}")
    print(f"Hit rate: {semantic_results['hit_rate']:.2f}")
    print(f"Semantic hit rate: {semantic_results['semantic_hit_rate']:.2f}")
    print(f"Average response time: {semantic_results['avg_response_time']:.2f}s")
    print(f"Average cache hit time: {semantic_results['avg_cache_hit_time']:.2f}s")
    print(f"Average cache miss time: {semantic_results['avg_cache_miss_time']:.2f}s")
    print(f"Speedup: {semantic_results['speedup']:.2f}x")
    print("\nCache Statistics:")
    for key, value in semantic_results["cache_stats"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
