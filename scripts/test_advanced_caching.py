#!/usr/bin/env python
"""Test script for advanced caching strategies in the RAG system.

This script evaluates the performance of the advanced caching strategies
implemented in the RAG system, including:
1. Semantic caching for LLM responses
2. Persistent disk caching for embeddings
3. Parallel processing for batch embedding generation

Usage:
    python test_advanced_caching.py [--clear-cache] [--test-semantic] [--test-embedding] [--test-parallel]
"""
import sys
import os
import time
import logging
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG system components
from src.rag_system import RAGSystem
from src.llm.semantic_cache import semantic_cache, get_stats as get_semantic_stats
from src.embeddings.model_cache import get_stats as get_embedding_stats
from src.embeddings.generator import EmbeddingGenerator

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
    first_pass_stats = get_semantic_stats()
    
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
        cache_stats = get_semantic_stats()
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
    results["cache_stats"] = get_semantic_stats()
    
    return results


def test_embedding_cache(rag_system: RAGSystem, num_texts: int = 100) -> Dict[str, Any]:
    """Test the embedding cache performance.
    
    Args:
        rag_system: RAG system instance
        num_texts: Number of test texts to generate
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing embedding cache performance with {num_texts} texts...")
    
    # Generate test texts
    test_texts = []
    for i in range(num_texts):
        # Generate random text of varying length
        length = random.randint(10, 100)
        test_text = f"Test text {i} with random words: " + " ".join(
            [f"word{random.randint(1, 1000)}" for _ in range(length)]
        )
        test_texts.append(test_text)
    
    # Get embedding generator
    embedding_generator = rag_system.embedding_generator
    
    results = {
        "memory_hits": 0,
        "disk_hits": 0,
        "misses": 0,
        "single_embedding_times": [],
        "batch_embedding_times": [],
        "memory_hit_times": [],
        "disk_hit_times": [],
        "miss_times": []
    }
    
    # First pass: Generate embeddings for all texts individually
    logger.info("First pass: Generating embeddings individually...")
    for text in test_texts:
        start_time = time.time()
        embedding = embedding_generator.generate_embedding(text)
        elapsed_time = time.time() - start_time
        
        results["single_embedding_times"].append(elapsed_time)
        results["miss_times"].append(elapsed_time)
        results["misses"] += 1
        
        logger.debug(f"Generated embedding in {elapsed_time:.4f}s (Cache miss)")
    
    # Get cache stats after first pass
    first_pass_stats = get_embedding_stats()
    
    # Second pass: Generate embeddings again to test memory cache
    logger.info("Second pass: Testing memory cache hits...")
    for text in test_texts:
        start_time = time.time()
        embedding = embedding_generator.generate_embedding(text)
        elapsed_time = time.time() - start_time
        
        results["single_embedding_times"].append(elapsed_time)
        results["memory_hit_times"].append(elapsed_time)
        results["memory_hits"] += 1
        
        logger.debug(f"Generated embedding in {elapsed_time:.4f}s (Memory cache hit)")
    
    # Test batch processing
    logger.info("Testing batch embedding generation...")
    
    # Split texts into batches
    batch_size = 20
    batches = [test_texts[i:i+batch_size] for i in range(0, len(test_texts), batch_size)]
    
    for batch in batches:
        start_time = time.time()
        embeddings = embedding_generator.generate_embeddings(batch)
        elapsed_time = time.time() - start_time
        
        results["batch_embedding_times"].append(elapsed_time)
        
        logger.debug(f"Generated {len(batch)} embeddings in batch in {elapsed_time:.4f}s")
    
    # Calculate statistics
    total_embeddings = results["memory_hits"] + results["disk_hits"] + results["misses"]
    results["total_embeddings"] = total_embeddings
    results["hit_rate"] = (results["memory_hits"] + results["disk_hits"]) / total_embeddings if total_embeddings > 0 else 0
    
    results["avg_single_time"] = statistics.mean(results["single_embedding_times"]) if results["single_embedding_times"] else 0
    results["avg_memory_hit_time"] = statistics.mean(results["memory_hit_times"]) if results["memory_hit_times"] else 0
    results["avg_disk_hit_time"] = statistics.mean(results["disk_hit_times"]) if results["disk_hit_times"] else 0
    results["avg_miss_time"] = statistics.mean(results["miss_times"]) if results["miss_times"] else 0
    
    # Calculate batch statistics
    if results["batch_embedding_times"]:
        total_batch_time = sum(results["batch_embedding_times"])
        total_batch_embeddings = sum(len(test_texts[i:i+batch_size]) for i in range(0, len(test_texts), batch_size))
        results["avg_time_per_embedding_in_batch"] = total_batch_time / total_batch_embeddings
    else:
        results["avg_time_per_embedding_in_batch"] = 0
    
    # Calculate speedups
    results["memory_hit_speedup"] = results["avg_miss_time"] / results["avg_memory_hit_time"] if results["avg_memory_hit_time"] > 0 else 0
    results["batch_speedup"] = results["avg_single_time"] / results["avg_time_per_embedding_in_batch"] if results["avg_time_per_embedding_in_batch"] > 0 else 0
    
    # Get final cache stats
    results["cache_stats"] = get_embedding_stats()
    
    return results


def test_parallel_processing(rag_system: RAGSystem, batch_sizes: List[int] = [10, 50, 100, 200, 500]) -> Dict[str, Any]:
    """Test parallel processing performance for different batch sizes.
    
    Args:
        rag_system: RAG system instance
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing parallel processing performance...")
    
    # Get embedding generator
    embedding_generator = rag_system.embedding_generator
    
    results = {
        "batch_sizes": batch_sizes,
        "sequential_times": [],
        "parallel_times": [],
        "speedups": []
    }
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Generate test texts
        test_texts = []
        for i in range(batch_size):
            # Generate random text of varying length
            length = random.randint(10, 100)
            test_text = f"Parallel test text {i} with random words: " + " ".join(
                [f"word{random.randint(1, 1000)}" for _ in range(length)]
            )
            test_texts.append(test_text)
        
        # Test sequential processing
        start_time = time.time()
        for text in test_texts:
            embedding = embedding_generator.generate_embedding(text)
        sequential_time = time.time() - start_time
        
        results["sequential_times"].append(sequential_time)
        
        logger.info(f"Sequential processing for {batch_size} texts: {sequential_time:.2f}s")
        
        # Test parallel processing
        start_time = time.time()
        embeddings = embedding_generator.generate_embeddings(test_texts)
        parallel_time = time.time() - start_time
        
        results["parallel_times"].append(parallel_time)
        
        logger.info(f"Parallel processing for {batch_size} texts: {parallel_time:.2f}s")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        results["speedups"].append(speedup)
        
        logger.info(f"Speedup: {speedup:.2f}x")
    
    return results


def main():
    """Main function to run the advanced caching tests."""
    parser = argparse.ArgumentParser(description="Test advanced caching strategies in the RAG system")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all caches before testing")
    parser.add_argument("--test-semantic", action="store_true", help="Test semantic caching")
    parser.add_argument("--test-embedding", action="store_true", help="Test embedding caching")
    parser.add_argument("--test-parallel", action="store_true", help="Test parallel processing")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not (args.test_semantic or args.test_embedding or args.test_parallel):
        args.all = True
    
    # Initialize the RAG system
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem(
        vector_db_type="faiss",
        use_quantized_embeddings=True,
        quantization_type="int8"
    )
    
    # Clear caches if requested
    if args.clear_cache:
        logger.info("Clearing all caches...")
        semantic_cache.clear()
        # Clear embedding cache through the RAG system
        rag_system.embedding_generator._load_model()  # Ensure model is loaded
        
    # Create results directory
    results_dir = project_root / "data" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Test semantic cache
    if args.test_semantic or args.all:
        logger.info("Running semantic cache tests...")
        test_queries = generate_test_queries()
        semantic_results = test_semantic_cache(rag_system, test_queries)
        all_results["semantic_cache"] = semantic_results
        
        # Save semantic cache results
        semantic_results_file = results_dir / "semantic_cache_results.json"
        with open(semantic_results_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_results, f, indent=2)
            
        logger.info(f"Semantic cache test results saved to {semantic_results_file}")
        
        # Print summary
        logger.info("Semantic Cache Test Summary:")
        logger.info(f"Total queries: {semantic_results['total_queries']}")
        logger.info(f"Exact hits: {semantic_results['exact_hits']}")
        logger.info(f"Semantic hits: {semantic_results['semantic_hits']}")
        logger.info(f"Misses: {semantic_results['misses']}")
        logger.info(f"Hit rate: {semantic_results['hit_rate']:.2f}")
        logger.info(f"Semantic hit rate: {semantic_results['semantic_hit_rate']:.2f}")
        logger.info(f"Average response time: {semantic_results['avg_response_time']:.2f}s")
        logger.info(f"Average cache hit time: {semantic_results['avg_cache_hit_time']:.2f}s")
        logger.info(f"Average cache miss time: {semantic_results['avg_cache_miss_time']:.2f}s")
        logger.info(f"Speedup: {semantic_results['speedup']:.2f}x")
    
    # Test embedding cache
    if args.test_embedding or args.all:
        logger.info("Running embedding cache tests...")
        embedding_results = test_embedding_cache(rag_system, num_texts=100)
        all_results["embedding_cache"] = embedding_results
        
        # Save embedding cache results
        embedding_results_file = results_dir / "embedding_cache_results.json"
        with open(embedding_results_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_results, f, indent=2)
            
        logger.info(f"Embedding cache test results saved to {embedding_results_file}")
        
        # Print summary
        logger.info("Embedding Cache Test Summary:")
        logger.info(f"Total embeddings: {embedding_results['total_embeddings']}")
        logger.info(f"Memory hits: {embedding_results['memory_hits']}")
        logger.info(f"Disk hits: {embedding_results['disk_hits']}")
        logger.info(f"Misses: {embedding_results['misses']}")
        logger.info(f"Hit rate: {embedding_results['hit_rate']:.2f}")
        logger.info(f"Average single embedding time: {embedding_results['avg_single_time']:.4f}s")
        logger.info(f"Average memory hit time: {embedding_results['avg_memory_hit_time']:.4f}s")
        logger.info(f"Average time per embedding in batch: {embedding_results['avg_time_per_embedding_in_batch']:.4f}s")
        logger.info(f"Memory hit speedup: {embedding_results['memory_hit_speedup']:.2f}x")
        logger.info(f"Batch processing speedup: {embedding_results['batch_speedup']:.2f}x")
    
    # Test parallel processing
    if args.test_parallel or args.all:
        logger.info("Running parallel processing tests...")
        parallel_results = test_parallel_processing(rag_system)
        all_results["parallel_processing"] = parallel_results
        
        # Save parallel processing results
        parallel_results_file = results_dir / "parallel_processing_results.json"
        with open(parallel_results_file, 'w', encoding='utf-8') as f:
            json.dump(parallel_results, f, indent=2)
            
        logger.info(f"Parallel processing test results saved to {parallel_results_file}")
        
        # Print summary
        logger.info("Parallel Processing Test Summary:")
        for i, batch_size in enumerate(parallel_results["batch_sizes"]):
            logger.info(f"Batch size {batch_size}:")
            logger.info(f"  Sequential time: {parallel_results['sequential_times'][i]:.2f}s")
            logger.info(f"  Parallel time: {parallel_results['parallel_times'][i]:.2f}s")
            logger.info(f"  Speedup: {parallel_results['speedups'][i]:.2f}x")
    
    # Save all results
    all_results_file = results_dir / "advanced_caching_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"All test results saved to {all_results_file}")
    
    logger.info("Advanced caching tests completed.")


if __name__ == "__main__":
    main()
