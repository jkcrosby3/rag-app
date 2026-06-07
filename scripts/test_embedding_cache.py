#!/usr/bin/env python
"""Test script for the advanced embedding cache.

This script tests the performance of the advanced embedding cache
implemented in the RAG system, including persistent disk caching
and memory caching.
"""
import sys
import os
import time
import logging
import random
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the embedding generator and cache
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.model_cache import get_stats, get_cache_for_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_texts(num_texts: int = 100) -> List[str]:
    """Generate test texts for embedding.
    
    Args:
        num_texts: Number of test texts to generate
        
    Returns:
        List of test texts
    """
    test_texts = []
    for i in range(num_texts):
        # Generate random text of varying length
        length = random.randint(10, 100)
        test_text = f"Test text {i} with random words: " + " ".join(
            [f"word{random.randint(1, 1000)}" for _ in range(length)]
        )
        test_texts.append(test_text)
    
    return test_texts


def test_embedding_cache(num_texts: int = 100) -> Dict[str, Any]:
    """Test the embedding cache performance.
    
    Args:
        num_texts: Number of test texts to generate
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing embedding cache performance with {num_texts} texts...")
    
    # Generate test texts
    test_texts = generate_test_texts(num_texts)
    
    # Get embedding generator
    embedding_generator = EmbeddingGenerator()
    
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
    first_pass_stats = get_stats()
    logger.info(f"First pass complete. Cache stats: {first_pass_stats}")
    
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
    results["cache_stats"] = get_stats()
    
    return results


def main():
    """Main function to run the embedding cache test."""
    # Test embedding cache
    embedding_results = test_embedding_cache(num_texts=50)
    
    # Print summary
    print("\n" + "="*80)
    print("EMBEDDING CACHE TEST SUMMARY")
    print("="*80)
    print(f"Total embeddings: {embedding_results['total_embeddings']}")
    print(f"Memory hits: {embedding_results['memory_hits']}")
    print(f"Disk hits: {embedding_results['disk_hits']}")
    print(f"Misses: {embedding_results['misses']}")
    print(f"Hit rate: {embedding_results['hit_rate']:.2f}")
    print(f"Average single embedding time: {embedding_results['avg_single_time']:.4f}s")
    print(f"Average memory hit time: {embedding_results['avg_memory_hit_time']:.4f}s")
    print(f"Average time per embedding in batch: {embedding_results['avg_time_per_embedding_in_batch']:.4f}s")
    print(f"Memory hit speedup: {embedding_results['memory_hit_speedup']:.2f}x")
    print(f"Batch processing speedup: {embedding_results['batch_speedup']:.2f}x")
    print("\nCache Statistics:")
    for key, value in embedding_results["cache_stats"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
