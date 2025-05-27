#!/usr/bin/env python
"""Test script for combined embedding and semantic caching strategies.

This script evaluates the performance of both the embedding cache and semantic cache
together to provide a comprehensive view of the advanced caching strategies implemented
in the RAG system.
"""
import sys
import os
import time
import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG system components
from src.embeddings.model_cache import get_stats as get_embedding_stats, get_cache_for_model
from src.llm.semantic_cache import get_stats as get_semantic_stats
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.quantized_generator import QuantizedEmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_texts() -> List[Dict[str, Any]]:
    """Generate test texts for embedding generation.
    
    Returns:
        List of test text dictionaries
    """
    # Define base texts
    base_texts = [
        {
            "text": "The Glass-Steagall Act was a landmark banking legislation that separated commercial and investment banking activities.",
            "category": "glass_steagall",
            "description": "Basic description of Glass-Steagall"
        },
        {
            "text": "The Securities and Exchange Commission (SEC) was created to regulate the securities industry and protect investors.",
            "category": "sec",
            "description": "Basic description of SEC"
        },
        {
            "text": "The New Deal was a series of programs and projects instituted by President Franklin D. Roosevelt during the Great Depression.",
            "category": "new_deal",
            "description": "Basic description of New Deal"
        },
        {
            "text": "The Federal Deposit Insurance Corporation (FDIC) was established to provide deposit insurance to bank depositors.",
            "category": "banking",
            "description": "Basic description of FDIC"
        },
        {
            "text": "The stock market crash of 1929 was a major American stock market crash that occurred in the fall of 1929.",
            "category": "market",
            "description": "Basic description of 1929 crash"
        }
    ]
    
    # Generate variations for each base text
    test_texts = []
    
    for base_text in base_texts:
        # Add the original text
        test_texts.append(base_text.copy())
        
        # Add variations with minor changes
        variation1 = base_text.copy()
        variation1["text"] = base_text["text"].replace("was", "is considered")
        variation1["is_variation"] = True
        test_texts.append(variation1)
        
        variation2 = base_text.copy()
        variation2["text"] = base_text["text"].replace(".", " in the United States.")
        variation2["is_variation"] = True
        test_texts.append(variation2)
    
    return test_texts


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


def test_embedding_cache(embedding_generator, test_texts: List[Dict[str, Any]], quantized: bool = False) -> Dict[str, Any]:
    """Test the embedding cache performance.
    
    Args:
        embedding_generator: Embedding generator instance
        test_texts: List of test texts
        quantized: Whether the embedding generator is quantized
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing {'quantized' if quantized else 'standard'} embedding cache performance...")
    
    # Clear the embedding cache
    cache = get_cache_for_model()
    cache.clear()
    
    results = {
        "exact_hits": 0,
        "misses": 0,
        "generation_times": [],
        "cache_hit_times": [],
        "cache_miss_times": [],
        "text_results": []
    }
    
    # First pass: Generate embeddings for all base texts
    logger.info("First pass: Generating embeddings for base texts...")
    for text_data in test_texts:
        if text_data.get("is_variation", False):
            continue
            
        text = text_data["text"]
        
        start_time = time.time()
        embedding = embedding_generator.generate_embedding(text)
        elapsed_time = time.time() - start_time
        
        results["generation_times"].append(elapsed_time)
        results["cache_miss_times"].append(elapsed_time)
        results["misses"] += 1
        
        results["text_results"].append({
            "text": text,
            "generation_time": elapsed_time,
            "cache_hit": False
        })
        
        logger.info(f"Text: '{text[:30]}...' - Time: {elapsed_time:.4f}s (Cache miss)")
    
    # Second pass: Generate embeddings for the same texts to test cache hits
    logger.info("Second pass: Testing cache hits for base texts...")
    for text_data in test_texts:
        if text_data.get("is_variation", False):
            continue
            
        text = text_data["text"]
        
        start_time = time.time()
        embedding = embedding_generator.generate_embedding(text)
        elapsed_time = time.time() - start_time
        
        results["generation_times"].append(elapsed_time)
        results["cache_hit_times"].append(elapsed_time)
        results["exact_hits"] += 1
        
        results["text_results"].append({
            "text": text,
            "generation_time": elapsed_time,
            "cache_hit": True
        })
        
        logger.info(f"Text: '{text[:30]}...' - Time: {elapsed_time:.4f}s (Cache hit)")
    
    # Third pass: Generate embeddings for variations to test cache behavior
    logger.info("Third pass: Testing cache behavior for text variations...")
    for text_data in test_texts:
        if not text_data.get("is_variation", False):
            continue
            
        text = text_data["text"]
        
        start_time = time.time()
        embedding = embedding_generator.generate_embedding(text)
        elapsed_time = time.time() - start_time
        
        results["generation_times"].append(elapsed_time)
        
        # Check if it was a hit or miss (all variations should be misses)
        cache_stats = get_embedding_stats()
        if cache_stats["hits"] > results["exact_hits"]:
            results["exact_hits"] += 1
            results["cache_hit_times"].append(elapsed_time)
            cache_hit = True
            logger.info(f"Text: '{text[:30]}...' - Time: {elapsed_time:.4f}s (Unexpected cache hit)")
        else:
            results["misses"] += 1
            results["cache_miss_times"].append(elapsed_time)
            cache_hit = False
            logger.info(f"Text: '{text[:30]}...' - Time: {elapsed_time:.4f}s (Cache miss)")
        
        results["text_results"].append({
            "text": text,
            "generation_time": elapsed_time,
            "cache_hit": cache_hit
        })
    
    # Calculate statistics
    total_texts = results["exact_hits"] + results["misses"]
    results["total_texts"] = total_texts
    results["hit_rate"] = results["exact_hits"] / total_texts if total_texts > 0 else 0
    
    results["avg_generation_time"] = statistics.mean(results["generation_times"]) if results["generation_times"] else 0
    results["avg_cache_hit_time"] = statistics.mean(results["cache_hit_times"]) if results["cache_hit_times"] else 0
    results["avg_cache_miss_time"] = statistics.mean(results["cache_miss_times"]) if results["cache_miss_times"] else 0
    
    results["speedup"] = results["avg_cache_miss_time"] / results["avg_cache_hit_time"] if results["avg_cache_hit_time"] > 0 else 0
    
    # Get final cache stats
    results["cache_stats"] = get_embedding_stats()
    
    return results


def generate_mock_documents(query: str, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Generate mock retrieved documents for a query.
    
    Args:
        query: User's query
        topics: Optional list of topics to filter by
        
    Returns:
        List of mock retrieved documents
    """
    all_topics = ["glass_steagall", "new_deal", "sec"]
    
    # If topics are specified, use them, otherwise use all topics
    if topics is None:
        topics = all_topics
    
    # Generate mock documents
    documents = []
    for i in range(3):  # Generate 3 mock documents
        # Select a topic based on the query and specified topics
        if "Glass-Steagall" in query or "Banking Act" in query:
            topic = "glass_steagall" if "glass_steagall" in topics else random.choice(topics)
        elif "SEC" in query:
            topic = "sec" if "sec" in topics else random.choice(topics)
        elif "New Deal" in query:
            topic = "new_deal" if "new_deal" in topics else random.choice(topics)
        else:
            topic = random.choice(topics)
            
        # Generate a mock document
        document = {
            "text": f"This is a mock document {i+1} about {topic}. It contains information relevant to the query: {query[:50]}...",
            "metadata": {
                "topic": topic,
                "file_name": f"{topic}_document_{i+1}.txt"
            },
            "similarity": round(random.uniform(0.7, 0.95), 4)
        }
        documents.append(document)
    
    return documents


def generate_mock_response(query: str, documents: List[Dict[str, Any]]) -> str:
    """Generate a mock LLM response for a query and documents.
    
    Args:
        query: User's query
        documents: Retrieved documents
        
    Returns:
        Mock LLM response
    """
    # Extract topics from documents
    topics = set()
    for doc in documents:
        topic = doc.get("metadata", {}).get("topic")
        if topic:
            topics.add(topic)
    
    # Generate a mock response based on the query and topics
    if "Glass-Steagall" in query or "Banking Act" in query:
        return f"""Based on the provided documents, the Glass-Steagall Act (also known as the Banking Act of 1933) was a landmark legislation that separated commercial and investment banking activities. 

The key provisions included:
1. Separation of commercial and investment banking
2. Creation of the Federal Deposit Insurance Corporation (FDIC)
3. Regulation of interest rates on deposits

This act was a response to the financial crisis of the Great Depression and aimed to prevent banks from engaging in risky investment activities with depositors' money.

The documents indicate that this separation remained in effect until the late 1990s when the act was effectively repealed by the Gramm-Leach-Bliley Act."""
    
    elif "SEC" in query:
        return f"""According to the retrieved documents, the Securities and Exchange Commission (SEC) was established during the Great Depression as part of the regulatory reforms to restore investor confidence in the markets.

The SEC's primary purpose was to:
1. Enforce securities laws and regulations
2. Require public companies to disclose financial information
3. Prevent fraud and market manipulation
4. Protect investors

The SEC was created by the Securities Exchange Act of 1934, following the stock market crash of 1929. It was designed to provide transparency and accountability in the securities markets, which had been largely unregulated before the Great Depression."""
    
    elif "New Deal" in query:
        return f"""The documents show that the New Deal was a series of programs and projects instituted by President Franklin D. Roosevelt during the Great Depression to provide relief, recovery, and reform to the United States economy.

Key New Deal programs included:
1. The Civilian Conservation Corps (CCC)
2. The Works Progress Administration (WPA)
3. The Social Security Act
4. The Tennessee Valley Authority (TVA)

These programs aimed to stimulate economic recovery, provide jobs, and establish a social safety net. The New Deal represented a significant expansion of the federal government's role in the economy and society."""
    
    elif "Compare" in query or "similarities and differences" in query:
        return f"""Based on the provided documents, there are several similarities and differences between the New Deal programs and Glass-Steagall regulations:

Similarities:
1. Both were responses to the Great Depression
2. Both involved increased government regulation of the economy
3. Both aimed to restore stability and confidence in the financial system

Differences:
1. Scope: The New Deal was a broad set of programs across many sectors, while Glass-Steagall focused specifically on banking
2. Implementation: New Deal programs created new agencies and infrastructure, while Glass-Steagall primarily regulated existing institutions
3. Duration: Many New Deal programs still exist today, while Glass-Steagall was effectively repealed in 1999

Both represented significant shifts in government policy during a time of economic crisis."""
    
    else:
        return f"""According to the retrieved documents, the major financial reforms after the Great Depression included:

1. The Glass-Steagall Act (Banking Act of 1933), which separated commercial and investment banking activities and established the FDIC
2. The Securities Act of 1933, which required transparency in financial statements and established laws against misrepresentation and fraud
3. The Securities Exchange Act of 1934, which created the SEC to regulate the securities industry
4. The Banking Act of 1935, which strengthened the Federal Reserve's control over credit
5. Various New Deal programs that reformed other aspects of the financial system

These reforms fundamentally changed the financial regulatory landscape in the United States and many remained in effect for decades."""


def test_semantic_cache(test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test the semantic cache performance with mock responses.
    
    Args:
        test_queries: List of test queries
        
    Returns:
        Dictionary with test results
    """
    from src.llm.semantic_cache import semantic_cache, set_response
    
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
        
        # Generate mock documents and response
        start_time = time.time()
        documents = generate_mock_documents(query, topics)
        
        # Add a delay to simulate API call
        time.sleep(0.5)  # 500ms delay
        
        # Generate mock response
        response = generate_mock_response(query, documents)
        
        # Cache the response
        set_response(query, documents, response)
        
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
        
        # Generate mock documents (same as before)
        start_time = time.time()
        documents = generate_mock_documents(query, topics)
        
        # Try to get from cache
        cached_response = semantic_cache.get(query, documents)
        
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
        
        # Generate mock documents
        start_time = time.time()
        documents = generate_mock_documents(query, topics)
        
        # Try to get from cache
        cached_response = semantic_cache.get(query, documents)
        
        if cached_response is not None:
            # Semantic cache hit
            elapsed_time = time.time() - start_time
            results["semantic_hits"] += 1
            results["cache_hit_times"].append(elapsed_time)
            hit_type = "semantic"
            cache_hit = True
            logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Semantic cache hit)")
        else:
            # Cache miss, generate mock response
            time.sleep(0.5)  # 500ms delay to simulate API call
            response = generate_mock_response(query, documents)
            set_response(query, documents, response)
            
            elapsed_time = time.time() - start_time
            results["misses"] += 1
            results["cache_miss_times"].append(elapsed_time)
            hit_type = "miss"
            cache_hit = False
            logger.info(f"Query: '{query[:50]}...' - Time: {elapsed_time:.2f}s (Cache miss)")
        
        results["response_times"].append(elapsed_time)
        
        results["query_results"].append({
            "query": query,
            "topics": topics,
            "response_time": elapsed_time,
            "cache_hit": cache_hit,
            "hit_type": hit_type
        })
    
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


def main():
    """Main function to run the combined caching test."""
    parser = argparse.ArgumentParser(description='Test combined caching strategies')
    parser.add_argument('--embedding-only', action='store_true', help='Test only embedding cache')
    parser.add_argument('--semantic-only', action='store_true', help='Test only semantic cache')
    parser.add_argument('--quantized', action='store_true', help='Use quantized embedding model')
    args = parser.parse_args()
    
    # Create results directory
    results_dir = project_root / "data" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test embedding cache if requested or if no specific test is requested
    if args.embedding_only or (not args.embedding_only and not args.semantic_only):
        # Initialize the embedding generator
        if args.quantized:
            logger.info("Initializing quantized embedding generator...")
            embedding_generator = QuantizedEmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        else:
            logger.info("Initializing standard embedding generator...")
            embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        
        # Generate test texts
        test_texts = generate_test_texts()
        
        # Test embedding cache
        embedding_results = test_embedding_cache(embedding_generator, test_texts, args.quantized)
        
        # Save results
        model_type = "quantized" if args.quantized else "standard"
        embedding_results_file = results_dir / f"embedding_cache_{model_type}_results.json"
        with open(embedding_results_file, 'w', encoding='utf-8') as f:
            # Convert non-serializable objects to strings
            serializable_results = json.dumps(embedding_results, default=str)
            json.dump(json.loads(serializable_results), f, indent=2)
            
        logger.info(f"Embedding cache test results saved to {embedding_results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"EMBEDDING CACHE TEST SUMMARY ({model_type.upper()})")
        print("="*80)
        print(f"Total texts: {embedding_results['total_texts']}")
        print(f"Exact hits: {embedding_results['exact_hits']}")
        print(f"Misses: {embedding_results['misses']}")
        print(f"Hit rate: {embedding_results['hit_rate']:.2f}")
        print(f"Average generation time: {embedding_results['avg_generation_time']:.4f}s")
        print(f"Average cache hit time: {embedding_results['avg_cache_hit_time']:.4f}s")
        print(f"Average cache miss time: {embedding_results['avg_cache_miss_time']:.4f}s")
        print(f"Speedup: {embedding_results['speedup']:.2f}x")
        print("\nCache Statistics:")
        for key, value in embedding_results["cache_stats"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("="*80)
    
    # Test semantic cache if requested or if no specific test is requested
    if args.semantic_only or (not args.embedding_only and not args.semantic_only):
        # Generate test queries
        test_queries = generate_test_queries()
        
        # Test semantic cache
        semantic_results = test_semantic_cache(test_queries)
        
        # Save results
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
    
    # If both tests were run, print combined summary
    if not args.embedding_only and not args.semantic_only:
        print("\n" + "="*80)
        print("COMBINED CACHING STRATEGIES SUMMARY")
        print("="*80)
        model_type = "quantized" if args.quantized else "standard"
        print(f"Embedding Cache ({model_type}):")
        print(f"  Hit rate: {embedding_results['hit_rate']:.2f}")
        print(f"  Speedup: {embedding_results['speedup']:.2f}x")
        print(f"  Average cache hit time: {embedding_results['avg_cache_hit_time']:.4f}s")
        
        print("\nSemantic Cache:")
        print(f"  Hit rate: {semantic_results['hit_rate']:.2f}")
        print(f"  Semantic hit rate: {semantic_results['semantic_hit_rate']:.2f}")
        print(f"  Speedup: {semantic_results['speedup']:.2f}x")
        print(f"  Average cache hit time: {semantic_results['avg_cache_hit_time']:.2f}s")
        
        print("\nEstimated Combined Impact:")
        embedding_speedup = embedding_results['speedup']
        semantic_speedup = semantic_results['speedup']
        embedding_hit_rate = embedding_results['hit_rate']
        semantic_hit_rate = semantic_results['hit_rate']
        
        # Calculate weighted average speedup based on hit rates
        combined_speedup = (embedding_speedup * embedding_hit_rate + semantic_speedup * semantic_hit_rate) / (embedding_hit_rate + semantic_hit_rate) if (embedding_hit_rate + semantic_hit_rate) > 0 else 0
        
        print(f"  Combined speedup: {combined_speedup:.2f}x")
        print(f"  Estimated response time reduction: {(1 - 1/combined_speedup) * 100:.1f}%")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
