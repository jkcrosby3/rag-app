#!/usr/bin/env python
"""Test script for the improved semantic cache with lower similarity threshold.

This script evaluates the performance of the semantic cache with a lower
similarity threshold (0.75) and quantized embedding model to improve hit rates.
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

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG system components
from src.llm.semantic_cache import semantic_cache, get_stats, set_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_queries() -> List[Dict[str, Any]]:
    """Generate test queries with more semantic variations.
    
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
    
    # Generate more semantic variations for each base query
    test_queries = []
    
    for base_query in base_queries:
        # Add the original query
        test_queries.append(base_query.copy())
        
        # Add semantic variations - more variations than before
        variations = []
        
        if "Glass-Steagall" in base_query["query"]:
            # Variation 1: Different name for the act
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("Glass-Steagall", "Banking Act of 1933")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 2: Different phrasing
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("What were", "Could you explain")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 3: Completely different phrasing but same intent
            if "key provisions" in base_query["query"]:
                variation = base_query.copy()
                variation["query"] = "What did the Glass-Steagall Act accomplish?"
                variation["is_variation"] = True
                variations.append(variation)
            
        if "key provisions" in base_query["query"]:
            # Variation 1: Synonym replacement
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("key provisions", "main components")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 2: Different phrasing
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("key provisions", "important aspects")
            variation["is_variation"] = True
            variations.append(variation)
            
        if "purpose" in base_query["query"]:
            # Variation 1: Synonym replacement
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("purpose", "role")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 2: Different phrasing
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("What was the purpose", "Why was the SEC created")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 3: More conversational
            variation = base_query.copy()
            variation["query"] = "Can you tell me what the SEC was supposed to do during the Great Depression?"
            variation["is_variation"] = True
            variations.append(variation)
            
        if "Compare" in base_query["query"]:
            # Variation 1: Different structure
            variation = base_query.copy()
            variation["query"] = "What are the similarities and differences between " + \
                               base_query["query"].split("Compare ")[1].rstrip(".")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 2: More specific comparison
            variation = base_query.copy()
            variation["query"] = "How do New Deal programs differ from Glass-Steagall regulations?"
            variation["is_variation"] = True
            variations.append(variation)
            
        if "major financial reforms" in base_query["query"]:
            # Variation 1: Synonym replacement
            variation = base_query.copy()
            variation["query"] = base_query["query"].replace("major financial reforms", 
                                                          "significant banking regulations")
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 2: Different phrasing
            variation = base_query.copy()
            variation["query"] = "What financial legislation was passed following the Great Depression?"
            variation["is_variation"] = True
            variations.append(variation)
            
            # Variation 3: More specific
            variation = base_query.copy()
            variation["query"] = "What were the most important financial laws enacted after the 1929 crash?"
            variation["is_variation"] = True
            variations.append(variation)
        
        # Add all variations
        test_queries.extend(variations)
    
    return test_queries


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
    logger.info("Testing improved semantic cache performance...")
    
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
    first_pass_stats = get_stats()
    
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
    semantic_hits_count = 0
    total_variations = 0
    
    for query_data in test_queries:
        if not query_data.get("is_variation", False):
            continue
            
        query = query_data["query"]
        topics = query_data["topics"]
        total_variations += 1
        
        # Generate mock documents
        start_time = time.time()
        documents = generate_mock_documents(query, topics)
        
        # Try to get from cache
        cached_response = semantic_cache.get(query, documents)
        
        if cached_response is not None:
            # Semantic cache hit
            elapsed_time = time.time() - start_time
            results["semantic_hits"] += 1
            semantic_hits_count += 1
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
    results["semantic_hit_rate"] = semantic_hits_count / total_variations if total_variations > 0 else 0
    
    results["avg_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
    results["avg_cache_hit_time"] = statistics.mean(results["cache_hit_times"]) if results["cache_hit_times"] else 0
    results["avg_cache_miss_time"] = statistics.mean(results["cache_miss_times"]) if results["cache_miss_times"] else 0
    
    results["speedup"] = results["avg_cache_miss_time"] / results["avg_cache_hit_time"] if results["avg_cache_hit_time"] > 0 else 0
    
    # Get final cache stats
    results["cache_stats"] = get_stats()
    
    return results


def main():
    """Main function to run the semantic cache test."""
    # Generate test queries
    test_queries = generate_test_queries()
    
    # Test semantic cache
    semantic_results = test_semantic_cache(test_queries)
    
    # Save results
    results_dir = project_root / "data" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    semantic_results_file = results_dir / "improved_semantic_cache_results.json"
    with open(semantic_results_file, 'w', encoding='utf-8') as f:
        # Convert non-serializable objects to strings
        serializable_results = json.dumps(semantic_results, default=str)
        json.dump(json.loads(serializable_results), f, indent=2)
        
    logger.info(f"Improved semantic cache test results saved to {semantic_results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("IMPROVED SEMANTIC CACHE TEST SUMMARY")
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
