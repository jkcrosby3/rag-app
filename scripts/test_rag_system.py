"""
End-to-end test script for the RAG system.

This script tests the RAG system with various queries to validate
performance, accuracy, and response times.
"""
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test queries covering different topics and query types
TEST_QUERIES = [
    {
        "query": "What were the key provisions of the Glass-Steagall Act?",
        "topics": ["glass_steagall"],
        "description": "Factual query about Glass-Steagall provisions"
    },
    {
        "query": "How did the Glass-Steagall Act separate commercial and investment banking?",
        "topics": ["glass_steagall"],
        "description": "Specific mechanism query about Glass-Steagall"
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
        "topics": None,  # Test without topic filtering
        "description": "Broad query across all topics"
    }
]


def run_test_queries(vector_db_type: str = "faiss") -> Dict[str, Any]:
    """Run test queries through the RAG system and collect performance metrics.
    
    Args:
        vector_db_type: Vector database backend to use ('faiss' or 'elasticsearch')
        
    Returns:
        Dict containing test results and performance metrics
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG system
    logger.info(f"Initializing RAG system with {vector_db_type} backend")
    rag_system = RAGSystem(vector_db_type=vector_db_type)
    
    results = {
        "vector_db_type": vector_db_type,
        "total_queries": len(TEST_QUERIES),
        "queries": [],
        "avg_response_time": 0,
        "max_response_time": 0,
        "min_response_time": float('inf')
    }
    
    total_time = 0
    
    # Process each test query
    for i, test_query in enumerate(TEST_QUERIES):
        query = test_query["query"]
        topics = test_query["topics"]
        description = test_query["description"]
        
        logger.info(f"Running test query {i+1}/{len(TEST_QUERIES)}: {description}")
        logger.info(f"Query: {query}")
        if topics:
            logger.info(f"Topics: {topics}")
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Process query
            result = rag_system.process_query(
                query=query,
                top_k=3,
                filter_topics=topics
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            total_time += response_time
            
            # Update min/max response times
            results["max_response_time"] = max(results["max_response_time"], response_time)
            results["min_response_time"] = min(results["min_response_time"], response_time)
            
            # Store query result
            query_result = {
                "query": query,
                "topics": topics,
                "description": description,
                "response_time": response_time,
                "num_docs_retrieved": len(result["retrieved_documents"]),
                "response_length": len(result["response"]),
                "success": True
            }
            
            logger.info(f"Query processed successfully in {response_time:.2f} seconds")
            logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
            logger.info(f"Response length: {len(result['response'])} characters")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            
            # Store error result
            query_result = {
                "query": query,
                "topics": topics,
                "description": description,
                "error": str(e),
                "success": False
            }
        
        # Add query result to results
        results["queries"].append(query_result)
        
        # Add a separator between queries
        logger.info("-" * 80)
    
    # Calculate average response time
    successful_queries = [q for q in results["queries"] if q["success"]]
    if successful_queries:
        results["avg_response_time"] = total_time / len(successful_queries)
    
    # If all queries failed, set min_response_time to 0
    if results["min_response_time"] == float('inf'):
        results["min_response_time"] = 0
    
    # Log summary
    logger.info("Test Summary:")
    logger.info(f"Total queries: {len(TEST_QUERIES)}")
    logger.info(f"Successful queries: {len(successful_queries)}")
    logger.info(f"Average response time: {results['avg_response_time']:.2f} seconds")
    logger.info(f"Min response time: {results['min_response_time']:.2f} seconds")
    logger.info(f"Max response time: {results['max_response_time']:.2f} seconds")
    
    return results


def save_results(results: Dict[str, Any], output_file: str = "test_results.json"):
    """Save test results to a file.
    
    Args:
        results: Test results dict
        output_file: Output file path
    """
    output_path = project_root / "data" / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Test results saved to {output_path}")


def main():
    """Run the end-to-end tests."""
    logger.info("Starting RAG system end-to-end tests")
    
    # Run tests with FAISS backend
    results = run_test_queries(vector_db_type="faiss")
    
    # Save results
    save_results(results)
    
    # Check if any query exceeds the 2-second response time requirement
    slow_queries = [q for q in results["queries"] if q.get("response_time", 0) > 2.0]
    if slow_queries:
        logger.warning(f"Found {len(slow_queries)} queries exceeding the 2-second response time requirement:")
        for q in slow_queries:
            logger.warning(f"  - '{q['query']}': {q.get('response_time', 0):.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
