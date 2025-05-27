"""
Demo script for the RAG (Retrieval-Augmented Generation) system.

This script demonstrates the complete RAG pipeline:
1. Take a user query
2. Generate embeddings for the query
3. Retrieve relevant documents using vector similarity
4. Display the results
"""
import logging
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.generator import EmbeddingGenerator
from src.retrieval.retriever import Retriever


def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RAG Demo')
    parser.add_argument('query', type=str, help='Query to search for')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results to return')
    parser.add_argument('--topics', type=str, help='Comma-separated list of topics to filter by')
    return parser.parse_args()


def main():
    """Run the RAG demo."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting RAG demo")
    
    # Parse arguments
    args = parse_arguments()
    query = args.query
    top_k = args.top_k
    filter_topics = args.topics.split(',') if args.topics else None
    
    logger.info(f"Query: {query}")
    logger.info(f"Top-k: {top_k}")
    if filter_topics:
        logger.info(f"Filter topics: {filter_topics}")
    
    # Define paths
    vector_db_path = project_root / "data" / "vector_db"
    
    # Check if vector database exists
    if not vector_db_path.exists():
        logger.error(f"Vector database not found at {vector_db_path}")
        logger.error("Please run 'python scripts/build_vector_db.py' first")
        return 1
    
    try:
        # Initialize retriever
        retriever = Retriever(vector_db_path=vector_db_path)
        logger.info(f"Loaded vector database with {retriever.get_document_count()} documents")
        
        # Retrieve relevant documents
        logger.info("Retrieving relevant documents...")
        results = retriever.retrieve(query, top_k=top_k, filter_topics=filter_topics)
        
        # Display results
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        if not results:
            print("\nNo relevant documents found.")
        else:
            print(f"\nFound {len(results)} relevant documents:")
            
            for i, result in enumerate(results):
                similarity = result.get('similarity', 0)
                topic = result.get('metadata', {}).get('topic', 'unknown')
                file_name = result.get('metadata', {}).get('file_name', 'unknown')
                chunk_index = result.get('metadata', {}).get('chunk_index', 0)
                total_chunks = result.get('metadata', {}).get('total_chunks', 1)
                
                print(f"\n--- Result {i+1} (Similarity: {similarity:.4f}) ---")
                print(f"Topic: {topic}")
                print(f"Source: {file_name} (Chunk {chunk_index+1}/{total_chunks})")
                print("-"*40)
                
                # Print a preview of the text (first 300 characters)
                text = result.get('text', '')
                print(f"{text[:300]}...")
                
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error in RAG demo: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
