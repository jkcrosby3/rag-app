"""
Test script to verify document retrieval in the RAG system
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Constants
    DEFAULT_VECTOR_DB_PATH = os.path.join("data", "vector_db")
    DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20241022"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Initialize RAG system
    rag_system = RAGSystem(
        vector_db_type="faiss",
        vector_db_path=DEFAULT_VECTOR_DB_PATH,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        llm_api_key=api_key,
        llm_model_name=DEFAULT_MODEL_NAME,
        use_quantized_embeddings=True
    )
    
    # Test query
    query = "What were the effects of the banking crisis during the Great Depression?"
    
    # Process query
    result = rag_system.process_query(
        query=query,
        top_k=5,
        filter_topics=None,
        temperature=0.7,
        max_tokens=1000,
        use_cache=True
    )
    
    # Print retrieved documents
    print(f"\nRetrieved {len(result.get('retrieved_documents', []))} documents for query: {query}")
    
    for i, doc in enumerate(result.get('retrieved_documents', [])):
        print(f"\n--- Document {i+1} ---")
        print(f"Similarity: {doc.get('similarity', 0):.4f}")
        
        # Print metadata
        if 'metadata' in doc:
            print("Metadata:")
            for key, value in doc['metadata'].items():
                print(f"  {key}: {value}")
        
        # Print text preview
        print("\nText preview:")
        print(f"{doc.get('text', '')[:300]}...")
    
    # Save result to file for inspection
    with open("test_result.json", "w") as f:
        # Convert result to JSON-serializable format
        serializable_result = {
            "query": result.get("query", ""),
            "response": result.get("response", ""),
            "retrieved_documents": []
        }
        
        for doc in result.get("retrieved_documents", []):
            serializable_doc = {
                "text": str(doc.get("text", ""))[:500],
                "similarity": float(doc.get("similarity", 0))
            }
            
            if "metadata" in doc:
                serializable_doc["metadata"] = {}
                for k, v in doc["metadata"].items():
                    serializable_doc["metadata"][k] = str(v)
            
            serializable_result["retrieved_documents"].append(serializable_doc)
        
        json.dump(serializable_result, f, indent=2)
    
    print(f"\nSaved result to test_result.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
