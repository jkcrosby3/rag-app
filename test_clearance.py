import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.document_processing.classification_chunker import ClassificationChunker
from src.embeddings.generator import EmbeddingGenerator
from src.vector_db.faiss_db import FAISSVectorDB
from src.rag_system import RAGSystem
from src.security.clearance_manager import ClearanceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clearance_system():
    """Test the clearance-based access control system."""
    # Initialize components
    chunker = ClassificationChunker()
    embedding_generator = EmbeddingGenerator()
    vector_db = FAISSVectorDB(index_path="data/vector_db")
    clearance_manager = ClearanceManager()
    
    # Process test document
    document_path = Path("data/documents/test_classification.txt")
    with open(document_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create chunks with classification metadata
    chunks = chunker.chunk_text(content)
    
    # Generate embeddings and add to vector database
    for chunk in chunks:
        embedding = embedding_generator.generate_embedding(chunk["text"])
        chunk["embedding"] = embedding
        vector_db.add_document(chunk)
    
    # Initialize test clearances
    test_clearances = {
        "user1": {
            "level": {
                "classification": "U",
                "components": []
            }
        },
        "user2": {
            "level": {
                "classification": "C",
                "components": []
            }
        },
        "user3": {
            "level": {
                "classification": "S",
                "components": []
            }
        },
        "user4": {
            "level": {
                "classification": "TS",
                "components": []
            }
        }
    }
    
    # Add test clearances to clearance manager
    for user_id, clearance in test_clearances.items():
        clearance_manager.add_clearance(
            user_id=user_id,
            clearance_level=clearance["level"]
        )
    
    # Test users with different clearances
    test_users = [
        {
            "id": "user1",
            "expected": ["U"]
        },
        {
            "id": "user2",
            "expected": ["U", "C"]
        },
        {
            "id": "user3",
            "expected": ["U", "C", "S"]
        },
        {
            "id": "user4",
            "expected": ["U", "C", "S", "TS"]
        }
    ]
    
    for user in test_users:
        logger.info(f"Testing user {user['id']}")
        
        # Initialize RAG system with user clearance
        rag_system = RAGSystem(
            user_id=user['id'],
            clearance_manager=clearance_manager,
            llm_api_key=None,  # No LLM needed for this test
            llm_model_name="test"  # Dummy model name
        )
        
        # Test query
        query = "Show me all the sections in the document"
        result = rag_system.process_query(
            query, 
            top_k=10,
            test_mode=True  # Use test mode to skip LLM generation
        )
        
        # Verify access
        accessible_classifications = set()
        for doc in result["documents"]:
            # Get classification from top level
            classification = doc.get("classification", {
                "classification": "U",
                "components": []
            })
            
            # Check if document is redacted
            is_redacted = doc.get("redacted", False)
            if not is_redacted:
                accessible_classifications.add(classification["classification"])  # Add base classification
        
        # Check if accessible classifications match expected
        assert set(user["expected"]) == accessible_classifications, \
            f"User {user['id']} should only see classifications: {user['expected']}"
        
        logger.info(f"User {user['id']} test passed - can access: {accessible_classifications}")
    
    # Process test document
    document_path = Path("data/documents/test_classification.txt")
    with open(document_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create chunks with classification metadata
    chunks = chunker.chunk_text(content)
    
    # Generate embeddings and add to vector database
    for chunk in chunks:
        embedding = embedding_generator.generate_embedding(chunk["text"])
        chunk["embedding"] = embedding
        vector_db.add_document(chunk)
    
    # Test different user clearances
    test_users = [
        {"id": "user1", "expected": ["U"]},
        {"id": "user2", "expected": ["U", "C"]},
        {"id": "user3", "expected": ["U", "C", "S"]},
        {"id": "user4", "expected": ["U", "C", "S", "TS"]}
    ]

if __name__ == "__main__":
    test_clearance_system()
