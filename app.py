"""
RAG System Web Interface

This module provides a Streamlit-based web interface for the RAG system,
allowing users to ask questions, view responses, and see supporting documents.
"""
import os
import sys
import json
import time
import datetime
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_VECTOR_DB_PATH = os.path.join("data", "vector_db")
DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20241022"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOPICS = ["glass_steagall", "new_deal", "sec"]

# Initialize session state
def init_session_state():
    if "rag_system" not in st.session_state:
        # Get API key from environment or user input
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        try:
            # Initialize RAG system
            st.session_state.rag_system = RAGSystem(
                vector_db_type="faiss",
                vector_db_path=DEFAULT_VECTOR_DB_PATH,
                embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                llm_api_key=api_key,
                llm_model_name=DEFAULT_MODEL_NAME,
                use_quantized_embeddings=True
            )
            st.session_state.system_ready = True
        except Exception as e:
            st.session_state.system_ready = False
            st.session_state.system_error = str(e)
            logger.error(f"Error initializing RAG system: {str(e)}", exc_info=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_session_id" not in st.session_state:
        # Generate a unique session ID based on timestamp
        st.session_state.current_session_id = f"session_{int(time.time())}"

def save_session(session_data, filename=None):
    """Save the current session to a JSON file."""
    if filename is None:
        # Generate filename based on session ID and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
    
    # Ensure the sessions directory exists
    os.makedirs("sessions", exist_ok=True)
    filepath = os.path.join("sessions", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    
    return filepath

def save_response(response_data, filename=None):
    """Save a single response to a text file."""
    if filename is None:
        # Generate filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.txt"
    
    # Ensure the responses directory exists
    os.makedirs("responses", exist_ok=True)
    filepath = os.path.join("responses", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"QUERY: {response_data['query']}\n\n")
        f.write(f"RESPONSE:\n{response_data['response']}\n\n")
        f.write("SUPPORTING DOCUMENTS:\n")
        for i, doc in enumerate(response_data['retrieved_documents']):
            f.write(f"\n--- Document {i+1} ---\n")
            f.write(f"Topic: {doc.get('metadata', {}).get('topic', 'unknown')}\n")
            f.write(f"Source: {doc.get('metadata', {}).get('file_name', 'unknown')}\n")
            f.write(f"Similarity: {doc.get('similarity', 0):.4f}\n")
            f.write(f"Content: {doc.get('text', '')[:500]}...\n")
    
    return filepath

def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def main():
    st.set_page_config(
        page_title="RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("RAG System")
        st.markdown("---")
        
        # API Key input if not already set
        if not os.getenv("ANTHROPIC_API_KEY"):
            api_key = st.text_input("Anthropic API Key", type="password")
            if api_key and "rag_system" in st.session_state:
                st.session_state.rag_system.llm_client.api_key = api_key
        
        # Topic filter
        st.subheader("Filter by Topics")
        topics = st.multiselect(
            "Select topics to include",
            options=DEFAULT_TOPICS,
            default=[]
        )
        
        # Advanced settings
        st.subheader("Advanced Settings")
        with st.expander("Query Parameters"):
            top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Max response tokens", 100, 2000, 1000, 100)
            use_cache = st.checkbox("Use response cache", value=True)
        
        # Session management
        st.subheader("Session Management")
        if st.button("Save Current Session"):
            if st.session_state.chat_history:
                filepath = save_session({
                    "session_id": st.session_state.current_session_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chat_history": st.session_state.chat_history
                })
                st.success(f"Session saved to {filepath}")
            else:
                st.warning("No chat history to save")
        
        if st.button("Start New Session"):
            st.session_state.chat_history = []
            st.session_state.current_session_id = f"session_{int(time.time())}"
            st.success("Started new session")
        
        # Display cache stats
        if "rag_system" in st.session_state and st.session_state.system_ready:
            st.subheader("Cache Statistics")
            with st.expander("View Cache Stats"):
                try:
                    from src.llm.semantic_cache import get_stats as get_semantic_cache_stats
                    from src.embeddings.model_cache import get_stats as get_embedding_cache_stats
                    
                    semantic_stats = get_semantic_cache_stats()
                    embedding_stats = get_embedding_cache_stats()
                    
                    st.markdown("**Semantic Cache**")
                    st.text(f"Size: {semantic_stats.get('size', 0)}/{semantic_stats.get('max_size', 0)}")
                    st.text(f"Hit rate: {semantic_stats.get('hit_rate', 0):.2f}")
                    
                    st.markdown("**Embedding Cache**")
                    st.text(f"Size: {embedding_stats.get('size', 0)}/{embedding_stats.get('max_size', 0)}")
                    st.text(f"Hit rate: {embedding_stats.get('hit_rate', 0):.2f}")
                except Exception as e:
                    st.error(f"Error loading cache stats: {str(e)}")
    
    # Main content
    st.title("RAG System Interface")
    
    # Check if system is ready
    if not st.session_state.get("system_ready", False):
        st.error(f"RAG system is not ready: {st.session_state.get('system_error', 'Unknown error')}")
        st.info("Please check your API keys and configuration.")
        return
    
    # Query input
    query = st.text_area("Enter your question:", height=100)
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", use_container_width=True)
    
    # Process query when submitted
    if submit_button and query:
        with st.spinner("Processing your query..."):
            try:
                # Process the query
                start_time = time.time()
                result = st.session_state.rag_system.process_query(
                    query=query,
                    top_k=top_k,
                    filter_topics=topics if topics else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_cache=use_cache
                )
                total_time = time.time() - start_time
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": query,
                    "result": result
                })
                
                # Scroll to the bottom to show new response
                st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
    
    # Display chat history
    st.markdown("---")
    st.subheader("Conversation History")
    
    if not st.session_state.chat_history:
        st.info("No conversation history yet. Ask a question to get started!")
    else:
        for i, item in enumerate(st.session_state.chat_history):
            result = item["result"]
            
            # Create a container for each Q&A pair
            with st.container():
                # Query
                st.markdown(f"**Question:**")
                st.markdown(f"> {result['query']}")
                
                # Response
                st.markdown(f"**Answer:**")
                st.markdown(result['response'])
                
                # Supporting documents in an expander
                with st.expander("View Supporting Documents"):
                    for j, doc in enumerate(result['retrieved_documents']):
                        similarity = doc.get('similarity', 0)
                        topic = doc.get('metadata', {}).get('topic', 'unknown')
                        file_name = doc.get('metadata', {}).get('file_name', 'unknown')
                        
                        st.markdown(f"**Document {j+1}** (Similarity: {similarity:.4f})")
                        st.markdown(f"**Topic:** {topic}")
                        st.markdown(f"**Source:** {file_name}")
                        st.text(doc.get('text', '')[:500] + "...")
                
                # Performance metrics in an expander
                with st.expander("View Performance Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Time", format_time(result['processing_time']))
                    with col2:
                        st.metric("Embedding Time", format_time(result['embedding_time']))
                    with col3:
                        st.metric("Retrieval Time", format_time(result['retrieval_time']))
                    with col4:
                        st.metric("Generation Time", format_time(result['generation_time']))
                
                # Save this response
                if st.button(f"Save Response #{i+1}", key=f"save_{i}"):
                    filepath = save_response(result)
                    st.success(f"Response saved to {filepath}")
            
            # Add a separator between items
            st.markdown("---")

if __name__ == "__main__":
    main()
