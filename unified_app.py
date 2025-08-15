"""
Unified RAG System Interface

This module provides a single interface for both querying the RAG system
and managing documents through a Streamlit-based web interface.
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

from rag_system import RAGSystem
from document_management.document_manager import DocumentManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG system and document manager
@st.cache_resource
def get_rag_system():
    return RAGSystem()

@st.cache_resource
def get_document_manager():
    return DocumentManager()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.rag_system = get_rag_system()
    st.session_state.document_manager = get_document_manager()
    st.session_state.processing = False
    st.session_state.active_tab = "Query"  # Default tab

# Page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stTextArea>div>div>textarea {
        min-height: 100px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 10%;
    }
    .assistant-message {
        background-color: #e6f7ff;
        margin-right: 10%;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # Model settings
    st.subheader("Model Settings")
    model_type = st.selectbox(
        "LLM Model",
        ["claude", "openai"],  # Add more models as needed
        index=0
    )
    
    # RAG settings
    st.subheader("RAG Settings")
    top_k = st.slider("Number of documents to retrieve", 1, 10, 3)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)
    
    # System info
    st.subheader("System Info")
    st.text(f"Python: {sys.version.split()[0]}")
    st.text(f"Streamlit: {st.__version__}")

# Main app
def show_query_tab():
    """Display the query tab content."""
    st.title("Query RAG System")
    
    # Chat interface
    query = st.text_area("Ask a question about your documents:", height=100)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit", type="primary"):
            if query:
                st.session_state.processing = True
                try:
                    # Process the query
                    response = st.session_state.rag_system.query(
                        query=query,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        model_type=model_type
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response.get("sources", []),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                finally:
                    st.session_state.processing = False
                    st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    st.subheader("Conversation History")
    if not st.session_state.chat_history:
        st.info("No conversation history yet. Ask a question to get started!")
    else:
        for msg in st.session_state.chat_history:
            with st.container():
                if msg["role"] == "user":
                    st.markdown(f"**You**: {msg['content']}")
                else:
                    st.markdown(f"**Assistant**: {msg['content']}")
                    
                    # Show sources if available
                    if "sources" in msg and msg["sources"]:
                        with st.expander("View Sources"):
                            for i, source in enumerate(msg["sources"], 1):
                                st.markdown(f"**Source {i}**")
                                st.markdown(f"**Document**: {source.get('document', 'Unknown')}")
                                st.markdown(f"**Page**: {source.get('page', 'N/A')}")
                                st.markdown(f"**Similarity**: {source.get('similarity', 0):.2f}")
                                st.markdown("---")


def show_documents_tab():
    """Display the documents management tab content."""
    st.title("Document Management")
    
    # Upload files
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save the file temporarily
                        file_path = os.path.join("uploads", uploaded_file.name)
                        os.makedirs("uploads", exist_ok=True)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the document
                        st.session_state.document_manager.process_document(file_path)
                        st.success(f"Processed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # List documents
    st.subheader("Document Library")
    try:
        documents = st.session_state.document_manager.list_documents()
        if not documents.empty:
            for _, doc in documents.iterrows():
                with st.expander(f"📄 {doc['title']}"):
                    st.markdown(f"**ID**: {doc['id']}")
                    st.markdown(f"**Type**: {doc['type']}")
                    st.markdown(f"**Status**: {doc['status']}")
                    
                    if st.button(f"Process {doc['id']}"):
                        with st.spinner(f"Processing {doc['title']}..."):
                            try:
                                st.session_state.document_manager.process_document(doc['id'])
                                st.success(f"Processed: {doc['title']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing document: {str(e)}")
        else:
            st.info("No documents found. Upload some files to get started!")
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

# Main app layout
def main():
    """Main application layout."""
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Query", "Documents"])
    
    with tab1:
        show_query_tab()
    
    with tab2:
        show_documents_tab()

if __name__ == "__main__":
    main()
