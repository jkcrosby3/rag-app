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
project_root = Path(__file__).parent.parent.parent  # Go up to rag-app root
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from src.document_management.document_manager import DocumentManager
from src.utils.db_statistics import get_vector_db_statistics
from dotenv import load_dotenv
import pickle
from collections import defaultdict

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize RAG system and document manager
@st.cache_resource
def get_rag_system():
    # For hackathon demo, use a default user with appropriate clearance
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    vector_db_path = "data/vector_db"
    return RAGSystem(
        user_id="demo_user", 
        llm_api_key=api_key,
        vector_db_path=vector_db_path
    )

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
    top_k = st.slider("Number of documents to retrieve", 1, 50, 20)
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
                    response = st.session_state.rag_system.process_query(
                        query=query,
                        top_k=top_k
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
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"Error processing query: {str(e)}")
                    st.error(f"Full error details:\n{error_details}")
                    print(f"QUERY ERROR: {error_details}")  # Print to terminal
                    
                    # Write to log file
                    with open("query_error.log", "w") as f:
                        f.write(f"Error: {str(e)}\n\n")
                        f.write(f"Full traceback:\n{error_details}")
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


def show_statistics_tab():
    """Display database statistics."""
    st.title("Database Statistics")
    
    st.markdown("""
    This dashboard shows statistics about the Revolutionary War documents in the vector database.
    """)
    
    try:
        # Get statistics using the utility module
        with st.spinner("Loading database statistics..."):
            stats = get_vector_db_statistics("data/vector_db")
        
        # Display overall statistics
        st.header("📊 Overall Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Document Chunks", f"{stats['total_chunks']:,}")
        with col2:
            st.metric("Total Unique Documents", f"{stats['total_unique_files']:,}")
        
        st.markdown("---")
        
        # Display by collection type
        st.header("📚 Documents by Collection")
        
        # Define display names and descriptions for document types
        collection_info = {
            'books': {
                'name': '📚 Books',
                'description': 'Historical documents',
                'terminology': 'books'
            },
            'pension_files': {
                'name': '⚔️ Revolutionary War Pension Files',
                'description': 'Pension applications from Revolutionary War soldiers and officers. Each file represents one veteran.',
                'terminology': 'soldiers and officers'
            },
            'newspapers': {
                'name': '📰 Historical Newspapers',
                'description': 'Newspaper issues from the War of 1812 era (1809-1815).',
                'terminology': 'newspaper issues'
            },
            'collections': {
                'name': '📜 Revolutionary Era Collections',
                'description': 'Historical documents and collections from the Revolutionary War period.',
                'terminology': 'historical documents'
            }
        }
        
        # Display each collection type
        for doc_type in sorted(stats['by_type'].keys()):
            info = stats['by_type'][doc_type]
            collection = collection_info.get(doc_type, {
                'name': doc_type.replace('_', ' ').title(),
                'description': 'Historical documents',
                'terminology': 'documents'
            })
            
            with st.expander(collection['name'], expanded=True):
                st.markdown(f"**Description:** {collection['description']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Documents", f"{info['unique_files']:,}")
                with col2:
                    st.metric("Total Chunks", f"{info['chunks']:,}")
                with col3:
                    st.metric("Avg Chunks/Doc", f"{info['avg_chunks_per_doc']:.1f}")
        
        st.markdown("---")
        
        # Additional insights
        st.header("💡 Key Insights")
        
        pension_count = stats['by_type'].get('pension_files', {}).get('unique_files', 0)
        newspaper_count = stats['by_type'].get('newspapers', {}).get('unique_files', 0)
        collection_count = stats['by_type'].get('collections', {}).get('unique_files', 0)
        
        st.markdown(f"""
        - **Revolutionary War Veterans**: Approximately **{pension_count:,}** soldiers and officers have pension files in the database
        - **Historical Newspapers**: **{newspaper_count:,}** newspaper issues from the War of 1812 era (1809-1815)
        - **Revolutionary Era Documents**: **{collection_count:,}** additional historical documents and collections
        - **Total Coverage**: The database spans multiple primary source collections from the American Revolution
        
        **Note on Terminology:**
        - **Soldiers**: Enlisted personnel who served in the Continental Army or state militias
        - **Officers**: Commissioned leaders (lieutenants, captains, majors, colonels, generals)
        - **Veterans**: Both soldiers and officers who survived and applied for pensions
        """)
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")
        import traceback
        st.code(traceback.format_exc())


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
    
    # Show vector database stats
    st.subheader("Vector Database Status")
    try:
        # Get comprehensive statistics
        stats = get_vector_db_statistics("data/vector_db")
        doc_count = stats['total_chunks']
        
        st.success(f"✅ **{doc_count:,} document chunks** loaded in vector database")
        
        # Display breakdown by document type
        newspaper_count = stats['by_type'].get('newspapers', {}).get('unique_files', 0)
        pension_count = stats['by_type'].get('pension_files', {}).get('unique_files', 0)
        collection_count = stats['by_type'].get('collections', {}).get('unique_files', 0)
        books_count = stats['by_type'].get('books', {}).get('unique_files', 0)
        
        st.markdown(f"""
        **Revolutionary War Documents Processed:**
        - **{newspaper_count:,}** newspaper text files from War of 1812 era (with enriched metadata)
        - **{pension_count:,}** Revolutionary War pension files
        - **{collection_count:,}** Revolutionary era collection documents
        - **{books_count:,}** historical books
        - All documents chunked and embedded for semantic search
        - Ready for querying!
        
        Try asking questions like:
        - "What newspapers are available?"
        - "Tell me about the Revolutionary War"
        - "What battles are mentioned in newspapers?"
        - "Who were Revolutionary War veterans?"
        """)
        
    except Exception as e:
        st.error(f"Error loading vector database stats: {str(e)}")

# Main app layout
def main():
    """Main application layout."""
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Query", "Statistics", "Documents"])
    
    with tab1:
        show_query_tab()
    
    with tab2:
        show_statistics_tab()
    
    with tab3:
        show_documents_tab()

if __name__ == "__main__":
    main()
