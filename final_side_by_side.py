"""
Final RAG System Web Interface with side-by-side source documents
"""
import os
import sys
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash, Response
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

# Constants
DEFAULT_VECTOR_DB_PATH = os.path.join("data", "vector_db")
DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20241022"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "rag-app-secret-key")

# Initialize RAG system
rag_system = None
system_error = None

try:
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
    logger.info("RAG system initialized successfully")
except Exception as e:
    system_error = str(e)
    logger.error(f"Error initializing RAG system: {str(e)}", exc_info=True)

# Test query to ensure documents are retrieved
def test_document_retrieval():
    """Test document retrieval to ensure it's working"""
    try:
        query = "What were the effects of the banking crisis during the Great Depression?"
        result = rag_system.process_query(
            query=query,
            top_k=5,
            filter_topics=None,
            temperature=0.7,
            max_tokens=1000,
            use_cache=False
        )
        
        doc_count = len(result.get('retrieved_documents', []))
        logger.info(f"Test query retrieved {doc_count} documents")
        
        if doc_count > 0:
            first_doc = result.get('retrieved_documents', [])[0]
            logger.info(f"First document similarity: {first_doc.get('similarity', 0)}")
            if 'metadata' in first_doc:
                logger.info(f"First document metadata: {first_doc['metadata']}")
            if 'text' in first_doc:
                logger.info(f"First document text preview: {first_doc['text'][:100]}...")
            
            # Add a test item to the session for debugging
            test_chat_item = {
                'id': str(uuid.uuid4()),
                'query': query,
                'response': 'This is a test response to verify source document display.',
                'sources': []
            }
            
            # Add the retrieved documents as sources
            for doc in result.get('retrieved_documents', []):
                source = {
                    'id': str(uuid.uuid4()),
                    'text': doc.get('text', '')[:500] + '...' if doc.get('text') else '',
                    'similarity': float(doc.get('similarity', 0)),
                    'metadata': doc.get('metadata', {})
                }
                test_chat_item['sources'].append(source)
            
            # Store this test item in the app config for debugging
            app.config['TEST_CHAT_ITEM'] = test_chat_item
        
        return doc_count > 0
    except Exception as e:
        logger.error(f"Error in test document retrieval: {str(e)}", exc_info=True)
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page and handle form submissions."""
    # Initialize session if needed
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Check if this is a new session
    is_new_session = session.get('is_new_session', False)
    if is_new_session:
        # Clear the flag
        session['is_new_session'] = False
        # Don't add test chat item for a new session
        logger.info("New session detected, not adding test chat item")
    elif not session['chat_history'] and 'TEST_CHAT_ITEM' in app.config:
        # Add test chat item if available and session is empty (and not a new session)
        session['chat_history'].append(app.config['TEST_CHAT_ITEM'])
        session.modified = True
        logger.info("Added test chat item to session")
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            query = request.form.get('query', '')
            
            if not query:
                flash("Query cannot be empty", "danger")
                return redirect(url_for('index'))
            
            # Process the query
            try:
                result = rag_system.process_query(
                    query=query,
                    top_k=5,
                    filter_topics=None,
                    temperature=0.7,
                    max_tokens=1000,
                    use_cache=False
                )
                # Log the raw result for debugging
                logger.info(f"Raw result keys: {list(result.keys())}")
            except Exception as e:
                logger.error(f"Error in process_query: {str(e)}", exc_info=True)
                raise e
            
            # Log the result for debugging
            logger.info(f"Query result keys: {result.keys()}")
            doc_count = len(result.get('retrieved_documents', []))
            logger.info(f"Retrieved {doc_count} documents for query: {query}")
            
            # Create a simplified result for storage
            chat_item = {
                'id': str(uuid.uuid4()),
                'query': query,
                'response': result.get('response', ''),
                'sources': []
            }
            
            # Add source documents
            retrieved_docs = result.get('retrieved_documents', [])
            
            # Ensure we have documents
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    try:
                        # Extract metadata
                        metadata = {}
                        if 'metadata' in doc and isinstance(doc['metadata'], dict):
                            metadata = doc['metadata'].copy()
                            
                        # Add document ID if not present
                        if 'id' not in metadata and 'source' in metadata:
                            # Extract document ID from source path
                            source_path = metadata.get('source', '')
                            if source_path:
                                filename = source_path.split('/')[-1].split('\\')[-1]
                                metadata['id'] = filename
                        
                        # Extract document name
                        doc_name = None
                        if 'title' in metadata:
                            doc_name = metadata['title']
                        elif 'filename' in metadata:
                            doc_name = metadata['filename']
                        elif 'file_name' in metadata:
                            doc_name = metadata['file_name']
                        elif 'source' in metadata:
                            source_path = metadata.get('source', '')
                            if source_path:
                                doc_name = source_path.split('/')[-1].split('\\')[-1]
                        
                        # Ensure we have a document name
                        if doc_name:
                            metadata['document_name'] = doc_name
                        else:
                            metadata['document_name'] = f"Document {i+1}"
                                
                        # Add document URL if available
                        if 'url' not in metadata and 'source' in metadata:
                            source_path = metadata.get('source', '')
                            if source_path:
                                # Create a relative URL to the document if it's a local file
                                if os.path.exists(source_path):
                                    metadata['url'] = f"/view_document?path={source_path}"
                                    
                        # Add page info if available
                        if 'page' in metadata:
                            metadata['location'] = f"Page {metadata['page']}"
                        elif 'chunk_index' in metadata:
                            metadata['location'] = f"Chunk {metadata['chunk_index']}"
                        
                        # Log metadata for debugging
                        logger.info(f"Document {i+1} metadata: {metadata}")
                        if 'document_name' in metadata:
                            logger.info(f"Document {i+1} name: {metadata['document_name']}")
                        else:
                            logger.info(f"Document {i+1} has no name extracted")
                        
                        # Get document text safely
                        doc_text = doc.get('text', '')[:500] + '...' if doc.get('text') else ''
                        
                        # Create source object
                        source = {
                            'id': str(uuid.uuid4()),
                            'text': doc_text,
                            'similarity': float(doc.get('similarity', 0)),
                            'metadata': metadata
                        }
                        
                        chat_item['sources'].append(source)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {i}: {str(e)}")
            
            # Add to chat history
            session['chat_history'].append(chat_item)
            session.modified = True
            
            # Redirect to show updated history
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('index'))
    
    # Print the chat history for debugging
    logger.info(f"Current chat history: {json.dumps(session.get('chat_history', []), indent=2)}")
    
    # HTML Template with side-by-side layout
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG System with Side-by-Side Sources</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
        <style>
            body {
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
            }
            .chat-container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .message-row {
                display: flex;
                margin-bottom: 20px;
                gap: 20px;
            }
            .message {
                flex: 0 0 60%;
            }
            .sources-panel {
                flex: 0 0 38%;
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
                max-height: 300px;
                overflow-y: auto;
            }
            .message-content {
                padding: 15px;
                border-radius: 10px;
                position: relative;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .user-message .message-content {
                background-color: #e3f2fd;
                color: #0d47a1;
            }
            .system-message .message-content {
                background-color: white;
                color: #212529;
                border: 1px solid #e9ecef;
            }
            .message-label {
                font-size: 0.75rem;
                color: #6c757d;
                margin-bottom: 4px;
                font-weight: 600;
            }
            .source-text {
                font-family: monospace;
                white-space: pre-wrap;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                font-size: 0.85em;
                max-height: 150px;
                overflow-y: auto;
            }
            .source-metadata {
                font-size: 0.85em;
                color: #666;
            }
            .sources-title {
                color: #495057;
                font-size: 0.9rem;
                font-weight: 600;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #dee2e6;
            }
            .input-area {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                position: sticky;
                bottom: 0;
                z-index: 100;
            }
            .chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #dee2e6;
            }
            .source-item {
                margin-bottom: 10px;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                overflow: hidden;
            }
            .source-header {
                padding: 8px 12px;
                background-color: #e9ecef;
                font-weight: 500;
                font-size: 0.9rem;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .source-link {
                color: #007bff;
                text-decoration: none;
                margin-left: 8px;
            }
            .source-link:hover {
                text-decoration: underline;
            }
            .source-body {
                padding: 10px;
                border-top: 1px solid #dee2e6;
                display: none;
            }
            .source-body.show {
                display: block;
            }
            .navigation-links {
                position: fixed;
                right: 20px;
                z-index: 1000;
            }
            .navigation-links.top {
                top: 20px;
            }
            .navigation-links.bottom {
                bottom: 20px;
            }
            .nav-btn {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: #007bff;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                text-decoration: none;
            }
            .nav-btn:hover {
                background-color: #0056b3;
                color: white;
            }
        </style>
    </head>
    <body id="top">
        <!-- Navigation links -->
        <div class="navigation-links top">
            <a href="#bottom" class="nav-btn" title="Go to bottom">
                <i class="bi bi-arrow-down"></i>
            </a>
        </div>
        <div class="navigation-links bottom">
            <a href="#top" class="nav-btn" title="Go to top">
                <i class="bi bi-arrow-up"></i>
            </a>
        </div>
        
        <div class="container">
            <div class="row">
                <div class="col-12">
                    <div class="chat-container">
                        <div class="chat-header">
                            <h3>RAG System with Source Documents</h3>
                            <form method="POST" action="/new_session" class="d-inline">
                                <button type="submit" class="btn btn-sm btn-outline-danger">New Conversation</button>
                            </form>
                        </div>
                        
                        {% with messages = get_flashed_messages(with_categories=true) %}
                            {% if messages %}
                                {% for category, message in messages %}
                                    <div class="alert alert-{{ category }} mt-3">{{ message }}</div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}
                        
                        <div class="chat-messages">
                            {% if chat_history %}
                                {% for item in chat_history %}
                                    <!-- User query -->
                                    <div class="message-row">
                                        <div class="message user-message">
                                            <div class="message-content">
                                                <div class="message-label">You</div>
                                                {{ item.query }}
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- System response with sources -->
                                    <div class="message-row">
                                        <div class="message system-message">
                                            <div class="message-content">
                                                <div class="message-label">RAG System</div>
                                                {{ item.response }}
                                            </div>
                                        </div>
                                        
                                        <div class="sources-panel">
                                            <div class="sources-title">Source Documents ({{ item.sources|length }})</div>
                                            {% if item.sources %}
                                                {% for source in item.sources %}
                                                <div class="source-item">
                                                    <div class="source-header" onclick="toggleSource('{{ source.id }}')">
                                                        <div>
                                                            <div class="fw-bold">
                                                                {% if source.metadata.document_name %}
                                                                    {{ source.metadata.document_name }}
                                                                {% else %}
                                                                    Document {{ loop.index }}
                                                                {% endif %}
                                                            </div>
                                                            <div>
                                                                <small class="text-muted">ID: {{ source.metadata.id|default('N/A') }}</small>
                                                                {% if source.metadata.location %}
                                                                    <small class="text-muted ms-2">{{ source.metadata.location }}</small>
                                                                {% endif %}
                                                            </div>
                                                        </div>
                                                        <div>
                                                            <span class="badge bg-primary">{{ (source.similarity * 100)|round(1) }}%</span>
                                                        </div>
                                                    </div>
                                                    <div id="{{ source.id }}" class="source-body">
                                                        {% if source.metadata %}
                                                        <div class="source-metadata mb-2">
                                                            <strong>Metadata:</strong>
                                                            <ul class="list-unstyled">
                                                                {% for key, value in source.metadata.items() %}
                                                                <li>
                                                                    <strong>{{ key }}:</strong> 
                                                                    {% if key == 'source' and source.metadata.url %}
                                                                        <a href="{{ source.metadata.url }}" target="_blank" class="text-primary">{{ value }} <i class="bi bi-box-arrow-up-right"></i></a>
                                                                    {% else %}
                                                                        {{ value }}
                                                                    {% endif %}
                                                                </li>
                                                                {% endfor %}
                                                            </ul>
                                                        </div>
                                                        {% endif %}
                                                        <div class="source-text">
                                                            {{ source.text }}
                                                        </div>
                                                    </div>
                                                </div>
                                                {% endfor %}
                                            {% else %}
                                                <div class="text-center text-muted">
                                                    No source documents available
                                                </div>
                                            {% endif %}
                                        </div>
                                    </div>
                                {% endfor %}
                            {% else %}
                                <div class="text-center text-muted py-5">
                                    <p>No conversation yet. Ask a question to get started!</p>
                                </div>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div class="input-area" id="bottom">
                        <form method="POST" action="/">
                            <div class="row align-items-center">
                                <div class="col">
                                    <textarea name="query" class="form-control" rows="2" placeholder="Ask a question about the Great Depression..." required></textarea>
                                </div>
                                <div class="col-auto">
                                    <button type="submit" class="btn btn-primary">Send</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Function to toggle source document visibility
            function toggleSource(id) {
                const element = document.getElementById(id);
                if (element) {
                    element.classList.toggle('show');
                }
            }
        </script>
    </body>
    </html>
    """
    
    # Render the template with chat history
    return render_template_string(template, chat_history=session.get('chat_history', []))

@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session."""
    # Clear all session data
    session.clear()
    # Initialize empty chat history
    session['chat_history'] = []
    # Set a flag to indicate this is a fresh session
    session['is_new_session'] = True
    return redirect(url_for('index'))

@app.route('/view_document')
def view_document():
    """View a document."""
    path = request.args.get('path', '')
    
    if not path or not os.path.exists(path):
        return "Document not found", 404
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(path)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ filename }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                pre { white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="mb-3">
                    <a href="{{ url_for('index') }}" class="btn btn-primary">&laquo; Back to RAG System</a>
                </div>
                <div class="card">
                    <div class="card-header">
                        <h4>{{ filename }}</h4>
                    </div>
                    <div class="card-body">
                        <pre>{{ content }}</pre>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """, filename=filename, content=content)
    except Exception as e:
        return f"Error reading document: {str(e)}", 500

if __name__ == "__main__":
    # Run a test to ensure document retrieval is working
    docs_working = test_document_retrieval()
    if not docs_working:
        logger.warning("Document retrieval test failed. Check your vector database and configuration.")
    
    # Run the Flask app with auto-reloader disabled to prevent TensorFlow conflicts
    app.run(debug=True, port=5004, use_reloader=False)
