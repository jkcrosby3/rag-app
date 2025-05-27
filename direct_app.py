"""
Direct RAG System Web Interface for testing source documents
"""
import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
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
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            query = request.form.get('query', '')
            enable_clarification = request.form.get('enable_clarification') == 'on'
            
            if not query:
                flash("Query cannot be empty", "danger")
                return redirect(url_for('index'))
            
            # Process the query
            result = rag_system.process_query(
                query=query,
                top_k=5,
                filter_topics=None,
                temperature=0.7,
                max_tokens=1000,
                use_cache=False
            )
            
            # Log the result for debugging
            logger.info(f"Query result keys: {result.keys()}")
            doc_count = len(result.get('retrieved_documents', []))
            logger.info(f"Retrieved {doc_count} documents for query: {query}")
            
            # Create a simplified result for storage
            chat_item = {
                'timestamp': datetime.datetime.now().isoformat(),
                'query': query,
                'response': result.get('response', ''),
                'is_clarification': False,
                'sources': []
            }
            
            # Add source documents
            retrieved_docs = result.get('retrieved_documents', [])
            
            # Ensure we have documents
            if not retrieved_docs:
                logger.warning("No documents retrieved for the query")
                # Add a placeholder source to indicate no documents were found
                chat_item['sources'].append({
                    'text': 'No source documents were found for this query.',
                    'similarity': 0.0,
                    'metadata': {'note': 'No matching documents found in the database'},
                    'location': ''
                })
            else:
                logger.info(f"Processing {len(retrieved_docs)} documents for the response")
                
                for i, doc in enumerate(retrieved_docs):
                    try:
                        # Extract page numbers or locations if available
                        location = ''
                        metadata = {}
                        
                        if 'metadata' in doc and isinstance(doc['metadata'], dict):
                            # Copy metadata with string conversion for safety
                            for k, v in doc['metadata'].items():
                                metadata[k] = str(v)
                            
                            # Set location based on available metadata
                            if 'page' in doc['metadata']:
                                location = f"Page {doc['metadata']['page']}"
                            elif 'chunk_index' in doc['metadata']:
                                location = f"Chunk {doc['metadata']['chunk_index']}"
                        
                        # Get document text safely
                        doc_text = ''
                        if 'text' in doc and doc['text']:
                            # Handle potential encoding issues
                            try:
                                doc_text = str(doc['text'])[:500] + '...'
                            except UnicodeEncodeError:
                                doc_text = doc['text'].encode('utf-8', errors='replace').decode('utf-8')[:500] + '...'
                        
                        # Create source object
                        source = {
                            'text': doc_text,
                            'similarity': float(doc.get('similarity', 0)),
                            'metadata': metadata,
                            'location': location,
                            'index': i + 1  # 1-based index for display
                        }
                        
                        chat_item['sources'].append(source)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {i}: {str(e)}")
                        # Add error information as a source
                        chat_item['sources'].append({
                            'text': f'Error processing document: {str(e)}',
                            'similarity': 0.0,
                            'metadata': {'error': str(e)},
                            'location': ''
                        })
                
                logger.info(f"Added {len(chat_item['sources'])} sources to the response")
            
            # Add to chat history
            session['chat_history'].append(chat_item)
            session.modified = True
            
            # Redirect to show updated history
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('index'))
    
    # Render the template for GET requests
    return render_template('direct.html', 
                           chat_history=session.get('chat_history', []),
                           system_error=system_error)

@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session."""
    session['chat_history'] = []
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Run a test to ensure document retrieval is working
    docs_working = test_document_retrieval()
    if not docs_working:
        logger.warning("Document retrieval test failed. Check your vector database and configuration.")
    
    # Create HTML template if it doesn't exist
    template_path = os.path.join("templates", "direct.html")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Direct RAG Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
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
            margin-left: auto;
        }
        .system-message .message-content {
            background-color: white;
            color: #212529;
            border: 1px solid #e9ecef;
        }
        .clarification .message-content {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .source-body {
            padding: 10px;
            border-top: 1px solid #dee2e6;
            display: none;
        }
        .source-body.show {
            display: block;
        }
        .no-sources {
            color: #6c757d;
            font-style: italic;
            text-align: center;
            padding: 20px 0;
        }
    </style>
</head>
<body>
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
                                    <div class="message system-message {% if item.is_clarification %}clarification{% endif %}">
                                        <div class="message-content">
                                            <div class="message-label">RAG System</div>
                                            {{ item.response }}
                                        </div>
                                    </div>
                                    
                                    <div class="sources-panel">
                                        <div class="sources-title">Source Documents ({{ item.sources|length }})</div>
                                        {% if item.sources|length > 0 %}
                                            {% for source in item.sources %}
                                            <div class="source-item">
                                                <div class="source-header" onclick="toggleSource('source-{{ loop.index }}-{{ loop.parent.loop.index }}')">
                                                    <div>
                                                        {% if source.metadata is defined and source.metadata.title is defined %}
                                                            {{ source.metadata.title }}
                                                        {% elif source.metadata is defined and source.metadata.filename is defined %}
                                                            {{ source.metadata.filename }}
                                                        {% elif source.metadata is defined and source.metadata.file_name is defined %}
                                                            {{ source.metadata.file_name }}
                                                        {% elif source.metadata is defined and source.metadata.source is defined %}
                                                            {{ source.metadata.source.split('\\')[-1] }}
                                                        {% else %}
                                                            Source {{ source.index|default(loop.index) }}
                                                        {% endif %}
                                                    </div>
                                                    <div>
                                                        {% if source.location is defined and source.location %}
                                                            <span class="badge bg-secondary me-1">{{ source.location }}</span>
                                                        {% endif %}
                                                        <span class="badge bg-primary">{{ (source.similarity * 100)|round(1) }}%</span>
                                                    </div>
                                                </div>
                                                <div id="source-{{ loop.index }}-{{ loop.parent.loop.index }}" class="source-body">
                                                    {% if source.metadata is defined and source.metadata|length > 0 %}
                                                    <div class="source-metadata mb-2">
                                                        <strong>Metadata:</strong>
                                                        <div class="row mt-1">
                                                            {% for key, value in source.metadata.items() %}
                                                            <div class="col-md-6 mb-1">
                                                                <strong>{{ key|replace('_', ' ')|title }}:</strong> {{ value }}
                                                            </div>
                                                            {% endfor %}
                                                        </div>
                                                    </div>
                                                    {% endif %}
                                                    
                                                    <div class="source-text">
                                                        {{ source.text }}
                                                    </div>
                                                </div>
                                            </div>
                                            {% endfor %}
                                        {% else %}
                                            <div class="no-sources">
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
                
                <div class="input-area">
                    <form method="POST" action="/">
                        <div class="row align-items-center">
                            <div class="col">
                                <textarea name="query" class="form-control" rows="2" placeholder="Ask a question about the Great Depression..." required></textarea>
                            </div>
                            <div class="col-auto">
                                <div class="form-check mb-2">
                                    <input class="form-check-input" type="checkbox" name="enable_clarification" id="enable_clarification_checkbox" checked>
                                    <label class="form-check-label" for="enable_clarification_checkbox">Enable Clarifying Questions</label>
                                </div>
                                <button type="submit" class="btn btn-primary w-100">Send</button>
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
            """)
    
    # Run the Flask app with auto-reloader disabled to prevent TensorFlow conflicts
    app.run(debug=True, port=5002, use_reloader=False)
