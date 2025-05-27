"""
Simplified RAG System Web Interface for testing clarifying questions
"""
import os
import sys
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
from src.rag_system import RAGSystem
from src.conversation_manager import ConversationManager
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

# Expanded topics for better organization
DEFAULT_TOPICS = [
    "glass_steagall",
    "new_deal", 
    "sec",
    "causes",
    "effects",
    "monetary_policy",
    "banking_crisis",
    "stock_market",
    "recovery",
    "international"
]

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
            if enable_clarification:
                # Use conversational query with clarification
                conversation_history = []
                if 'chat_history' in session:
                    for item in session['chat_history']:
                        if 'query' in item and 'response' in item:
                            conversation_history.append({
                                'query': item['query'],
                                'response': item['response']
                            })
                
                result = rag_system.process_conversational_query(
                    query=query,
                    conversation_history=conversation_history,
                    enable_clarification=True
                )
                
                # Check if clarification is needed
                needs_clarification = result.get('needs_clarification', False)
                
                # Create a simplified result for storage
                chat_item = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'query': query,
                    'response': result.get('response', ''),
                    'is_clarification': needs_clarification,
                    'sources': []
                }
                
                # Add source documents
                retrieved_docs = result.get('retrieved_documents', [])
                for doc in retrieved_docs:
                    # Extract page numbers or locations if available
                    location = ''
                    if 'metadata' in doc and isinstance(doc['metadata'], dict):
                        if 'page' in doc['metadata']:
                            location = f"Page {doc['metadata']['page']}"
                        elif 'chunk_index' in doc['metadata']:
                            location = f"Chunk {doc['metadata']['chunk_index']}"
                    
                    source = {
                        'text': str(doc.get('text', ''))[:500] + '...',  # Truncate for display
                        'similarity': float(doc.get('similarity', 0)),
                        'metadata': {},
                        'location': location
                    }
                    
                    # Add metadata
                    if 'metadata' in doc and isinstance(doc['metadata'], dict):
                        for k, v in doc['metadata'].items():
                            source['metadata'][k] = str(v)
                    
                    chat_item['sources'].append(source)
                
                # Add to chat history
                session['chat_history'].append(chat_item)
                session.modified = True
                
            else:
                # Use standard query with minimal parameters to ensure we get documents
                result = rag_system.process_query(
                    query=query,
                    top_k=5,
                    filter_topics=None,
                    temperature=0.7,
                    max_tokens=1000,
                    use_cache=False  # Disable cache to ensure we get fresh documents
                )
                
                # Log the result for debugging
                logger.info(f"Query result keys: {result.keys()}")
                doc_count = len(result.get('retrieved_documents', []))
                logger.info(f"Retrieved {doc_count} documents for query: {query}")
                
                # If no documents were retrieved, try a more general query
                if doc_count == 0:
                    logger.warning("No documents retrieved, trying with a more general query")
                    general_query = "Great Depression banking crisis"
                    result = rag_system.process_query(
                        query=general_query,
                        top_k=5,
                        filter_topics=None,
                        temperature=0.7,
                        max_tokens=1000,
                        use_cache=False
                    )
                    logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents for general query")
                
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
    
    # Add debug info to the template context
    debug_info = {
        'has_chat_history': bool(session.get('chat_history')),
        'chat_history_length': len(session.get('chat_history', [])),
        'has_sources': False
    }
    
    # Check if any chat items have sources
    for item in session.get('chat_history', []):
        if item.get('sources') and len(item.get('sources', [])) > 0:
            debug_info['has_sources'] = True
            debug_info['source_count'] = len(item.get('sources', []))
            debug_info['first_source'] = item.get('sources', [])[0] if item.get('sources') else None
            break
    
    logger.info(f"Debug info: {debug_info}")
    
    # Render the template for GET requests
    return render_template('simple.html', 
                           chat_history=session.get('chat_history', []),
                           system_error=system_error,
                           debug_info=debug_info)

@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session."""
    session['chat_history'] = []
    return jsonify({
        'success': True,
        'message': 'New session started'
    })

if __name__ == "__main__":
    # Create HTML template if it doesn't exist
    template_path = os.path.join("templates", "simple.html")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple RAG Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body {
            padding: 20px;
            background-color: #f5f5f5;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            flex: 1;
            overflow-y: auto;
            max-height: calc(100vh - 220px);
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
            border-bottom-right-radius: 2px;
        }
        .system-message .message-content {
            background-color: white;
            color: #212529;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 2px;
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
        .accordion-button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        .badge {
            font-size: 0.75rem;
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
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 100;
        }
        .sticky-bottom {
            position: sticky;
            bottom: 0;
            z-index: 100;
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
        <h1>Simple RAG Interface</h1>
        <p class="lead">Ask questions about the Great Depression with clarifying questions</p>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }} mt-3">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="row">
            <div class="col-md-12">
                <div class="chat-container">
                    <div class="chat-header">
                        <h3>RAG System with Source Documents</h3>
                        <form method="POST" action="/new_session" class="d-inline">
                            <button type="submit" class="btn btn-sm btn-outline-danger">New Conversation</button>
                        </form>
                    </div>
                    
                    <div class="chat-messages">
                        {% if chat_history %}
                            {% for item in chat_history %}
                                <!-- Message row with sources side by side -->
                                <div class="message-row">
                                    <!-- Message column -->
                                    <div class="message {% if loop.index % 2 == 0 %}system-message{% else %}user-message{% endif %} {% if item.is_clarification %}clarification{% endif %}">
                                        <div class="message-content">
                                            <div class="message-label">{% if loop.index % 2 == 0 %}RAG System{% else %}You{% endif %}</div>
                                            {% if loop.index % 2 == 0 %}
                                                {{ item.response }}
                                            {% else %}
                                                {{ item.query }}
                                            {% endif %}
                                        </div>
                                    </div>
                                    
                                    <!-- Sources panel -->
                                    {% if loop.index % 2 == 0 and item.sources is defined %}
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
                                    {% endif %}
                                </div>
                            {% endfor %}
                        {% else %}
                            <div class="text-center text-muted py-5">
                                <p>No conversation yet. Ask a question to get started!</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
                
                <div class="input-area sticky-bottom mt-3">
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
            """)
    
    # Add JavaScript for toggling source documents
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    # Add toggle function if it doesn't exist
    if "function toggleSource" not in template_content:
        toggle_script = """
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
        """
        template_content = template_content.replace("</body>", toggle_script)
        
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_content)
    
    # Run the Flask app with auto-reloader disabled to prevent TensorFlow conflicts
    app.run(debug=True, port=5001, use_reloader=False)
