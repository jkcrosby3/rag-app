"""
Basic RAG System Web Interface with side-by-side source documents
"""
import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template_string, request, session, redirect, url_for, flash
from flask_session import Session
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

# Configure server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session lifetime
app.config['SESSION_USE_SIGNER'] = True

# Initialize Flask-Session
Session(app)

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
        
    # Debug session
    logger.info(f"Session ID: {session.sid if hasattr(session, 'sid') else 'No SID'}")
    logger.info(f"Session contains chat_history: {'chat_history' in session}")
    logger.info(f"Chat history length: {len(session.get('chat_history', []))}")
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            query = request.form.get('query', '')
            
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
                        
                        # Get document text safely
                        doc_text = doc.get('text', '')[:500] + '...' if doc.get('text') else ''
                        
                        # Create source object
                        source = {
                            'text': doc_text,
                            'similarity': float(doc.get('similarity', 0)),
                            'metadata': metadata
                        }
                        
                        chat_item['sources'].append(source)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {i}: {str(e)}")
            
            # Add to chat history
            current_history = session.get('chat_history', [])
            current_history.append(chat_item)
            session['chat_history'] = current_history
            session.modified = True
            
            # Log the updated history
            logger.info(f"Updated chat history length: {len(session.get('chat_history', []))}")
            
            # Don't redirect, just render the template with updated history
            return render_template_string(template, chat_history=session.get('chat_history', []))
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('index'))
    
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
            }
            .source-body {
                padding: 10px;
                border-top: 1px solid #dee2e6;
                display: none;
            }
            .source-body.show {
                display: block;
            }
            .nav-links {
                position: fixed;
                right: 20px;
                z-index: 1000;
            }
            .nav-links.top {
                top: 20px;
            }
            .nav-links.bottom {
                bottom: 20px;
            }
            .nav-btn {
                display: inline-block;
                padding: 5px 10px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
            }
            .nav-btn:hover {
                background-color: #0056b3;
                color: white;
            }
        </style>
    </head>
    <body id="top">
        <!-- Navigation links -->
        <div class="nav-links top">
            <a href="#bottom" class="nav-btn">Bottom</a>
        </div>
        <div class="nav-links bottom">
            <a href="#top" class="nav-btn">Top</a>
        </div>
        
        <div class="container">
            <div class="row">
                <div class="col-12">
                    <div class="chat-container">
                        <div class="chat-header">
                            <h3>RAG System with Side-by-Side Source Documents</h3>
                            <div class="d-flex align-items-center gap-2">
                                <div id="loading-indicator" class="d-none">
                                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <span class="ms-1 small">Processing...</span>
                                </div>
                                <form method="POST" action="/new_session" class="d-inline">
                                    <button type="submit" class="btn btn-sm btn-outline-danger">New Conversation</button>
                                </form>
                            </div>
                        </div>
                        
                        <div class="alert alert-info">
                            <i class="bi bi-info-circle-fill me-2"></i>
                            <strong>Tip:</strong> Click on each source document header to expand and view the full content and metadata.
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
                                                    <div class="source-header" onclick="toggleSource('source-{{ loop.index }}')">
                                                        Source {{ loop.index }} - {{ (source.similarity * 100)|round(1) }}% match
                                                    </div>
                                                    <div id="source-{{ loop.index }}" class="source-body">
                                                        {% if source.metadata %}
                                                        <div class="source-metadata mb-2">
                                                            <strong>Metadata:</strong>
                                                            <ul class="list-unstyled">
                                                                {% for key, value in source.metadata.items() %}
                                                                <li><strong>{{ key }}:</strong> {{ value }}</li>
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
                        <form method="POST" action="/" id="query-form">
                            <div class="row align-items-center">
                                <div class="col">
                                    <textarea name="query" class="form-control" rows="2" placeholder="Ask a question about the Great Depression..." required></textarea>
                                </div>
                                <div class="col-auto">
                                    <button type="submit" class="btn btn-primary" id="submit-btn">Send</button>
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
            
            // Show loading indicator when form is submitted
            document.addEventListener('DOMContentLoaded', function() {
                const form = document.getElementById('query-form');
                const loadingIndicator = document.getElementById('loading-indicator');
                
                if (form && loadingIndicator) {
                    form.addEventListener('submit', function(e) {
                        // Prevent default form submission
                        e.preventDefault();
                        
                        // Show loading indicator
                        loadingIndicator.classList.remove('d-none');
                        
                        // Disable submit button to prevent multiple submissions
                        const submitBtn = document.getElementById('submit-btn');
                        if (submitBtn) {
                            submitBtn.disabled = true;
                            submitBtn.innerHTML = 'Processing...';
                        }
                        
                        // Get the form data
                        const formData = new FormData(form);
                        
                        // Submit the form using fetch
                        fetch('/', {
                            method: 'POST',
                            body: formData,
                            headers: {
                                'X-Requested-With': 'XMLHttpRequest'
                            }
                        })
                        .then(response => response.text())
                        .then(html => {
                            // Replace the entire page content with the new HTML
                            document.documentElement.innerHTML = html;
                            
                            // Reattach event listeners
                            const newForm = document.getElementById('query-form');
                            if (newForm) {
                                newForm.addEventListener('submit', arguments.callee);
                            }
                            
                            // Scroll to the bottom
                            window.scrollTo(0, document.body.scrollHeight);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('An error occurred while processing your request. Please try again.');
                            
                            // Reset the form
                            if (submitBtn) {
                                submitBtn.disabled = false;
                                submitBtn.innerHTML = 'Send';
                            }
                            loadingIndicator.classList.add('d-none');
                        });
                    });
                }
            });
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
    # Force session to be saved
    session.modified = True
    logger.info("Session cleared and reset")
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Ensure the session directory exists
    session_dir = os.path.join(tempfile.gettempdir(), 'flask_session')
    os.makedirs(session_dir, exist_ok=True)
    logger.info(f"Using session directory: {session_dir}")
    
    # Print instructions
    print("\n" + "=" * 80)
    print("RAG System with Side-by-Side Source Documents")
    print("=" * 80)
    print("1. Open your browser to http://localhost:5006")
    print("2. Ask questions about the Great Depression")
    print("3. Click on source document headers to view metadata and content")
    print("4. Use the Top/Bottom navigation links to move around the page")
    print("5. The 'New Conversation' button will reset the session")
    print("=" * 80 + "\n")
    
    # Run the Flask app with auto-reloader disabled to prevent TensorFlow conflicts
    app.run(debug=True, port=5006, use_reloader=False)
