"""
RAG System Web Interface

This module provides a Flask-based web interface for the RAG system,
allowing users to ask questions, view responses, and see supporting documents.
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

from flask import Flask, render_template, request, jsonify, session
import json
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

# Document types for filtering
DOCUMENT_TYPES = [
    "academic",
    "book",
    "essay",
    "article",
    "primary_source",
    "government_document"
]

# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # Convert any non-serializable object to string
            return str(obj)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "rag-app-secret-key")

# Configure Flask to use our custom JSON encoder
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Create a custom jsonify function that uses our encoder
def custom_jsonify(*args, **kwargs):
    return app.response_class(
        json.dumps(dict(*args, **kwargs), cls=CustomJSONEncoder),
        mimetype='application/json'
    )

# Ensure directories exist
os.makedirs("sessions", exist_ok=True)
os.makedirs("responses", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

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

def save_session(session_data, filename=None):
    """Save the current session to a JSON file."""
    if filename is None:
        # Generate filename based on session ID and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
    
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
            topics = request.form.getlist('topics')
            doc_types = request.form.getlist('doc_types')
            top_k = int(request.form.get('top_k', 5))
            temperature = float(request.form.get('temperature', 0.7))
            max_tokens = int(request.form.get('max_tokens', 1000))
            use_cache = request.form.get('use_cache') == 'on'
            enable_clarification = request.form.get('enable_clarification') == 'on'
            
            if not query:
                flash("Query cannot be empty", "error")
                return redirect(url_for('index'))
            
            # Create a filter function for document types if specified
            doc_type_filter = None
            if doc_types:
                def doc_type_filter(doc):
                    doc_type = doc.get('metadata', {}).get('document_type', '')
                    # If no document_type is specified, include it by default
                    if not doc_type:
                        return True
                    return doc_type in doc_types
            
            # Process the query with filters
            if enable_clarification:
                # Use conversational query with clarification
                conversation_history = []
                if 'chat_history' in session:
                    for item in session['chat_history']:
                        if 'query' in item and 'result' in item and 'response' in item['result']:
                            conversation_history.append({
                                'query': item['query'],
                                'response': item['result']['response']
                            })
                
                result = rag_system.process_conversational_query(
                    query=query,
                    conversation_history=conversation_history,
                    top_k=top_k,
                    filter_topics=topics if topics else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    custom_filter=doc_type_filter,
                    enable_clarification=enable_clarification
                )
            else:
                # Use standard query
                result = rag_system.process_query(
                    query=query,
                    top_k=top_k,
                    filter_topics=topics if topics else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_cache=use_cache,
                    custom_filter=doc_type_filter
                )
            
            # Create a simplified version of the result for storage
            safe_result = {
                'query': query,
                'response': str(result.get('response', '')),
                'processing_time': float(result.get('processing_time', 0)),
                'embedding_time': float(result.get('embedding_time', 0)),
                'retrieval_time': float(result.get('retrieval_time', 0)),
                'generation_time': float(result.get('generation_time', 0)),
                'cache_hit': bool(result.get('cache_hit', False)),
                'needs_clarification': bool(result.get('needs_clarification', False)),
                'retrieved_documents': []
            }
            
            # Add simplified documents to result
            retrieved_docs = result.get('retrieved_documents', [])
            for doc in retrieved_docs:
                safe_doc = {
                    'text': str(doc.get('text', ''))[:1000],  # Limit text length
                    'similarity': float(doc.get('similarity', 0)),
                    'metadata': {}
                }
                
                # Add metadata
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    for k, v in doc['metadata'].items():
                        safe_doc['metadata'][k] = str(v)
                
                safe_result['retrieved_documents'].append(safe_doc)
            
            # Add to chat history
            session['chat_history'].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'query': query,
                'result': safe_result,
                'is_clarification': safe_result.get('needs_clarification', False)
            })
            session.modified = True
            
            # Redirect to show updated history
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('index'))
    
    # Render the template for GET requests
    return render_template('index_new.html', 
                           topics=DEFAULT_TOPICS,
                           document_types=DOCUMENT_TYPES,
                           chat_history=session.get('chat_history', []),
                           system_error=system_error)

@app.route('/query', methods=['POST'])
def process_query():
    """Process a standard query and return the response."""
    if rag_system is None:
        return Response(
            json.dumps({"success": False, "error": system_error or "RAG system is not initialized"}),
            mimetype='application/json'
        )
    
    # Enable more detailed logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Log the request
    logger.debug(f"Received query request: {request.data}")
    
    try:
        # Get query parameters
        data = request.json
        logger.debug(f"Request JSON parsed successfully: {data}")
        
        query = data.get('query', '')
        topics = data.get('topics', [])
        doc_types = data.get('document_types', [])
        top_k = int(data.get('top_k', 5))
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 1000))
        use_cache = data.get('use_cache', True)
        
        logger.debug(f"Query parameters: query='{query}', topics={topics}, doc_types={doc_types}, top_k={top_k}, temp={temperature}")
        
        if not query:
            return Response(
                json.dumps({"success": False, "error": "Query cannot be empty"}),
                mimetype='application/json'
            )
        
        # Create a filter function for document types if specified
        doc_type_filter = None
        if doc_types:
            def doc_type_filter(doc):
                doc_type = doc.get('metadata', {}).get('document_type', '')
                # If no document_type is specified, include it by default
                if not doc_type:
                    return True
                return doc_type in doc_types
        
        # Process the query with filters
        logger.debug(f"Processing query with RAG system")
        result = rag_system.process_query(
            query=query,
            top_k=top_k,
            filter_topics=topics if topics else None,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            # Add custom filter function if document types are specified
            custom_filter=doc_type_filter
        )
        logger.debug(f"Query processed successfully")
        
        # Create a simplified version of the result for storage and response
        # Only include essential data that can be safely serialized
        safe_result = {
            'query': query,
            'response': str(result.get('response', '')),
            'processing_time': float(result.get('processing_time', 0)),
            'embedding_time': float(result.get('embedding_time', 0)),
            'retrieval_time': float(result.get('retrieval_time', 0)),
            'generation_time': float(result.get('generation_time', 0)),
            'cache_hit': bool(result.get('cache_hit', False)),
            'retrieved_documents': []
        }
        
        # Add simplified documents to result
        retrieved_docs = result.get('retrieved_documents', [])
        logger.debug(f"Processing {len(retrieved_docs)} retrieved documents")
        
        for i, doc in enumerate(retrieved_docs):
            try:
                safe_doc = {
                    'text': str(doc.get('text', ''))[:1000],  # Limit text length
                    'similarity': float(doc.get('similarity', 0)),
                    'metadata': {}
                }
                
                # Add metadata
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    for k, v in doc['metadata'].items():
                        safe_doc['metadata'][k] = str(v)
                
                safe_result['retrieved_documents'].append(safe_doc)
                logger.debug(f"Processed document {i+1}/{len(retrieved_docs)} successfully")
            except Exception as doc_error:
                logger.error(f"Error processing document {i}: {str(doc_error)}", exc_info=True)
                # Add a placeholder for the failed document
                safe_result['retrieved_documents'].append({
                    'text': f"[Error processing document: {str(doc_error)}]",
                    'similarity': 0.0,
                    'metadata': {'error': str(doc_error)}
                })
        
        # Add to chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'result': safe_result,
            'is_clarification': False
        })
        session.modified = True
        logger.debug("Added to chat history successfully")
        
        # Convert to JSON string manually
        try:
            response_json = json.dumps({"success": True, "result": safe_result})
            logger.debug("JSON serialization successful")
            return Response(response_json, mimetype='application/json')
        except Exception as json_error:
            logger.error(f"Error serializing to JSON: {str(json_error)}", exc_info=True)
            # Return a simplified error response
            return Response(
                json.dumps({"success": False, "error": f"JSON serialization error: {str(json_error)}"}),
                mimetype='application/json'
            )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return Response(
            json.dumps({"success": False, "error": str(e)}),
            mimetype='application/json'
        )

@app.route('/conversational_query', methods=['POST'])
def process_conversational_query():
    """Process a conversational query with potential clarifying questions."""
    if rag_system is None:
        return Response(
            json.dumps({"success": False, "error": system_error or "RAG system is not initialized"}),
            mimetype='application/json'
        )
    
    # Enable more detailed logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Log the request
    logger.debug(f"Received conversational query request: {request.data}")
    
    try:
        # Get query parameters
        data = request.json
        logger.debug(f"Request JSON parsed successfully: {data}")
        
        query = data.get('query', '')
        topics = data.get('topics', [])
        doc_types = data.get('document_types', [])
        top_k = int(data.get('top_k', 5))
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 1000))
        enable_clarification = data.get('enable_clarification', True)
        
        # Get conversation history from session or request
        conversation_history = data.get('conversation_history', [])
        if not conversation_history and 'chat_history' in session:
            # Convert session chat history to conversation history format
            for item in session['chat_history']:
                if 'query' in item and 'result' in item and 'response' in item['result']:
                    conversation_history.append({
                        'query': item['query'],
                        'response': item['result']['response']
                    })
        
        logger.debug(f"Conversational query parameters: query='{query}', topics={topics}, enable_clarification={enable_clarification}")
        logger.debug(f"Conversation history has {len(conversation_history)} turns")
        
        if not query:
            return Response(
                json.dumps({"success": False, "error": "Query cannot be empty"}),
                mimetype='application/json'
            )
        
        # Create a filter function for document types if specified
        doc_type_filter = None
        if doc_types:
            def doc_type_filter(doc):
                doc_type = doc.get('metadata', {}).get('document_type', '')
                # If no document_type is specified, include it by default
                if not doc_type:
                    return True
                return doc_type in doc_types
        
        # Process the conversational query
        logger.debug(f"Processing conversational query with RAG system")
        result = rag_system.process_conversational_query(
            query=query,
            conversation_history=conversation_history,
            top_k=top_k,
            filter_topics=topics if topics else None,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_filter=doc_type_filter,
            enable_clarification=enable_clarification
        )
        logger.debug(f"Conversational query processed successfully")
        
        # Create a simplified version of the result for storage and response
        safe_result = {
            'query': query,
            'response': str(result.get('response', '')),
            'needs_clarification': bool(result.get('needs_clarification', False)),
            'processing_time': float(result.get('processing_time', 0)),
            'embedding_time': float(result.get('embedding_time', 0)),
            'retrieval_time': float(result.get('retrieval_time', 0)),
            'generation_time': float(result.get('generation_time', 0)),
            'retrieved_documents': []
        }
        
        # Only add documents if this is not a clarifying question
        if not result.get('needs_clarification', False):
            # Add simplified documents to result
            retrieved_docs = result.get('retrieved_documents', [])
            logger.debug(f"Processing {len(retrieved_docs)} retrieved documents")
            
            for i, doc in enumerate(retrieved_docs):
                try:
                    safe_doc = {
                        'text': str(doc.get('text', ''))[:1000],  # Limit text length
                        'similarity': float(doc.get('similarity', 0)),
                        'metadata': {}
                    }
                    
                    # Add metadata
                    if 'metadata' in doc and isinstance(doc['metadata'], dict):
                        for k, v in doc['metadata'].items():
                            safe_doc['metadata'][k] = str(v)
                    
                    safe_result['retrieved_documents'].append(safe_doc)
                    logger.debug(f"Processed document {i+1}/{len(retrieved_docs)} successfully")
                except Exception as doc_error:
                    logger.error(f"Error processing document {i}: {str(doc_error)}", exc_info=True)
                    # Add a placeholder for the failed document
                    safe_result['retrieved_documents'].append({
                        'text': f"[Error processing document: {str(doc_error)}]",
                        'similarity': 0.0,
                        'metadata': {'error': str(doc_error)}
                    })
        
        # Add to chat history if not a clarifying question, or mark as clarification
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Add to session with appropriate type
        session['chat_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'result': safe_result,
            'is_clarification': safe_result['needs_clarification']
        })
        session.modified = True
        logger.debug("Added to chat history successfully")
        
        # Convert to JSON string manually
        try:
            response_json = json.dumps({"success": True, "result": safe_result})
            logger.debug("JSON serialization successful")
            return Response(response_json, mimetype='application/json')
        except Exception as json_error:
            logger.error(f"Error serializing to JSON: {str(json_error)}", exc_info=True)
            # Return a simplified error response
            return Response(
                json.dumps({"success": False, "error": f"JSON serialization error: {str(json_error)}"}),
                mimetype='application/json'
            )
    
    except Exception as e:
        logger.error(f"Error processing conversational query: {str(e)}", exc_info=True)
        return Response(
            json.dumps({"success": False, "error": str(e)}),
            mimetype='application/json'
        )
    
    # Enable more detailed logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Log the request
    logger.debug(f"Received query request: {request.data}")
    
    try:
        # Get query parameters
        data = request.json
        logger.debug(f"Request JSON parsed successfully: {data}")
        
        query = data.get('query', '')
        topics = data.get('topics', [])
        doc_types = data.get('document_types', [])
        top_k = int(data.get('top_k', 5))
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 1000))
        use_cache = data.get('use_cache', True)
        
        logger.debug(f"Query parameters: query='{query}', topics={topics}, doc_types={doc_types}, top_k={top_k}, temp={temperature}")
        
        if not query:
            return Response(
                json.dumps({"success": False, "error": "Query cannot be empty"}),
                mimetype='application/json'
            )
        
        # Create a filter function for document types if specified
        doc_type_filter = None
        if doc_types:
            def doc_type_filter(doc):
                doc_type = doc.get('metadata', {}).get('document_type', '')
                # If no document_type is specified, include it by default
                if not doc_type:
                    return True
                return doc_type in doc_types
        
        # Process the query with filters
        logger.debug(f"Processing query with RAG system")
        result = rag_system.process_query(
            query=query,
            top_k=top_k,
            filter_topics=topics if topics else None,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            # Add custom filter function if document types are specified
            custom_filter=doc_type_filter
        )
        logger.debug(f"Query processed successfully")
        
        # Create a simplified version of the result for storage and response
        # Only include essential data that can be safely serialized
        safe_result = {
            'query': query,
            'response': str(result.get('response', '')),
            'processing_time': float(result.get('processing_time', 0)),
            'embedding_time': float(result.get('embedding_time', 0)),
            'retrieval_time': float(result.get('retrieval_time', 0)),
            'generation_time': float(result.get('generation_time', 0)),
            'cache_hit': bool(result.get('cache_hit', False)),
            'retrieved_documents': []
        }
        
        # Add simplified documents to result
        retrieved_docs = result.get('retrieved_documents', [])
        logger.debug(f"Processing {len(retrieved_docs)} retrieved documents")
        
        for i, doc in enumerate(retrieved_docs):
            try:
                safe_doc = {
                    'text': str(doc.get('text', ''))[:1000],  # Limit text length
                    'similarity': float(doc.get('similarity', 0)),
                    'metadata': {}
                }
                
                # Add metadata
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    for k, v in doc['metadata'].items():
                        safe_doc['metadata'][k] = str(v)
                
                safe_result['retrieved_documents'].append(safe_doc)
                logger.debug(f"Processed document {i+1}/{len(retrieved_docs)} successfully")
            except Exception as doc_error:
                logger.error(f"Error processing document {i}: {str(doc_error)}", exc_info=True)
                # Add a placeholder for the failed document
                safe_result['retrieved_documents'].append({
                    'text': f"[Error processing document: {str(doc_error)}]",
                    'similarity': 0.0,
                    'metadata': {'error': str(doc_error)}
                })
        
        # Add to chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'result': safe_result
        })
        session.modified = True
        logger.debug("Added to chat history successfully")
        
        # Convert to JSON string manually
        try:
            response_json = json.dumps({"success": True, "result": safe_result})
            logger.debug("JSON serialization successful")
            return Response(response_json, mimetype='application/json')
        except Exception as json_error:
            logger.error(f"Error serializing to JSON: {str(json_error)}", exc_info=True)
            # Return a simplified error response
            return Response(
                json.dumps({"success": False, "error": f"JSON serialization error: {str(json_error)}"}),
                mimetype='application/json'
            )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return Response(
            json.dumps({"success": False, "error": str(e)}),
            mimetype='application/json'
        )

@app.route('/save_response', methods=['POST'])
def save_response_route():
    """Save a response to disk."""
    data = request.json
    result = data.get('result')
    
    if not result:
        return custom_jsonify({
            'success': False,
            'error': "No result data provided"
        })
    
    try:
        filepath = save_response(result)
        return custom_jsonify({
            'success': True,
            'filepath': filepath
        })
    except Exception as e:
        logger.error(f"Error saving response: {str(e)}", exc_info=True)
        return custom_jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/save_session', methods=['POST'])
def save_session_route():
    """Save the current session to disk."""
    if 'chat_history' not in session or not session['chat_history']:
        return custom_jsonify({
            'success': False,
            'error': "No chat history to save"
        })
    
    try:
        filepath = save_session({
            "session_id": session.get('session_id', f"session_{int(time.time())}"),
            "timestamp": datetime.datetime.now().isoformat(),
            "chat_history": session['chat_history']
        })
        return custom_jsonify({
            'success': True,
            'filepath': filepath
        })
    except Exception as e:
        logger.error(f"Error saving session: {str(e)}", exc_info=True)
        return custom_jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session."""
    session['chat_history'] = []
    session['session_id'] = f"session_{int(time.time())}"
    session.modified = True
    
    return jsonify({
        'success': True
    })

@app.route('/test_json', methods=['GET'])
def test_json():
    """Simple test endpoint to verify JSON response handling."""
    return Response(
        json.dumps({"success": True, "message": "JSON response working correctly"}),
        mimetype='application/json'
    )

@app.route('/cache_stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    try:
        from src.llm.semantic_cache import get_stats as get_semantic_cache_stats
        from src.embeddings.model_cache import get_stats as get_embedding_cache_stats
        
        semantic_stats = get_semantic_cache_stats()
        embedding_stats = get_embedding_cache_stats()
        
        return custom_jsonify({
            'success': True,
            'semantic_cache': semantic_stats,
            'embedding_cache': embedding_stats
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}", exc_info=True)
        return custom_jsonify({
            'success': False,
            'error': str(e)
        })

# Add a direct answer route for the specific question about the Great Depression
@app.route('/great_depression_causes')
def great_depression_causes():
    """Provide a direct answer about causes of the Great Depression."""
    causes = {
        "title": "Key Components That Led to the Great Depression",
        "causes": [
            "Stock Market Speculation and Crash (1929) - Excessive speculation in the stock market created an unsustainable bubble that collapsed dramatically in October 1929, wiping out billions in wealth.",
            "Banking System Weaknesses - Thousands of small, undercapitalized banks were vulnerable to runs and failures, with no federal deposit insurance to protect customers.",
            "Monetary Policy Mistakes - The Federal Reserve's contractionary policies and failure to serve as a lender of last resort exacerbated the economic downturn.",
            "Excessive Debt and Leverage - Both consumers and businesses had taken on high levels of debt during the 1920s boom, creating financial fragility.",
            "Structural Weaknesses in Agriculture - Farmers faced declining prices and overproduction issues throughout the 1920s, weakening a significant sector of the economy.",
            "Income Inequality - Wealth concentration meant that prosperity wasn't broadly shared, limiting consumer purchasing power and creating economic imbalances.",
            "International Financial Instability - The gold standard, war debts, and trade imbalances created rigidity in the international financial system.",
            "Trade Policy Failures - The Smoot-Hawley Tariff Act of 1930 raised import duties, triggering retaliatory measures that collapsed international trade.",
            "Inadequate Government Response - Initial government policies under President Hoover were insufficient to address the scale of the economic crisis.",
            "Psychological Factors - Loss of confidence in the economy created a self-reinforcing cycle of reduced spending, investment, and employment."
        ]
    }
    return render_template('causes.html', causes=causes)

if __name__ == "__main__":
    # Create HTML template for the causes page
    causes_template_path = os.path.join("templates", "causes.html")
    if not os.path.exists(causes_template_path):
        with open(causes_template_path, "w", encoding="utf-8") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ causes.title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .container {
            max-width: 800px;
        }
        .cause-item {
            margin-bottom: 15px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        h1 {
            margin-bottom: 30px;
        }
        .back-link {
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ causes.title }}</h1>
        
        <ol>
            {% for cause in causes.causes %}
            <li class="cause-item">{{ cause }}</li>
            {% endfor %}
        </ol>
        
        <div class="back-link">
            <a href="/" class="btn btn-primary">Back to RAG System</a>
        </div>
    </div>
</body>
</html>
            """)

if __name__ == "__main__":
    # Create HTML template if it doesn't exist
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .container {
            max-width: 1200px;
        }
        .query-container {
            margin-bottom: 20px;
        }
        .response-container {
            margin-bottom: 30px;
            padding: 15px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .document-container {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #e9ecef;
        }
        .metrics-container {
            margin-top: 10px;
            font-size: 0.9em;
        }
        .query-text {
            font-weight: bold;
        }
        .response-text {
            white-space: pre-wrap;
        }
        .sidebar {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
        }
        .history-item {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .history-item-clarification {
            border-left: 4px solid #ffc107;
            background-color: #fff8e1;
        }
        .clarification-input {
            border: 1px solid #ffc107;
            background-color: #fffdf7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="col-md-8">
                <h1 class="mb-4">RAG System Interface</h1>
                
                {% if system_error %}
                <div class="alert alert-danger" role="alert">
                    <h4 class="alert-heading">System Error</h4>
                    <p>{{ system_error }}</p>
                    <hr>
                    <p class="mb-0">Please check your API keys and configuration.</p>
                </div>
                {% endif %}
                
                <div class="query-container">
                    <h3>Ask a Question</h3>
                    <form method="POST" action="/">
                        <div class="mb-3">
                            <textarea name="query" class="form-control" rows="3" placeholder="Enter your question here..." required></textarea>
                        </div>
                        <div class="mb-3">
                            <h5>Example Questions:</h5>
                            <div class="d-grid gap-2">
                                <button type="button" class="btn btn-outline-secondary example-question-btn">What are the top 10 issues that enabled the Great Depression to happen?</button>
                                <button type="button" class="btn btn-outline-secondary example-question-btn">How did the banking crisis contribute to the Great Depression?</button>
                                <button type="button" class="btn btn-outline-secondary example-question-btn">What role did monetary policy play in the recovery?</button>
                            </div>
                            <div class="mt-3">
                                <a href="/great_depression_causes" class="btn btn-success w-100">View Key Components of the Great Depression</a>
                            </div>
                        </div>
                        
                        <!-- Hidden form fields for settings -->
                        <input type="hidden" name="top_k" id="top_k_hidden" value="5">
                        <input type="hidden" name="temperature" id="temperature_hidden" value="0.7">
                        <input type="hidden" name="max_tokens" id="max_tokens_hidden" value="1000">
                        
                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="use_cache" id="use_cache_checkbox" checked>
                                <label class="form-check-label" for="use_cache_checkbox">Use Cache</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="enable_clarification" id="enable_clarification_checkbox" checked>
                                <label class="form-check-label" for="enable_clarification_checkbox">Enable Clarifying Questions</label>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">Submit</button>
                    </form>
                    
                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ category }} mt-3">{{ message }}</div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                </div>
                
                <hr>
                
                <div id="response-area">
                    <h3>Conversation History</h3>
                    {% if not chat_history %}
                    <div class="alert alert-info">
                        No conversation history yet. Ask a question to get started!
                    </div>
                    {% else %}
                    {% for item in chat_history %}
                    <div class="history-item {% if item.is_clarification %}history-item-clarification{% endif %}">
                        <div class="query-text mb-2">{% if item.is_clarification %}Ambiguous Query{% else %}Question{% endif %}: {{ item.query }}</div>
                        <div class="response-text mb-3">
                            {% if item.is_clarification %}
                            <div class="alert alert-warning p-2">
                                <strong>Clarifying Question:</strong> {{ item.result.response }}
                            </div>
                            {% else %}
                            {{ item.result.response }}
                            {% endif %}
                        </div>
                        
                        <button class="btn btn-sm btn-outline-secondary mb-2" type="button" data-bs-toggle="collapse" data-bs-target="#documents-{{ loop.index }}">
                            View Supporting Documents
                        </button>
                        
                        <div class="collapse" id="documents-{{ loop.index }}">
                            <div class="document-container">
                                {% for doc in item.result.retrieved_documents %}
                                <div class="mb-3">
                                    <h6>Document {{ loop.index }} (Similarity: {{ "%.4f"|format(doc.similarity) }})</h6>
                                    <div><strong>Topic:</strong> {{ doc.metadata.topic if doc.metadata and doc.metadata.topic else "unknown" }}</div>
                                    <div><strong>Source:</strong> {{ doc.metadata.file_name if doc.metadata and doc.metadata.file_name else "unknown" }}</div>
                                    <div class="mt-2">{{ doc.text[:500] }}...</div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                        
                        <button class="btn btn-sm btn-outline-secondary mb-2 ms-2" type="button" data-bs-toggle="collapse" data-bs-target="#metrics-{{ loop.index }}">
                            View Performance Metrics
                        </button>
                        
                        <div class="collapse" id="metrics-{{ loop.index }}">
                            <div class="metrics-container">
                                <div><strong>Total Time:</strong> {{ "%.2f"|format(item.result.processing_time) }} seconds</div>
                                <div><strong>Embedding Time:</strong> {{ "%.2f"|format(item.result.embedding_time) }} seconds</div>
                                <div><strong>Retrieval Time:</strong> {{ "%.2f"|format(item.result.retrieval_time) }} seconds</div>
                                <div><strong>Generation Time:</strong> {{ "%.2f"|format(item.result.generation_time) }} seconds</div>
                            </div>
                        </div>
                        
                        <button class="btn btn-sm btn-outline-primary mb-2 ms-2 save-response-btn" data-index="{{ loop.index0 }}">
                            Save Response
                        </button>
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="sidebar">
                    <h3>Settings</h3>
                    
                    <div class="mb-3">
                        <label for="api-key" class="form-label">Anthropic API Key</label>
                        <input type="password" class="form-control" id="api-key" placeholder="Enter your API key">
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">Filter by Topics</label>
                        <div id="topics-container">
                            {% for topic in topics %}
                            <div class="form-check">
                                <input class="form-check-input topic-checkbox" type="checkbox" value="{{ topic }}" id="topic-{{ topic }}">
                                <label class="form-check-label" for="topic-{{ topic }}">
                                    {{ topic }}
                                </label>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label">Filter by Document Types</label>
                        <div id="document-types-container">
                            {% for doc_type in document_types %}
                            <div class="form-check">
                                <input class="form-check-input doc-type-checkbox" type="checkbox" value="{{ doc_type }}" id="doc-type-{{ doc_type }}">
                                <label class="form-check-label" for="doc-type-{{ doc_type }}">
                                    {{ doc_type }}
                                </label>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="top-k" class="form-label">Number of documents to retrieve</label>
                        <input type="range" class="form-range" id="top-k" min="1" max="10" value="5">
                        <div class="d-flex justify-content-between">
                            <span>1</span>
                            <span id="top-k-value">5</span>
                            <span>10</span>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="temperature" class="form-label">Temperature</label>
                        <input type="range" class="form-range" id="temperature" min="0" max="1" step="0.1" value="0.7">
                        <div class="d-flex justify-content-between">
                            <span>0.0</span>
                            <span id="temperature-value">0.7</span>
                            <span>1.0</span>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="max-tokens" class="form-label">Maximum tokens in response</label>
                        <input type="range" class="form-range" id="max-tokens" min="100" max="2000" step="100" value="1000">
                        <div class="d-flex justify-content-between">
                            <span>100</span>
                            <span id="max-tokens-value">1000</span>
                            <span>2000</span>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="use-cache" checked>
                            <label class="form-check-label" for="use-cache">Use Cache</label>
                        </div>
                        <div class="form-check mt-2">
                            <input class="form-check-input" type="checkbox" id="enable-clarification" checked>
                            <label class="form-check-label" for="enable-clarification">Enable Clarifying Questions</label>
                            <small class="form-text text-muted d-block">When enabled, the system may ask for clarification before answering ambiguous questions.</small>
                        </div>
                    </div>
                    <hr>
                    
                    <h3>Session Management</h3>
                    
                    <div class="d-grid gap-2">
                        <button id="save-session-btn" class="btn btn-outline-primary">Save Current Session</button>
                        <button id="new-session-btn" class="btn btn-outline-secondary">Start New Session</button>
                    </div>
                    
                    <hr>
                    
                    <h3>Cache Statistics</h3>
                    <button id="refresh-stats-btn" class="btn btn-sm btn-outline-secondary mb-2">Refresh Stats</button>
                    
                    <div id="cache-stats-container">
                        <div class="mb-2">
                            <h6>Semantic Cache</h6>
                            <div id="semantic-cache-stats">Loading...</div>
                        </div>
                        
                        <div>
                            <h6>Embedding Cache</h6>
                            <div id="embedding-cache-stats">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update range slider values
        document.getElementById('top-k').addEventListener('input', function() {
            document.getElementById('top-k-value').textContent = this.value;
        });
        
        document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('temperature-value').textContent = this.value;
        });
        
        document.getElementById('max-tokens').addEventListener('input', function() {
            document.getElementById('max-tokens-value').textContent = this.value;
        });
        
        // Submit query
        document.getElementById('submit-btn').addEventListener('click', function() {
            const query = document.getElementById('query-input').value.trim();
            if (!query) {
                alert('Please enter a question');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').classList.remove('d-none');
            document.getElementById('submit-btn').disabled = true;
            
            // Reset and start progress bar animation
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = '0%';
            
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(function() {
                // Increment progress, but slow down as it gets closer to 90%
                if (progress < 30) {
                    progress += 5; // Fast at first (embedding)
                } else if (progress < 60) {
                    progress += 3; // Medium speed (retrieval)
                } else if (progress < 90) {
                    progress += 0.5; // Slow down (generation)
                }
                // Cap at 90% - the last 10% happens when we get the response
                progress = Math.min(progress, 90);
                progressBar.style.width = progress + '%';
                
                // Add progress text to loading message
                if (progress < 30) {
                    document.querySelector('#loading span.ms-2').textContent = 'Processing query... (Creating embeddings)';
                } else if (progress < 60) {
                    document.querySelector('#loading span.ms-2').textContent = 'Processing query... (Retrieving documents)';
                } else {
                    document.querySelector('#loading span.ms-2').textContent = 'Processing query... (Generating response)';
                }
            }, 100);
            
            // Get selected topics
            const selectedTopics = [];
            document.querySelectorAll('.topic-checkbox:checked').forEach(function(checkbox) {
                selectedTopics.push(checkbox.value);
            });
            
            // Get selected document types
            const selectedDocTypes = [];
            document.querySelectorAll('.doc-type-checkbox:checked').forEach(function(checkbox) {
                selectedDocTypes.push(checkbox.value);
            });
            
            // Get other parameters
            const topK = parseInt(document.getElementById('top-k').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const maxTokens = parseInt(document.getElementById('max-tokens').value);
            const useCache = document.getElementById('use-cache').checked;
            
            // Get API key if provided
            const apiKey = document.getElementById('api-key').value.trim();
            
            // Determine which endpoint to use based on clarification setting
            const enableClarification = document.getElementById('enable-clarification').checked;
            const endpoint = enableClarification ? '/conversational_query' : '/query';
            
            // First test the JSON endpoint to make sure it's working
            fetch('/test_json')
                .then(response => response.json())
                .then(data => {
                    console.log('JSON test endpoint working:', data);
                    
                    // If test is successful, proceed with the actual query
                    // Prepare request data
                    const requestData = {
                        query: query,
                        topics: selectedTopics,
                        document_types: selectedDocTypes,
                        top_k: topK,
                        temperature: temperature,
                        max_tokens: maxTokens,
                        use_cache: useCache,
                        enable_clarification: enableClarification,
                        api_key: apiKey
                    };
                    
                    // Send the actual query
                    return fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestData)
                    });
                })
                .then(response => {
                    // Complete the progress bar when we get a response
                    clearInterval(progressInterval);
                    progressBar.style.width = '100%';
                    document.querySelector('#loading span.ms-2').textContent = 'Processing query... (Complete)';
                    
                    // Check if response is ok before trying to parse JSON
                    if (!response.ok) {
                        throw new Error(`Server responded with status: ${response.status}`);
                    }
                    
                    // Try to parse the response as text first to debug
                    return response.text().then(text => {
                        try {
                            // Try to parse as JSON
                            return JSON.parse(text);
                        } catch (e) {
                            console.error('Failed to parse response as JSON:', text);
                            throw new Error('Failed to parse server response as JSON. Response starts with: ' + text.substring(0, 50));
                        }
                    });
                })
            .then(data => {
                // Hide loading indicator after a short delay to show 100% completion
                setTimeout(() => {
                    document.getElementById('loading').classList.add('d-none');
                    document.getElementById('submit-btn').disabled = false;
                }, 500);
                
                if (data.success) {
                    // Check if this is a clarifying question
                    if (data.result && data.result.needs_clarification) {
                        // Handle clarifying question - don't reload page
                        const responseArea = document.getElementById('response-area');
                        
                        // Create a clarification container
                        const clarificationDiv = document.createElement('div');
                        clarificationDiv.className = 'alert alert-warning mt-3';
                        
                        // Add the clarifying question
                        const questionHeader = document.createElement('h5');
                        questionHeader.textContent = 'I need some clarification:';
                        
                        const questionText = document.createElement('p');
                        questionText.textContent = data.result.response;
                        
                        // Create a response input
                        const responseInput = document.createElement('textarea');
                        responseInput.className = 'form-control mt-2 mb-2';
                        responseInput.placeholder = 'Type your clarification here...';
                        responseInput.rows = 2;
                        
                        // Create a submit button
                        const submitButton = document.createElement('button');
                        submitButton.className = 'btn btn-primary';
                        submitButton.textContent = 'Submit Clarification';
                        submitButton.addEventListener('click', function() {
                            const clarification = responseInput.value.trim();
                            if (clarification) {
                                // Set the clarification as the new query
                                document.getElementById('query-input').value = clarification;
                                // Click the submit button to send the clarification
                                document.getElementById('submit-btn').click();
                                // Remove the clarification div
                                clarificationDiv.remove();
                            }
                        });
                        
                        // Assemble the clarification container
                        clarificationDiv.appendChild(questionHeader);
                        clarificationDiv.appendChild(questionText);
                        clarificationDiv.appendChild(responseInput);
                        clarificationDiv.appendChild(submitButton);
                        
                        // Insert at the top of the response area
                        responseArea.insertBefore(clarificationDiv, responseArea.firstChild);
                    } else {
                        // Regular response - reload the page to show updated history
                        window.location.reload();
                    }
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                // Stop progress animation
                clearInterval(progressInterval);
                
                // Hide loading indicator
                document.getElementById('loading').classList.add('d-none');
                document.getElementById('submit-btn').disabled = false;
                
                // Display error in UI instead of alert
                const responseArea = document.getElementById('response-area');
                const errorDiv = document.createElement('div');
                errorDiv.className = 'alert alert-danger mt-3';
                errorDiv.innerHTML = `<strong>Error:</strong> ${error.message || error}`;
                errorDiv.innerHTML += '<p class="mt-2">Please try again. If the problem persists, check the server logs or refresh the page.</p>';
                
                // Add a retry button
                const retryButton = document.createElement('button');
                retryButton.className = 'btn btn-outline-danger mt-2';
                retryButton.textContent = 'Retry Query';
                retryButton.addEventListener('click', function() {
                    document.getElementById('submit-btn').click();
                    errorDiv.remove();
                });
                
                errorDiv.appendChild(retryButton);
                
                // Insert at the top of the response area
                responseArea.insertBefore(errorDiv, responseArea.firstChild);
                
                // Log error to console for debugging
                console.error('Query error:', error);
            });
        });
        
        // Save response
        document.querySelectorAll('.save-response-btn').forEach(function(button) {
            button.addEventListener('click', function() {
                const index = parseInt(this.getAttribute('data-index'));
                const result = {{ chat_history|tojson }}[index].result;
                
                fetch('/save_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        result: result
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Response saved to ' + data.filepath);
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            });
        });
        
        // Save session
        document.getElementById('save-session-btn').addEventListener('click', function() {
            fetch('/save_session', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Session saved to ' + data.filepath);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Error: ' + error);
            });
        });
        
        // New session
        document.getElementById('new-session-btn').addEventListener('click', function() {
            if (confirm('Are you sure you want to start a new session? This will clear your current conversation history.')) {
                fetch('/new_session', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.reload();
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            }
        });
        
        // Refresh cache stats
        function refreshCacheStats() {
            fetch('/cache_stats')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const semanticCache = data.semantic_cache;
                    const embeddingCache = data.embedding_cache;
                    
                    let semanticHtml = `
                        <div>Size: ${semanticCache.size}/${semanticCache.max_size}</div>
                        <div>Hit rate: ${(semanticCache.hit_rate * 100).toFixed(2)}%</div>
                        <div>Exact hits: ${semanticCache.hits}</div>
                        <div>Semantic hits: ${semanticCache.semantic_hits}</div>
                    `;
                    
                    let embeddingHtml = `
                        <div>Size: ${embeddingCache.size}/${embeddingCache.max_size}</div>
                        <div>Hit rate: ${(embeddingCache.hit_rate * 100).toFixed(2)}%</div>
                        <div>Memory usage: ${embeddingCache.estimated_memory_kb.toFixed(2)} KB</div>
                    `;
                    
                    document.getElementById('semantic-cache-stats').innerHTML = semanticHtml;
                    document.getElementById('embedding-cache-stats').innerHTML = embeddingHtml;
                } else {
                    document.getElementById('semantic-cache-stats').textContent = 'Error: ' + data.error;
                    document.getElementById('embedding-cache-stats').textContent = 'Error: ' + data.error;
                }
            })
            .catch(error => {
                document.getElementById('semantic-cache-stats').textContent = 'Error: ' + error;
                document.getElementById('embedding-cache-stats').textContent = 'Error: ' + error;
            });
        }
        
        document.getElementById('refresh-stats-btn').addEventListener('click', refreshCacheStats);
        
        // Initial load of cache stats
        refreshCacheStats();
        
        // Example question buttons
        document.querySelectorAll('.example-question-btn').forEach(function(button) {
            button.addEventListener('click', function() {
                document.querySelector('textarea[name="query"]').value = this.textContent;
                // Scroll to the submit button
                document.querySelector('button[type="submit"]').scrollIntoView({ behavior: 'smooth' });
            });
        });
        
        // Update hidden form fields when settings change
        document.getElementById('top-k-range').addEventListener('input', function() {
            document.getElementById('top_k_hidden').value = this.value;
        });
        
        document.getElementById('temperature-range').addEventListener('input', function() {
            document.getElementById('temperature_hidden').value = this.value;
        });
        
        document.getElementById('max-tokens-range').addEventListener('input', function() {
            document.getElementById('max_tokens_hidden').value = this.value;
        });
        
        // Sync checkbox states
        document.getElementById('use-cache').addEventListener('change', function() {
            document.getElementById('use_cache_checkbox').checked = this.checked;
        });
        
        document.getElementById('enable-clarification').addEventListener('change', function() {
            document.getElementById('enable_clarification_checkbox').checked = this.checked;
        });
        
        // Add topic checkboxes to form on submit
        document.querySelector('form').addEventListener('submit', function(e) {
            // Get selected topics
            document.querySelectorAll('input[name="topic"]:checked').forEach(checkbox => {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'topics';
                input.value = checkbox.value;
                this.appendChild(input);
            });
            
            // Get selected document types
            document.querySelectorAll('input[name="doc_type"]:checked').forEach(checkbox => {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'doc_types';
                input.value = checkbox.value;
                this.appendChild(input);
            });
            
            // Add API key if provided
            const apiKey = document.getElementById('api-key').value.trim();
            if (apiKey) {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'api_key';
                input.value = apiKey;
                this.appendChild(input);
            }
        });
    </script>
</body>
</html>
            """)
    
    # Create directories for templates and static files
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Run the Flask app with auto-reloader disabled to prevent TensorFlow conflicts
    app.run(debug=True, port=5000, use_reloader=False)
