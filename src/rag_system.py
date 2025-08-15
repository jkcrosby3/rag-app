"""
Complete RAG (Retrieval-Augmented Generation) system implementation.

This module integrates document retrieval with LLM generation to create
a complete RAG system that can use either FAISS or Elasticsearch as the
vector database backend.

The RAG system follows these steps:
1. Process a user query
2. Generate embeddings for the query
3. Retrieve relevant documents from the vector database
4. Format the retrieved documents as context
5. Generate a response using an LLM with the context

This implementation includes advanced features like:
- Multiple vector database backends (FAISS, Elasticsearch)
- Quantized embedding models for faster inference
- Semantic caching for similar queries
- Performance monitoring and optimization
"""
import logging
import os
import sys
import argparse
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from dotenv import load_dotenv
from src.security.clearance_manager import ClearanceManager

# Apply huggingface_hub compatibility patch before importing sentence-transformers
# This ensures compatibility between different versions of huggingface_hub
import importlib.util
if importlib.util.find_spec("huggingface_hub"):
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        # Apply monkey patch for cached_download
        from src.utils.hf_compatibility import cached_download
        huggingface_hub.cached_download = cached_download
        logging.info("Applied huggingface_hub compatibility patch for cached_download")

# Add the project root to the Python path to fix import issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .embeddings.generator import EmbeddingGenerator
from .embeddings.quantized_generator import QuantizedEmbeddingGenerator
from .vector_db.faiss_db import FAISSVectorDB
from src.llm.llm_factory import get_llm_client, get_available_llm_types
from src.preloader import preloader, preload_models

# Import advanced caching modules
from src.llm.semantic_cache import get_stats as get_semantic_cache_stats, semantic_cache
from src.embeddings.model_cache import get_stats as get_embedding_cache_stats, get_cache_for_model
from src.conversation_manager import ConversationManager

# Conditionally import Elasticsearch
try:
    from src.vector_db.elasticsearch_db import ElasticsearchVectorDB
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system integrating retrieval and generation.
    
    This class provides a unified interface for document retrieval and LLM generation,
    with support for multiple vector database backends, quantized embeddings, and
    advanced caching strategies for optimal performance.
    
    Features:
    - Multiple vector database backends (FAISS, Elasticsearch)
    - Quantized embedding generation for faster inference
    - Semantic caching for LLM responses
    - Persistent disk caching for embeddings
    - Parallel processing for batch operations
    """

    def __init__(
        self,
        vector_db_type: Literal["faiss", "elasticsearch"] = "faiss",
        vector_db_path: Optional[Union[str, Path]] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_api_key: Optional[str] = None,
        llm_model_name: str = "claude-3-5-sonnet-20241022",
        llm_type: str = "claude",  # Default to Claude
        llm_config: Optional[Dict[str, Any]] = None,  # Additional LLM-specific config
        es_config: Optional[Dict[str, Any]] = None,
        use_preloader: bool = True,
        use_quantized_embeddings: bool = True,  # Default to quantized for better performance
        quantization_type: str = "int8",
        semantic_similarity_threshold: float = 0.75,  # Lower threshold for better semantic cache hits
        user_id: Optional[str] = None,  # User identifier for clearance verification
        clearance_manager: Optional[ClearanceManager] = None  # Clearance manager instance
    ):  # Add user_id parameter for clearance verification
        """Initialize the RAG system.

        Args:
            vector_db_type: Type of vector database to use ('faiss' or 'elasticsearch')
            vector_db_path: Path to the FAISS vector database (only for FAISS)
            embedding_model_name: Name of the embedding model to use
            llm_api_key: API key for the LLM service
            llm_model_name: Name of the LLM model to use
            es_config: Elasticsearch configuration (only for Elasticsearch)
            use_preloader: Whether to use the preloader for faster initialization
            use_quantized_embeddings: Whether to use quantized embedding generation for faster inference
            quantization_type: Type of quantization to use ('int8' or 'int4')
        """
        self.vector_db_type = vector_db_type
        start_time = time.time()
        
        # Create cache directory if it doesn't exist
        cache_dir = Path(project_root) / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store settings
        self.use_quantized_embeddings = use_quantized_embeddings
        self.quantization_type = quantization_type
        self.embedding_model_name = embedding_model_name
        
        # Configure semantic cache with the specified threshold
        semantic_cache.similarity_threshold = semantic_similarity_threshold
        logger.info(f"Configured semantic cache with similarity threshold: {semantic_similarity_threshold}")
        
        # Check if models are already preloaded
        if use_preloader and preloader.is_preloading_complete():
            logger.info("Using preloaded models")
            # If using quantized embeddings, we can't use the preloaded standard model
            if use_quantized_embeddings:
                logger.info(f"Initializing quantized embedding generator with {quantization_type} quantization")
                self.embedding_generator = QuantizedEmbeddingGenerator(
                    model_name=embedding_model_name,
                    quantization_type=quantization_type
                )
            else:
                self.embedding_generator = preloader.get_embedding_generator()
            # Get LLM client from preloader or create new one
            try:
                self.llm_client = preloader.get_llm_client()
            except (AttributeError, NotImplementedError):
                # Fallback to creating a new client if preloader doesn't support it
                llm_config = llm_config or {}
                llm_config.update({
                    'api_key': llm_api_key,
                    'model_name': llm_model_name
                })
                self.llm_client = get_llm_client(llm_type, **llm_config)
        elif use_preloader:
            # Start preloading in background if not already done
            logger.info("Starting model preloading in background")
            preload_models(embedding_model_name=embedding_model_name, llm_model_name=llm_model_name)
            
            # Initialize our own instances for immediate use
            if use_quantized_embeddings:
                logger.info(f"Initializing quantized embedding generator with {quantization_type} quantization")
                self.embedding_generator = QuantizedEmbeddingGenerator(
                    model_name=embedding_model_name,
                    quantization_type=quantization_type
                )
            else:
                self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
            
            # Initialize LLM client using factory
            llm_config = llm_config or {}
            llm_config.update({
                'api_key': llm_api_key,
                'model_name': llm_model_name
            })
            self.llm_client = get_llm_client(llm_type, **llm_config)
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(self.llm_client)
        else:
            # Initialize without preloading
            if use_quantized_embeddings:
                logger.info(f"Initializing quantized embedding generator with {quantization_type} quantization")
                self.embedding_generator = QuantizedEmbeddingGenerator(
                    model_name=embedding_model_name,
                    quantization_type=quantization_type
                )
            else:
                self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
            
            # Initialize LLM client using factory
            llm_config = llm_config or {}
            llm_config.update({
                'api_key': llm_api_key,
                'model_name': llm_model_name
            })
            self.llm_client = get_llm_client(llm_type, **llm_config)
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(self.llm_client)
        
        # Initialize clearance manager
        self.clearance_manager = clearance_manager or ClearanceManager()
        self.user_id = user_id
        
        # Initialize vector database
        if vector_db_type == "faiss":
            if not vector_db_path:
                vector_db_path = project_root / "data" / "vector_db"
            self.vector_db = FAISSVectorDB(index_path=vector_db_path)
            logger.info(f"Initialized FAISS vector database with {self.vector_db.get_document_count()} documents")
        elif vector_db_type == "elasticsearch":
            if not ELASTICSEARCH_AVAILABLE:
                raise ImportError("Elasticsearch support not available. Please install elasticsearch package.")
            
            es_config = es_config or {}
            self.vector_db = ElasticsearchVectorDB(
                index_name=es_config.get("index_name", "rag_documents"),
                es_hosts=es_config.get("es_hosts"),
                es_api_key=es_config.get("es_api_key"),
                es_cloud_id=es_config.get("es_cloud_id"),
                dimension=384,  # Default for all-MiniLM-L6-v2
                similarity_metric=es_config.get("similarity_metric", "cosine")
            )
            logger.info(f"Initialized Elasticsearch vector database with {self.vector_db.get_document_count()} documents")
        else:
            raise ValueError(f"Unsupported vector database type: {vector_db_type}")
        
        init_time = time.time() - start_time
        logger.info(f"RAG system initialized in {init_time:.2f} seconds")
    
    def process_query(
        self,
        query: str,
        top_k: int = 5,
        filter_topics: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_cache: bool = True,
        return_cache_stats: bool = True,
        custom_filter: Optional[callable] = None,
        test_mode: bool = False
    ):  # Add test_mode parameter to skip LLM generation  # Removed max_classification parameter as it's now handled by clearance manager -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline.
        
        This method implements the core RAG functionality by:
        1. Converting the text query into an embedding vector
        2. Using the embedding to search the vector database for relevant documents
        3. Filtering documents by topic if requested
        4. Checking if a semantically similar query has been cached
        5. Either returning the cached response or generating a new one with the LLM
        6. Collecting performance metrics throughout the process
        
        Args:
            query: User's natural language query text
            top_k: Number of most relevant documents to retrieve from the vector database
            filter_topics: Optional list of topics to filter documents by (e.g., ['glass_steagall', 'new_deal'])
            system_prompt: Optional system prompt to guide the LLM's behavior
            temperature: Controls randomness in LLM generation (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum number of tokens for the LLM to generate in response
            use_cache: Whether to check the semantic cache before generating a new response
            return_cache_stats: Whether to include detailed cache statistics in the result
            custom_filter: Optional custom filter function to apply to retrieved documents
            
        Returns:
            Dictionary with query results and detailed performance metrics including:
            - query: The original query
            - retrieved_documents: List of relevant documents with similarity scores
            - response: The generated or cached LLM response
            - processing_time: Total time to process the query (seconds)
            - embedding_time: Time spent generating the query embedding
            - retrieval_time: Time spent retrieving documents
            - generation_time: Time spent generating the LLM response
            - cache_hit: Whether the response was from cache
            - cache_info: Detailed cache statistics (if requested)
        """
        start_time = time.time()
        cache_info = {}
        
        logger.info(f"Processing query: {query}")
        
        # Generate embedding for the query
        embedding_start = time.time()
        query_embedding = self.embedding_generator.generate_embedding(query)
        embedding_time = time.time() - embedding_start
        
        # Retrieve relevant documents
        retrieval_start = time.time()
        
        # Get user's clearance level
        user_clearance = self.clearance_manager.get_user_clearance(self.user_id)
        if not user_clearance:
            raise ValueError(f"User {self.user_id} does not have valid clearance")

        # Search vector database with classification filtering
        retrieved_documents = self.vector_db.search(
            query_embedding, 
            k=top_k,
            max_classification=user_clearance
        )
        
        # Apply clearance-based filtering and redaction
        if self.user_id:
            retrieved_documents = self.clearance_manager.redact_content(
                retrieved_documents,
                self.user_id
            )    
            # Filter by topics if specified
            if filter_topics:
                filtered_docs = []
                for doc in retrieved_documents:
                    topic = doc.get("metadata", {}).get("topic")
                    if topic in filter_topics:
                        filtered_docs.append(doc)
                retrieved_documents = filtered_docs[:top_k]
            
            # Apply custom filter if provided
            if custom_filter:
                filtered_docs = []
                for doc in retrieved_documents:
                    if custom_filter(doc):
                        filtered_docs.append(doc)
                retrieved_documents = filtered_docs[:top_k]
        retrieval_time = time.time() - retrieval_start
        
        # If in test mode, skip LLM generation and return documents directly
        if test_mode:
            logger.info("Test mode: Skipping LLM generation")
            response = {
                "query": query,
                "documents": retrieved_documents
            }
            return response

        # Check if we have a semantic cache hit before generating a response
        cache_hit = False
        if use_cache:
            cached_response = semantic_cache.get(query, retrieved_documents, system_prompt)
            if cached_response is not None:
                response = cached_response
                cache_hit = True
                logger.info(f"Using cached response for query: {query[:50]}...")
                generation_time = 0.0  # No generation time for cache hits
            else:
                # Generate response using the LLM
                generation_start = time.time()
                response = self.llm_client.generate_response(
                    query=query,
                    retrieved_documents=retrieved_documents,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                generation_time = time.time() - generation_start
        else:
            # Generate response without using cache
            generation_start = time.time()
            response = self.llm_client.generate_response(
                query=query,
                retrieved_documents=retrieved_documents,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            generation_time = time.time() - generation_start
        
        # Get detailed cache statistics
        if return_cache_stats:
            semantic_stats = get_semantic_cache_stats()
            embedding_stats = get_embedding_cache_stats(self.embedding_model_name)
            
            # Add cache hit information
            if use_cache:
                semantic_stats['current_query_cache_hit'] = cache_hit
            
            cache_info['semantic_cache'] = semantic_stats
            cache_info['embedding_cache'] = embedding_stats
        
        processing_time = time.time() - start_time
        
        # Prepare result with detailed information
        result = {
            'query': query,
            'retrieved_documents': retrieved_documents,
            'response': response,
            'processing_time': processing_time,
            'embedding_time': embedding_time,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'cache_hit': cache_hit if use_cache else False,
            'cache_info': cache_info if return_cache_stats else None,
            'quantized_model': self.use_quantized_embeddings,
            'quantization_type': self.quantization_type if self.use_quantized_embeddings else None,
            'embedding_model': self.embedding_model_name,
            'semantic_threshold': semantic_cache.similarity_threshold
        }
        
        logger.info(f"Retrieved {len(retrieved_documents)} relevant documents")
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        if cache_hit:
            logger.info("Response served from semantic cache")
        
        return result

    def process_conversational_query(
        self,
        query: str,
        conversation_history: List[Dict],
        top_k: int = 5,
        filter_topics: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_cache: bool = True,
        return_cache_stats: bool = True,
        custom_filter: Optional[callable] = None,
        enable_clarification: bool = True
    ) -> Dict[str, Any]:
        """Process a query in a conversational context, potentially asking clarifying questions.
        
        This method extends the standard process_query method by adding:
        1. Conversation history tracking
        2. Ambiguity detection
        3. Clarifying question generation when needed
        
        Args:
            query: User's natural language query text
            conversation_history: List of previous conversation turns
            top_k: Number of most relevant documents to retrieve
            filter_topics: Optional list of topics to filter documents by
            system_prompt: Optional system prompt to guide the LLM's behavior
            temperature: Controls randomness in LLM generation
            max_tokens: Maximum number of tokens for the LLM to generate
            use_cache: Whether to check the semantic cache
            return_cache_stats: Whether to include detailed cache statistics
            custom_filter: Optional custom filter function for documents
            enable_clarification: Whether to enable clarifying questions
            
        Returns:
            Dictionary with query results including:
            - query: The original query
            - response: The generated response or clarifying question
            - needs_clarification: Boolean indicating if this is a clarifying question
            - retrieved_documents: List of relevant documents (if not a clarification)
            - processing_time: Total time to process the query
            - other performance metrics
        """
        start_time = time.time()
        logger.info(f"Processing conversational query: {query}")
        
        # Check if query needs clarification
        if enable_clarification:
            needs_clarification, clarifying_question = self.conversation_manager.analyze_query_ambiguity(
                query=query,
                conversation_history=conversation_history
            )
            
            if needs_clarification and clarifying_question:
                logger.info(f"Generated clarifying question: {clarifying_question}")
                
                # Return the clarifying question without document retrieval
                return {
                    'query': query,
                    'response': clarifying_question,
                    'needs_clarification': True,
                    'processing_time': time.time() - start_time,
                    'retrieved_documents': [],
                    'embedding_time': 0,
                    'retrieval_time': 0,
                    'generation_time': time.time() - start_time,
                    'cache_hit': False
                }
        
        # If no clarification needed, process normally but with conversation context
        # First retrieve documents as in the regular process_query method
        embedding_start = time.time()
        query_embedding = self.embedding_generator.generate_embedding(query)
        embedding_time = time.time() - embedding_start
        
        # Retrieve relevant documents
        retrieval_start = time.time()
        
        # Handle different vector database implementations
        if self.vector_db_type == "faiss":
            retrieved_documents = self.vector_db.search(
                query_embedding, 
                k=top_k
            )
            
            # Filter by topics if specified
            if filter_topics:
                filtered_docs = []
                for doc in retrieved_documents:
                    topic = doc.get("metadata", {}).get("topic")
                    if topic in filter_topics:
                        filtered_docs.append(doc)
                retrieved_documents = filtered_docs[:top_k]
            
            # Apply custom filter if provided
            if custom_filter:
                filtered_docs = []
                for doc in retrieved_documents:
                    if custom_filter(doc):
                        filtered_docs.append(doc)
                retrieved_documents = filtered_docs[:top_k]
        else:
            # Elasticsearch implementation
            filter_query = None
            if filter_topics:
                filter_query = {
                    "terms": {
                        "metadata.topic": filter_topics
                    }
                }
            retrieved_documents = self.vector_db.search(
                query_embedding, 
                k=top_k,
                filter_query=filter_query
            )
        retrieval_time = time.time() - retrieval_start
        
        # Generate response using conversation context
        generation_start = time.time()
        response = self.conversation_manager.generate_response_with_context(
            query=query,
            retrieved_documents=retrieved_documents,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        generation_time = time.time() - generation_start
        
        # Get detailed cache statistics
        cache_info = {}
        if return_cache_stats:
            semantic_stats = get_semantic_cache_stats()
            embedding_stats = get_embedding_cache_stats(self.embedding_model_name)
            cache_info['semantic_cache'] = semantic_stats
            cache_info['embedding_cache'] = embedding_stats
        
        processing_time = time.time() - start_time
        
        # Prepare result with detailed information
        result = {
            'query': query,
            'retrieved_documents': retrieved_documents,
            'response': response,
            'needs_clarification': False,
            'processing_time': processing_time,
            'embedding_time': embedding_time,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'cache_hit': False,  # We don't use cache for conversational responses
            'cache_info': cache_info if return_cache_stats else None
        }
        
        logger.info(f"Retrieved {len(retrieved_documents)} relevant documents for conversational query")
        logger.info(f"Total processing time for conversational query: {processing_time:.2f} seconds")


def main():
    """Run the RAG system from the command line."""
    from dotenv import load_dotenv
    from src.security.clearance_manager import ClearanceManager

    # Load environment variables from .env file
    load_dotenv()
    
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='RAG System')
    parser.add_argument('query', type=str, help='Query to process')
    parser.add_argument('--vector-db', type=str, choices=['faiss', 'elasticsearch'], default='faiss',
                        help='Vector database backend to use')
    parser.add_argument('--vector-db-path', type=str, help='Path to FAISS vector database')
    parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--topics', type=str, help='Comma-separated list of topics to filter by')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for LLM generation')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens for LLM generation')
    parser.add_argument('--es-index', type=str, default='rag_documents', help='Elasticsearch index name')
    parser.add_argument('--es-hosts', type=str, help='Comma-separated list of Elasticsearch hosts')
    parser.add_argument('--no-preload', action='store_true', help='Disable model preloading')
    parser.add_argument('--quantized', action='store_true', help='Use quantized embedding model for faster inference')
    parser.add_argument('--quantization-type', type=str, choices=['int8', 'int4'], default='int8',
                        help='Type of quantization to use for the embedding model')
    
    args = parser.parse_args()
    
    # Check for API key
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        logger.error("Please set it with: set ANTHROPIC_API_KEY=your_api_key")
        return 1
    
    # Set up Elasticsearch config if needed
    es_config = None
    if args.vector_db == "elasticsearch":
        es_config = {
            "index_name": args.es_index,
            "es_hosts": args.es_hosts.split(",") if args.es_hosts else None
        }
    
    try:
        # Preload models in the background for faster subsequent queries
        if not args.no_preload:
            logger.info("Preloading models in background...")
            preload_models()
        
        # Initialize RAG system
        rag_system = RAGSystem(
            vector_db_type=args.vector_db,
            vector_db_path=args.vector_db_path,
            llm_api_key=anthropic_api_key,
            es_config=es_config,
            use_preloader=not args.no_preload,
            use_quantized_embeddings=args.quantized,
            quantization_type=args.quantization_type
        )
        
        # Process query
        filter_topics = args.topics.split(",") if args.topics else None
        result = rag_system.process_query(
            query=args.query,
            top_k=args.top_k,
            filter_topics=filter_topics,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            use_cache=True  # Always use cache for better performance
        )
        
        # Display results
        print("\n" + "="*80)
        print(f"QUERY: {result['query']}")
        print("="*80)
        
        print("\nRETRIEVED DOCUMENTS:")
        for i, doc in enumerate(result['retrieved_documents']):
            similarity = doc.get('similarity', 0)
            topic = doc.get('metadata', {}).get('topic', 'unknown')
            file_name = doc.get('metadata', {}).get('file_name', 'unknown')
            
            print(f"\n--- Document {i+1} (Similarity: {similarity:.4f}) ---")
            print(f"Topic: {topic}")
            print(f"Source: {file_name}")
            print("-"*40)
            print(f"{doc.get('text', '')[:200]}...")
        
        print("\n" + "="*80)
        print("GENERATED RESPONSE:")
        print("="*80)
        print(result['response'])
        print("\n" + "="*80)
        
        # Print timing information
        print("PERFORMANCE METRICS:")
        print(f"Total processing time: {result['processing_time']:.2f} seconds")
        print(f"Embedding generation: {result['embedding_time']:.2f} seconds")
        print(f"Document retrieval: {result['retrieval_time']:.2f} seconds")
        print(f"Response generation: {result['generation_time']:.2f} seconds")
                # Print cache information if available
        if result.get('cache_info'):
            cache_info = result['cache_info']
            print("\nCACHE STATISTICS:")
            if 'semantic_cache' in cache_info:
                sc = cache_info['semantic_cache']
                print(f"Semantic cache:")
                print(f"  Size: {sc.get('size', 0)}/{sc.get('max_size', 0)} entries")
                print(f"  Hit rate: {sc.get('hit_rate', 0):.2f}")
                print(f"  Exact hits: {sc.get('hits', 0)}")
                print(f"  Semantic hits: {sc.get('semantic_hits', 0)}")
                print(f"  Similarity threshold: {sc.get('similarity_threshold', 0):.2f}")
                print(f"  TTL: {sc.get('ttl_hours', 0)} hours")
            if 'embedding_cache' in cache_info:
                ec = cache_info['embedding_cache']
                print(f"Embedding cache:")
                print(f"  Size: {ec.get('size', 0)}/{ec.get('max_size', 0)} entries")
                print(f"  Hit rate: {ec.get('hit_rate', 0):.2f}")
                print(f"  Memory hit rate: {ec.get('memory_hit_rate', 0):.2f}")
                print(f"  Disk hit rate: {ec.get('disk_hit_rate', 0):.2f}")
                print(f"  Memory usage: {ec.get('estimated_memory_kb', 0):.2f} KB")
                print(f"  TTL: {ec.get('ttl_hours', 0)} hours")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error in RAG system: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
