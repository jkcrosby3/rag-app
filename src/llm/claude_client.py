"""Claude LLM client for the RAG system.

This module provides a client for interacting with Anthropic's Claude models
to generate responses based on retrieved documents.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Union, Any
from threading import Lock

import anthropic

# Import the semantic cache
from src.llm.semantic_cache import get_response, set_response, get_stats as get_cache_stats

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """Metaclass for implementing the Singleton pattern."""
    
    _instances = {}
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        """Ensure only one instance of a class exists."""
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


# Constants for default values
DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20241022"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# Default system prompt for the Claude model
DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant that answers questions based on the provided documents.
Your answers should:
1. Be accurate and based only on the provided context
2. Be comprehensive but concise
3. Include relevant quotes from the documents when appropriate
4. Acknowledge when information is not available in the provided context
5. Never make up information or use knowledge outside the provided context"""


class ClaudeClient(metaclass=SingletonMeta):
    """Client for interacting with Anthropic's Claude models.
    
    This class uses the Singleton pattern to ensure only one instance exists,
    keeping the API client loaded in memory between queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME
    ):
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY env var
            model_name: Name of the Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set the ANTHROPIC_API_KEY environment variable.")
            
        self.model_name = model_name
        
        # Initialize client with API version header
        if self.api_key:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers={
                    "anthropic-version": "2023-06-01"
                }
            )
        else:
            self.client = None
        
        logger.info(f"Initialized Claude client with model {model_name}")
    
    def generate_response(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """Generate a response based on the query and retrieved documents.
        
        Args:
            query: User's query
            retrieved_documents: List of retrieved documents
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise ValueError("Claude client not initialized. Please provide a valid API key.")
            
        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Check if response is in semantic cache
        cached_response = get_response(query, retrieved_documents, system_prompt)
        if cached_response is not None:
            logger.info(f"Using semantically cached response for query: {query[:50]}...")
            return cached_response
            
        # Format retrieved documents
        context = self._format_documents(retrieved_documents)
        
        # Create the message content
        user_message = f"""I need information about the following query:

Query: {query}

Here are the relevant documents:

{context}

Based on these documents, please provide a comprehensive answer to my query."""
        
        try:
            start_time = time.time()
            
            # Call the Claude API
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response text
            answer = response.content[0].text
            
            # Cache the response in the semantic cache
            set_response(query, retrieved_documents, answer, system_prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response from Claude in {generation_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response from Claude: {str(e)}")
            raise
    
    def generate_response_direct(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """Generate a direct response without retrieved documents.
        
        This method is used for generating clarifying questions or processing
        simple queries that don't require document retrieval.
        
        Args:
            system_prompt: System prompt to guide the model's behavior
            user_message: User's message/query
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise ValueError("Claude client not initialized. Please provide a valid API key.")
        
        try:
            start_time = time.time()
            
            # Call the Claude API
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response text
            answer = response.content[0].text
            
            generation_time = time.time() - start_time
            logger.info(f"Generated direct response from Claude in {generation_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating direct response from Claude: {str(e)}")
            raise
    
    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for inclusion in the prompt.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted document text
        """
        if not documents:
            return "No relevant documents found."
            
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            # Extract document metadata
            topic = doc.get("metadata", {}).get("topic", "unknown")
            file_name = doc.get("metadata", {}).get("file_name", "unknown")
            similarity = doc.get("similarity", 0.0)
            
            # Format document text
            doc_text = f"Document {i+1} [Topic: {topic}, Source: {file_name}, Relevance: {similarity:.2f}]:\n{doc.get('text', '')}\n"
            formatted_docs.append(doc_text)
            
        return "\n".join(formatted_docs)
