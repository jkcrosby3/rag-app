"""Base class for LLM clients in the RAG system.

This module defines the interface that all LLM clients must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
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
        pass
    
    @abstractmethod
    def generate_response_direct(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a direct response without retrieved documents.
        
        This is used for generating clarifying questions or processing
        simple queries that don't require document retrieval.
        
        Args:
            system_prompt: System prompt to guide the model's behavior
            user_message: User's message/query
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        pass
