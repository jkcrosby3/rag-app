"""LLM integration module for the RAG system.

This module provides functionality to interact with Large Language Models
for generating responses based on retrieved documents.

It includes a factory pattern for easy switching between different LLM providers
and a base class for implementing custom LLM clients.
"""

from .base_llm_client import BaseLLMClient
from .llm_factory import get_llm_client, get_available_llm_types, register_llm_client

# Re-export for easier imports
__all__ = [
    'BaseLLMClient',
    'get_llm_client',
    'get_available_llm_types',
    'register_llm_client'
]
