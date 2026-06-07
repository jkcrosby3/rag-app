"""Factory for creating LLM client instances.

This module provides a factory class to create and manage different LLM clients
based on configuration.
"""
from typing import Dict, Type, Any, Optional
import logging
from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Registry of available LLM clients
_llm_clients: Dict[str, Type[BaseLLMClient]] = {}


def register_llm_client(name: str):
    """Decorator to register an LLM client class with a name.
    
    Args:
        name: The name to register the LLM client under
    """
    def decorator(cls: Type[BaseLLMClient]):
        if not issubclass(cls, BaseLLMClient):
            raise TypeError(f"{cls.__name__} is not a subclass of BaseLLMClient")
        _llm_clients[name.lower()] = cls
        logger.info(f"Registered LLM client: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_llm_client(
    llm_type: str,
    **kwargs: Any
) -> BaseLLMClient:
    """Create an instance of the specified LLM client.
    
    Args:
        llm_type: Type of LLM client to create (e.g., 'claude')
        **kwargs: Additional arguments to pass to the LLM client constructor
        
    Returns:
        An instance of the specified LLM client
        
    Raises:
        ValueError: If the specified LLM type is not registered
    """
    llm_type = llm_type.lower()
    if llm_type not in _llm_clients:
        raise ValueError(
            f"Unknown LLM type: {llm_type}. "
            f"Available types: {', '.join(_llm_clients.keys())}"
        )
    
    logger.info(f"Creating LLM client of type: {llm_type}")
    return _llm_clients[llm_type](**kwargs)


def get_available_llm_types() -> list[str]:
    """Get a list of available LLM client types.
    
    Returns:
        List of available LLM client type names
    """
    return list(_llm_clients.keys())


# Import all LLM client implementations to register them
# This ensures they're available through the factory
# The imports are at the bottom to avoid circular imports
from .claude_client import ClaudeClient  # noqa: F401
