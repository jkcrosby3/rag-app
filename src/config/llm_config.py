"""
Configuration for available LLM models.
"""
from typing import Dict, Any

# Define available LLM models
LLM_MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Fast and cost-effective model from OpenAI",
        "provider": "openai",
        "default": True,
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2000,
            "model": "gpt-3.5-turbo"
        }
    },
    # Add more models as needed
    # "gpt-4": {
    #     "name": "GPT-4",
    #     "description": "Latest model from OpenAI",
    #     "provider": "openai",
    #     "parameters": {
    #         "temperature": 0.7,
    #         "max_tokens": 4000,
    #         "model": "gpt-4"
    #     }
    # }
}

def get_default_llm() -> str:
    """Get the default LLM model."""
    for model_id, config in LLM_MODELS.items():
        if config.get("default", False):
            return model_id
    return next(iter(LLM_MODELS.keys()))

def get_llm_config(model_id: str) -> Dict[str, Any]:
    """Get configuration for a specific LLM model."""
    return LLM_MODELS.get(model_id, LLM_MODELS[get_default_llm()])

def get_all_llms() -> Dict[str, Dict[str, Any]]:
    """Get all available LLM models."""
    return LLM_MODELS
