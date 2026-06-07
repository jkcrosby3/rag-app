"""
Query validation and formatting configuration.
"""
from typing import Dict, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum

@dataclass
class QueryValidation:
    """Validation rules for queries."""
    min_length: int = 5
    max_length: int = 2000
    required_keywords: Optional[list] = None
    forbidden_keywords: Optional[list] = None
    
    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate a query against the rules.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": []
        }
        
        # Check length
        if len(query) < self.min_length:
            results["valid"] = False
            results["errors"].append(f"Query is too short (min {self.min_length} characters)")
        
        if len(query) > self.max_length:
            results["valid"] = False
            results["errors"].append(f"Query is too long (max {self.max_length} characters)")
        
        # Check required keywords
        if self.required_keywords:
            missing = [kw for kw in self.required_keywords if not re.search(rf'\b{kw}\b', query, re.IGNORECASE)]
            if missing:
                results["valid"] = False
                results["errors"].append(f"Missing required keywords: {', '.join(missing)}")
        
        # Check forbidden keywords
        if self.forbidden_keywords:
            found = [kw for kw in self.forbidden_keywords if re.search(rf'\b{kw}\b', query, re.IGNORECASE)]
            if found:
                results["valid"] = False
                results["errors"].append(f"Contains forbidden keywords: {', '.join(found)}")
        
        return results

@dataclass
class ResponseFormat:
    """Configuration for response formatting."""
    style: str = "markdown"
    max_length: int = 2000
    include_sources: bool = True
    include_confidence: bool = True
    
    def format_response(self, response: str, sources: list = None, confidence: float = None) -> str:
        """
        Format the response according to the configured style.
        
        Args:
            response: The raw response text
            sources: List of source documents
            confidence: Confidence score
            
        Returns:
            Formatted response string
        """
        formatted = response
        
        if self.include_sources and sources:
            formatted += "\n\nSources:\n"
            for i, source in enumerate(sources, 1):
                formatted += f"{i}. {source}\n"
        
        if self.include_confidence and confidence is not None:
            formatted += f"\nConfidence: {confidence:.2%}"
        
        return formatted

# Default configurations
QUERY_VALIDATION = QueryValidation(
    min_length=5,
    max_length=2000,
    required_keywords=[],
    forbidden_keywords=['delete', 'drop', 'truncate']
)

RESPONSE_FORMAT = ResponseFormat(
    style="markdown",
    max_length=2000,
    include_sources=True,
    include_confidence=True
)
