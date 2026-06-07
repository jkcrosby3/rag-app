"""
Metadata validator for the RAG system.

This module provides validation for document metadata, ensuring it conforms to
industry standards and application requirements.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Mapping, Union
import re
from .security_config import SecurityConfig

logger = logging.getLogger(__name__)

class ClassificationHierarchy:
    """Manages classification hierarchy and validation."""
    
    # Base classification levels
    BASE_CLASSIFICATIONS = {
        "U": 0,  # Unclassified
        "C": 1,  # Confidential
        "S": 2,  # Secret
        "TS": 3  # Top Secret
    }
    
    # Additional classification components
    COMPONENTS = {
        "SI": 1,  # Special Intelligence
        "NOFORN": 2,  # No Foreign Dissemination
        "REL": 3  # Releasable To
    }

    @staticmethod
    def create_classification(
        base: str,
        components: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        originator: Optional[str] = None,
        declassification: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a classification dictionary.
        
        Args:
            base: Base classification level (U, C, S, TS)
            components: List of classification components
            timestamp: Classification timestamp
            originator: Originating agency
            declassification: Declassification date or instructions
            
        Returns:
            Classification dictionary
        """
        return {
            "classification": base,
            "components": components or [],
            "timestamp": timestamp,
            "originator": originator,
            "declassification": declassification
        }

    @staticmethod
    def parse_classification(classification: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse a classification string or dict into its components.
        
        Args:
            classification: Classification string (e.g., "TS//SI//NOFORN") or dict with classification info
            
        Returns:
            Dict with classification components
        """
        if isinstance(classification, dict):
            # If already a dict, validate it
            if not classification.get("classification"):
                raise ValueError("Classification dictionary must have 'classification' field")
            if not isinstance(classification.get("components", []), list):
                raise ValueError("Classification components must be a list")
            return classification
            
        # If string, parse it
        classification = str(classification).strip('()[]').upper()
        parts = classification.split('//')
        main_class = parts[0]
        components = parts[1:] if len(parts) > 1 else []
        
        # Validate main classification
        if main_class not in ClassificationHierarchy.BASE_CLASSIFICATIONS:
            raise ValueError(f"Invalid classification: {main_class}")
            
        return ClassificationHierarchy.create_classification(
            base=main_class,
            components=components
        )
    
    @staticmethod
    def get_classification_level(classification: Dict[str, Any]) -> int:
        """Get the numeric level of a classification.
        
        Args:
            classification: Classification dictionary
            
        Returns:
            Numeric level (0=U, 1=C, 2=S, 3=TS)
        """
        base_level = classification.get('classification', 'U')
        if base_level not in ClassificationHierarchy.BASE_CLASSIFICATIONS:
            return 0  # Default to Unclassified if invalid level
        return ClassificationHierarchy.BASE_CLASSIFICATIONS[base_level]
    
    @staticmethod
    def is_valid_classification(classification: Dict[str, Any]) -> bool:
        """Check if a classification dictionary is valid."""
        if not isinstance(classification, dict):
            raise TypeError("Classification must be a dictionary")
            
        return (
            classification['classification'] in ClassificationHierarchy.BASE_CLASSIFICATIONS and
            all(comp in ClassificationHierarchy.COMPONENTS for comp in classification['components'])
        )
    
    @staticmethod
    def is_accessible_by(
        user_clearance: Dict[str, Any],
        document_classification: Dict[str, Any]
    ) -> bool:
        """Check if a user's clearance can access a document's classification.
        
        Args:
            user_clearance: User's clearance dictionary
            document_classification: Document's classification dictionary
            
        Returns:
            True if user can access the document, False otherwise
        """
        if not isinstance(user_clearance, dict):
            raise TypeError("User clearance must be a dictionary")
        if not isinstance(document_classification, dict):
            raise TypeError("Document classification must be a dictionary")
            
        # Get base classification levels
        user_base = user_clearance.get('classification', 'U')
        doc_base = document_classification.get('classification', 'U')
        
        # Convert to numeric levels
        user_level = ClassificationHierarchy.BASE_CLASSIFICATIONS.get(user_base, 0)
        doc_level = ClassificationHierarchy.BASE_CLASSIFICATIONS.get(doc_base, 0)
        
        # Check if user has need-to-know access
        if document_classification.get("components"):
            for component in document_classification["components"]:
                if component not in user_clearance.get("components", []):
                    return False
        
        return user_level >= doc_level
    
    @staticmethod
    def filter_content_by_clearance(
        content: List[Dict[str, Any]],
        user_clearance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter content based on user's clearance level."""
        user_level = ClassificationHierarchy.get_classification_level(user_clearance)
        
        return [
            item for item in content
            if ClassificationHierarchy.get_classification_level(
                item.get('classification', {
                    "classification": "U",
                    "components": []
                })
            ) <= user_level
        ]
    
    @staticmethod
    def redact_content(
        content: List[Dict[str, Any]],
        user_clearance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Redact content that's above user's clearance level."""
        user_level = ClassificationHierarchy.get_classification_level(user_clearance)
        
        redacted_content = []
        for item in content:
            item_level = ClassificationHierarchy.get_classification_level(
                item.get('classification', {
                    "classification": "U",
                    "components": []
                })
            )
            
            if item_level > user_level:
                # Redact content above user's clearance
                redacted_content.append({
                    'text': '[REDACTED - ACCESS DENIED]',
                    'metadata': {
                        'classification': item['metadata'].get('classification', 'U'),
                        'redacted': True
                    }
                })
            else:
                redacted_content.append(item)
        
        return redacted_content
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from .security_config import SecurityConfig

logger = logging.getLogger(__name__)

class MetadataSchema:
    """Defines the metadata schema and validation rules."""
    REQUIRED_FIELDS = {
        "title": str,
        "source": str,
        "topic": str,
        "created_at": str,
        "modified_at": str,
        "version": str,
        "language": str,
        "file_size": int,
        "checksum": str,
        "encoding": str,
        "document_id": str,
        "classification": str
    }
    
    # Valid classification formats
    VALID_CLASSIFICATIONS = [
        "U",  # Unclassified
        "C",  # Confidential
        "S",  # Secret
        "TS"  # Top Secret
    ]
    
    OPTIONAL_FIELDS = {
        "author": str,
        "keywords": list,
        "summary": str,
        "content_type": str,
        "access_level": str,
        "license": str,
        "parent_document": str,
        "related_documents": list,
        "version_history": list,
        "access_groups": list
    }
    
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate metadata against the schema.
        
        Args:
            metadata: Dictionary containing metadata to validate
            
        Returns:
            True if metadata is valid, False otherwise
        """
        errors = []
        
        # Check required fields
        for field, expected_type in MetadataSchema.REQUIRED_FIELDS.items():
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
                continue
                
            value = metadata[field]
            if not isinstance(value, expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Check classification format
        classification = metadata.get('classification', 'U')
        if classification not in MetadataSchema.VALID_CLASSIFICATIONS:
            errors.append(f"Invalid classification: {classification}")
        
        # Check optional fields
        for field, expected_type in MetadataSchema.OPTIONAL_FIELDS.items():
            if field in metadata and not isinstance(metadata[field], expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(metadata[field]).__name__}")
        for field, expected_type in MetadataSchema.OPTIONAL_FIELDS.items():
            if field in metadata and not isinstance(metadata[field], expected_type):
                errors.append(f"Field {field} must be of type {expected_type.__name__}")
        
        if errors:
            logger.error("Metadata validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
            
        return True
    
    @staticmethod
    def get_classification_hierarchy() -> Dict[str, int]:
        """Get classification hierarchy."""
        return SecurityConfig.get_classification_hierarchy()
    
    @staticmethod
    def get_security_hierarchy() -> Dict[str, int]:
        """Get security level hierarchy."""
        return SecurityConfig.get_security_hierarchy()
    
    @staticmethod
    def get_sensitivity_hierarchy() -> Dict[str, int]:
        """Get sensitivity level hierarchy."""
        return SecurityConfig.get_sensitivity_hierarchy()
    
    @staticmethod
    def get_default_access_groups() -> List[str]:
        """Get default access groups."""
        return SecurityConfig.DEFAULT_ACCESS_GROUPS

    @staticmethod
    def generate_default_metadata(file_path: Path) -> Dict[str, Any]:
        """
        Generate default metadata for a document with security settings.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing default metadata with security settings
        """
        now = datetime.now().strftime(MetadataSchema.DATE_FORMAT)
        
        # Generate checksum
        checksum = MetadataSchema._generate_checksum(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Generate unique document ID
        doc_id = MetadataSchema._generate_document_id(file_path)
        
        return {
            "title": file_path.stem,
            "source": str(file_path),
            "topic": "",  # To be set by caller
            "created_date": now,
            "modified_date": now,
            "version": "1.0",
            "language": "en",
            "file_size": file_size,
            "checksum": checksum,
            "encoding": "utf-8",
            "document_id": doc_id,
            "classification": "public",
            "access_groups": SecurityConfig.DEFAULT_ACCESS_GROUPS.copy()
        }
    
    @staticmethod
    def _generate_document_id(file_path: Path) -> str:
        """Generate a unique document ID based on file path and checksum."""
        # Use file path and checksum to create a consistent but unique ID
        doc_id = f"doc_{file_path.stem}_{MetadataSchema._generate_checksum(file_path)[:8]}"
        return doc_id
    
    @staticmethod
    def _generate_checksum(file_path: Path) -> str:
        """Generate SHA-256 checksum for a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error generating checksum: {str(e)}")
            return ""

class MetadataManager:
    """Manages metadata operations including validation and merging."""
    
    def __init__(self):
        self.schema = MetadataSchema()
    
    def validate_and_merge_metadata(
        self, 
        base_metadata: Dict[str, Any],
        external_metadata: Optional[Dict[str, Any]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate and merge metadata from different sources.
        
        Args:
            base_metadata: Base metadata dictionary
            external_metadata: External metadata from JSON file
            custom_metadata: Custom metadata provided by user
            
        Returns:
            Validated and merged metadata dictionary
        """
        # Start with base metadata
        merged_metadata = base_metadata.copy()
        
        # Add external metadata
        if external_metadata:
            merged_metadata.update(external_metadata)
        
        # Add custom metadata
        if custom_metadata:
            merged_metadata.update(custom_metadata)
        
        # Validate the result
        if not self.schema.validate_metadata(merged_metadata):
            raise ValueError("Invalid metadata")
            
        return merged_metadata
