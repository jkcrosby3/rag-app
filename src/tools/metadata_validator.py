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
from typing import List, Dict, Any, Optional
import re
from .metadata_validator import MetadataManager
from .security_config import SecurityConfig

logger = logging.getLogger(__name__)

class MetadataSchema:
    """Defines the metadata schema and validation rules."""
    REQUIRED_FIELDS = {
        "title": str,
        "source": str,
        "topic": str,
        "created_date": str,
        "modified_date": str,
        "version": str,
        "language": str,
        "file_size": int,
        "checksum": str,
        "encoding": str,
        "document_id": str,
        "classification": str
    }
    
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
    def validate(metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata against the schema, including security and classification rules.
        
        Args:
            metadata: Dictionary containing metadata to validate
            
        Returns:
            True if metadata is valid, False otherwise
        """
        errors = []
        
        # Validate required fields
        for field, expected_type in MetadataSchema.REQUIRED_FIELDS.items():
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
                continue
            
            if not isinstance(metadata[field], expected_type):
                errors.append(f"Field {field} must be of type {expected_type.__name__}")
        
        # Validate dates
        for field in ["created_date", "modified_date"]:
            if field in metadata:
                try:
                    datetime.strptime(metadata[field], MetadataSchema.DATE_FORMAT)
                except ValueError:
                    errors.append(f"Invalid date format for {field}. Expected: {MetadataSchema.DATE_FORMAT}")
        
        # Validate classification and access
        if "classification" in metadata:
            if not SecurityConfig.validate_classification(metadata["classification"]):
                errors.append(f"Invalid classification: {metadata['classification']}")
        
        if "access_groups" in metadata:
            if not SecurityConfig.validate_access_groups(metadata["access_groups"]):
                errors.append(f"Invalid access groups format")
        
        # Validate optional fields
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
        if not self.schema.validate(merged_metadata):
            raise ValueError("Invalid metadata")
            
        return merged_metadata
