"""
Document Relationship Manager for the RAG system.

This module provides functionality to manage document relationships,
tags, and metadata without requiring document renaming.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class RelationshipManager:
    """Manages document relationships, tags, and metadata with versioning support."""
    
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        auto_save: bool = True,
        enable_versioning: bool = True
    ):
        """Initialize the relationship manager.
        
        Args:
            data_dir: Base directory for data storage
            auto_save: Whether to automatically save changes
        """
        try:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up relationship registry
            self.metadata_dir = self.data_dir / "metadata"
            self.metadata_dir.mkdir(exist_ok=True)
            
            # Set up versioning directory if enabled
            self.enable_versioning = enable_versioning
            if self.enable_versioning:
                self.versions_dir = self.metadata_dir / "versions"
                self.versions_dir.mkdir(exist_ok=True)
            
            self.relationship_path = self.metadata_dir / "document_relationships.json"
            self.tags_path = self.metadata_dir / "document_tags.json"
            
            self.relationships = self._load_relationships()
            self.tags = self._load_tags()
            self.auto_save = auto_save
            
            logger.info(f"Initialized RelationshipManager with {len(self.relationships.get('documents', {}))} documents")
            
        except Exception as e:
            logger.error(f"Error initializing RelationshipManager: {e}")
            raise
    
    def _load_relationships(self) -> Dict[str, Any]:
        """Load the relationship registry from disk.
        
        Returns:
            Relationship registry dictionary
        """
        if self.relationship_path.exists():
            try:
                with open(self.relationship_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading relationship registry: {e}")
                
        # Initialize empty registry
        return {
            "documents": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_tags(self) -> Dict[str, Any]:
        """Load the tags registry from disk.
        
        Returns:
            Tags registry dictionary
        """
        if self.tags_path.exists():
            try:
                with open(self.tags_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading tags registry: {e}")
                
        # Initialize empty registry
        return {
            "tags": {},
            "documents": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_relationships(self):
        """Save the relationship registry to disk."""
        timestamp = datetime.now()
        self.relationships["last_updated"] = timestamp.isoformat()
        
        try:
            # Save current version
            with open(self.relationship_path, "w", encoding="utf-8") as f:
                json.dump(self.relationships, f, indent=2, default=str)
            
            # Save versioned copy if enabled
            if self.enable_versioning:
                version_filename = f"relationships_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                version_path = self.versions_dir / version_filename
                with open(version_path, "w", encoding="utf-8") as f:
                    json.dump(self.relationships, f, indent=2, default=str)
                logger.debug(f"Saved relationship version: {version_filename}")
        except Exception as e:
            logger.error(f"Error saving relationship registry: {e}")
    
    def _save_tags(self):
        """Save the tags registry to disk."""
        timestamp = datetime.now()
        self.tags["last_updated"] = timestamp.isoformat()
        
        try:
            # Save current version
            with open(self.tags_path, "w", encoding="utf-8") as f:
                json.dump(self.tags, f, indent=2, default=str)
            
            # Save versioned copy if enabled
            if self.enable_versioning:
                version_filename = f"tags_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                version_path = self.versions_dir / version_filename
                with open(version_path, "w", encoding="utf-8") as f:
                    json.dump(self.tags, f, indent=2, default=str)
                logger.debug(f"Saved tags version: {version_filename}")
        except Exception as e:
            logger.error(f"Error saving tags registry: {e}")
    
    def save(self):
        """Save all registries to disk."""
        self._save_relationships()
        self._save_tags()
    
    def register_document(
        self,
        doc_id: str,
        original_name: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a document in the relationship registry.
        
        Args:
            doc_id: Document ID
            original_name: Original document name
            source_path: Path to document in source system
            metadata: Additional metadata
        """
        # Initialize document in relationship registry if not exists
        if doc_id not in self.relationships["documents"]:
            self.relationships["documents"][doc_id] = {
                "original_name": original_name,
                "source_path": source_path,
                "relationships": {},
                "metadata": metadata or {}
            }
        
        # Initialize document in tags registry if not exists
        if doc_id not in self.tags["documents"]:
            self.tags["documents"][doc_id] = {
                "tags": []
            }
        
        if self.auto_save:
            self.save()
    
    def add_relationship(
        self,
        source_doc_id: str,
        target_doc_id: str,
        relationship_type: str,
        bidirectional: bool = False
    ):
        """Add a relationship between documents.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            bidirectional: Whether to add the reverse relationship
        
        Returns:
            True if successful, False otherwise
        """
        # Check if documents exist
        if source_doc_id not in self.relationships["documents"]:
            logger.warning(f"Source document {source_doc_id} not found in registry")
            return False
            
        if target_doc_id not in self.relationships["documents"]:
            logger.warning(f"Target document {target_doc_id} not found in registry")
            return False
        
        # Initialize relationship type if not exists
        if relationship_type not in self.relationships["documents"][source_doc_id]["relationships"]:
            self.relationships["documents"][source_doc_id]["relationships"][relationship_type] = []
        
        # Add relationship if not exists
        if target_doc_id not in self.relationships["documents"][source_doc_id]["relationships"][relationship_type]:
            self.relationships["documents"][source_doc_id]["relationships"][relationship_type].append(target_doc_id)
        
        # Add bidirectional relationship if requested
        if bidirectional:
            if relationship_type not in self.relationships["documents"][target_doc_id]["relationships"]:
                self.relationships["documents"][target_doc_id]["relationships"][relationship_type] = []
            
            if source_doc_id not in self.relationships["documents"][target_doc_id]["relationships"][relationship_type]:
                self.relationships["documents"][target_doc_id]["relationships"][relationship_type].append(source_doc_id)
        
        if self.auto_save:
            self._save_relationships()
            
        return True
    
    def remove_relationship(
        self,
        source_doc_id: str,
        target_doc_id: str,
        relationship_type: str,
        bidirectional: bool = False
    ):
        """Remove a relationship between documents.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            bidirectional: Whether to remove the reverse relationship
        
        Returns:
            True if successful, False otherwise
        """
        # Check if documents exist
        if source_doc_id not in self.relationships["documents"]:
            logger.warning(f"Source document {source_doc_id} not found in registry")
            return False
            
        if target_doc_id not in self.relationships["documents"]:
            logger.warning(f"Target document {target_doc_id} not found in registry")
            return False
        
        # Check if relationship exists
        if (relationship_type not in self.relationships["documents"][source_doc_id]["relationships"] or
            target_doc_id not in self.relationships["documents"][source_doc_id]["relationships"][relationship_type]):
            logger.warning(f"Relationship {relationship_type} from {source_doc_id} to {target_doc_id} not found")
            return False
        
        # Remove relationship
        self.relationships["documents"][source_doc_id]["relationships"][relationship_type].remove(target_doc_id)
        
        # Remove bidirectional relationship if requested
        if bidirectional:
            if (relationship_type in self.relationships["documents"][target_doc_id]["relationships"] and
                source_doc_id in self.relationships["documents"][target_doc_id]["relationships"][relationship_type]):
                self.relationships["documents"][target_doc_id]["relationships"][relationship_type].remove(source_doc_id)
        
        if self.auto_save:
            self._save_relationships()
            
        return True
    
    def get_related_documents(
        self,
        doc_id: str,
        relationship_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get documents related to the specified document.
        
        Args:
            doc_id: Document ID
            relationship_type: Type of relationship to filter by
        
        Returns:
            Dictionary of relationship types and related document IDs
        """
        if doc_id not in self.relationships["documents"]:
            logger.warning(f"Document {doc_id} not found in registry")
            return {}
        
        relationships = self.relationships["documents"][doc_id]["relationships"]
        
        if relationship_type:
            return {relationship_type: relationships.get(relationship_type, [])}
        
        return relationships
    
    def add_tag(self, doc_id: str, tag: str, category: Optional[str] = None):
        """Add a tag to a document.
        
        Args:
            doc_id: Document ID
            tag: Tag to add
            category: Optional tag category
        
        Returns:
            True if successful, False otherwise
        """
        # Check if document exists
        if doc_id not in self.tags["documents"]:
            logger.warning(f"Document {doc_id} not found in tags registry")
            return False
        
        # Format tag with category if provided
        full_tag = f"{category}:{tag}" if category else tag
        
        # Add tag if not exists
        if full_tag not in self.tags["documents"][doc_id]["tags"]:
            self.tags["documents"][doc_id]["tags"].append(full_tag)
        
        # Register tag in global registry if not exists
        if full_tag not in self.tags["tags"]:
            self.tags["tags"][full_tag] = {
                "category": category,
                "value": tag,
                "document_count": 0
            }
        
        # Update document count
        self.tags["tags"][full_tag]["document_count"] += 1
        
        if self.auto_save:
            self._save_tags()
            
        return True
    
    def remove_tag(self, doc_id: str, tag: str, category: Optional[str] = None):
        """Remove a tag from a document.
        
        Args:
            doc_id: Document ID
            tag: Tag to remove
            category: Optional tag category
        
        Returns:
            True if successful, False otherwise
        """
        # Check if document exists
        if doc_id not in self.tags["documents"]:
            logger.warning(f"Document {doc_id} not found in tags registry")
            return False
        
        # Format tag with category if provided
        full_tag = f"{category}:{tag}" if category else tag
        
        # Check if tag exists
        if full_tag not in self.tags["documents"][doc_id]["tags"]:
            logger.warning(f"Tag {full_tag} not found for document {doc_id}")
            return False
        
        # Remove tag
        self.tags["documents"][doc_id]["tags"].remove(full_tag)
        
        # Update document count in global registry
        if full_tag in self.tags["tags"]:
            self.tags["tags"][full_tag]["document_count"] -= 1
            
            # Remove tag from global registry if no documents have it
            if self.tags["tags"][full_tag]["document_count"] <= 0:
                del self.tags["tags"][full_tag]
        
        if self.auto_save:
            self._save_tags()
            
        return True
    
    def get_document_tags(self, doc_id: str, category: Optional[str] = None) -> List[str]:
        """Get tags for a document.
        
        Args:
            doc_id: Document ID
            category: Optional category to filter by
        
        Returns:
            List of tags
        """
        if doc_id not in self.tags["documents"]:
            logger.warning(f"Document {doc_id} not found in tags registry")
            return []
        
        tags = self.tags["documents"][doc_id]["tags"]
        
        if category:
            return [tag.split(":", 1)[1] for tag in tags if tag.startswith(f"{category}:")]
        
        return tags
    
    def get_documents_by_tag(self, tag: str, category: Optional[str] = None) -> List[str]:
        """Get documents with a specific tag.
        
        Args:
            tag: Tag to search for
            category: Optional tag category
        
        Returns:
            List of document IDs
        """
        full_tag = f"{category}:{tag}" if category else tag
        
        return [
            doc_id for doc_id, doc in self.tags["documents"].items()
            if full_tag in doc["tags"]
        ]
    
    def get_all_tags(self, category: Optional[str] = None) -> List[str]:
        """Get all tags in the registry.
        
        Args:
            category: Optional category to filter by
        
        Returns:
            List of tags
        """
        if category:
            return [
                tag.split(":", 1)[1] for tag in self.tags["tags"]
                if tag.startswith(f"{category}:")
            ]
        
        return list(self.tags["tags"].keys())
    
    def get_tag_categories(self) -> Set[str]:
        """Get all tag categories in the registry.
        
        Returns:
            Set of categories
        """
        categories = set()
        
        for tag in self.tags["tags"]:
            if ":" in tag:
                category = tag.split(":", 1)[0]
                categories.add(category)
        
        return categories
        
    # Version management methods
    
    def get_available_versions(self) -> Dict[str, List[str]]:
        """Get available versions of relationship and tag registries.
        
        Returns:
            Dictionary with 'relationships' and 'tags' keys, each containing a list of version filenames
        """
        if not self.enable_versioning:
            logger.warning("Versioning is not enabled")
            return {"relationships": [], "tags": []}
        
        relationship_versions = [f.name for f in self.versions_dir.glob("relationships_*.json")]
        tag_versions = [f.name for f in self.versions_dir.glob("tags_*.json")]
        
        return {
            "relationships": sorted(relationship_versions, reverse=True),
            "tags": sorted(tag_versions, reverse=True)
        }
    
    def load_relationship_version(self, version_filename: str) -> bool:
        """Load a specific version of the relationship registry.
        
        Args:
            version_filename: Filename of the version to load
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_versioning:
            logger.warning("Versioning is not enabled")
            return False
        
        version_path = self.versions_dir / version_filename
        if not version_path.exists():
            logger.warning(f"Version {version_filename} not found")
            return False
        
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                self.relationships = json.load(f)
            
            if self.auto_save:
                self._save_relationships()
                
            logger.info(f"Loaded relationship version: {version_filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading relationship version: {e}")
            return False
    
    def load_tags_version(self, version_filename: str) -> bool:
        """Load a specific version of the tags registry.
        
        Args:
            version_filename: Filename of the version to load
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_versioning:
            logger.warning("Versioning is not enabled")
            return False
        
        version_path = self.versions_dir / version_filename
        if not version_path.exists():
            logger.warning(f"Version {version_filename} not found")
            return False
        
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                self.tags = json.load(f)
            
            if self.auto_save:
                self._save_tags()
                
            logger.info(f"Loaded tags version: {version_filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading tags version: {e}")
            return False
            
    def create_snapshot(self, snapshot_name: str) -> str:
        """Create a named snapshot of the current relationship and tag registries.
        
        Args:
            snapshot_name: Name for the snapshot
            
        Returns:
            Snapshot filename
        """
        if not self.enable_versioning:
            logger.warning("Versioning is not enabled")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshot_{timestamp}_{snapshot_name}.json"
        snapshot_path = self.versions_dir / snapshot_filename
        
        try:
            snapshot = {
                "name": snapshot_name,
                "timestamp": timestamp,
                "relationships": self.relationships,
                "tags": self.tags
            }
            
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
                
            logger.info(f"Created snapshot: {snapshot_filename}")
            return snapshot_filename
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return ""
    
    def load_snapshot(self, snapshot_filename: str) -> bool:
        """Load a named snapshot of relationship and tag registries.
        
        Args:
            snapshot_filename: Filename of the snapshot to load
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_versioning:
            logger.warning("Versioning is not enabled")
            return False
        
        snapshot_path = self.versions_dir / snapshot_filename
        if not snapshot_path.exists():
            logger.warning(f"Snapshot {snapshot_filename} not found")
            return False
        
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            
            self.relationships = snapshot["relationships"]
            self.tags = snapshot["tags"]
            
            if self.auto_save:
                self._save_relationships()
                self._save_tags()
                
            logger.info(f"Loaded snapshot: {snapshot_filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading snapshot: {e}")
            return False
    
    def update_document_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Update document metadata.
        
        Args:
            doc_id: Document ID
            metadata: Metadata to update
        
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.relationships["documents"]:
            logger.warning(f"Document {doc_id} not found in registry")
            return False
        
        # Update metadata
        self.relationships["documents"][doc_id]["metadata"].update(metadata)
        
        if self.auto_save:
            self._save_relationships()
            
        return True
    
    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document metadata
        """
        if doc_id not in self.relationships["documents"]:
            logger.warning(f"Document {doc_id} not found in registry")
            return {}
        
        return self.relationships["documents"][doc_id]["metadata"]
