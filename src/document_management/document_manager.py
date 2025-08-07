"""
Document Manager for the RAG system.

This module provides functionality to manage document ingestion,
listing, and processing from various sources.
"""
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Set, Any, Tuple
from datetime import datetime
import threading
import time

# Import internal modules
from src.document_processing.batch_processor import BatchProcessor
from src.document_processing.chunker import Chunker
from src.document_management.relationship_manager import RelationshipManager

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document ingestion, tracking, and processing."""
    
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the document manager.
        
        Args:
            data_dir: Base directory for data storage
            config_path: Path to configuration file
        """
        try:
            # Set up data directory
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using data directory: {self.data_dir}")
            
            # Set up document directories
            self.documents_dir = self.data_dir / "documents"
            self.documents_dir.mkdir(exist_ok=True)
            
            self.staging_dir = self.data_dir / "staging"
            self.staging_dir.mkdir(exist_ok=True)
            
            self.processed_dir = self.data_dir / "processed"
            self.processed_dir.mkdir(exist_ok=True)
            
            self.chunks_dir = self.data_dir / "chunks"
            self.chunks_dir.mkdir(exist_ok=True)
            
            self.embedded_dir = self.data_dir / "embedded"
            self.embedded_dir.mkdir(exist_ok=True)
            
            # Load configuration
            self.config = {}
            if config_path:
                config_path = Path(config_path)
                if config_path.exists():
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            self.config = json.load(f)
                        logger.info(f"Loaded configuration from {config_path}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing configuration file {config_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading configuration file {config_path}: {e}")
                else:
                    logger.warning(f"Configuration file not found: {config_path}")
            
            # Initialize batch processor, chunker, and relationship manager
            try:
                self.batch_processor = BatchProcessor()
                self.chunker = Chunker()
                self.relationship_manager = RelationshipManager(data_dir=self.data_dir)
                
                logger.info("Successfully initialized batch processor, chunker, and relationship manager")
                
                # Initialize document registry
                self.registry_path = self.data_dir / "document_registry.json"
                self.registry = self._load_registry()
                logger.info(f"Loaded document registry with {len(self.registry.get('documents', {}))} documents")
                
                # Log about existing documents
                if self.registry_path.exists():
                    existing_docs = self.registry.get('documents', {})
                    logger.info(f"Found {len(existing_docs)} existing documents in registry")
                else:
                    logger.info("No document registry found")
                
            except ImportError as e:
                logger.error(f"Failed to initialize document processing components: {e}")
                raise ImportError(f"Required document processing components not available: {e}") from e
            
            except Exception as e:
                logger.error(f"Error initializing document manager: {e}")
                raise
        except Exception as e:
            logger.error(f"Fatal error in document manager initialization: {e}")
            raise
        
    def search_documents(self, query: str) -> str:
        """Search through documents using the vector database.
        
        Args:
            query: Search query
            
        Returns:
            str: Search results
        """
        try:
            # Import and initialize retriever
            from src.retrieval.retriever import DocumentRetriever
            retriever = DocumentRetriever()
            
            # Perform search
            results = retriever.search(query)
            
            # Format results
            formatted_results = "\n".join([
                f"{doc['score']:.2f}: {doc['text']}" 
                for doc in results
            ])
            
            return formatted_results
        except ImportError as e:
            logger.error(f"Failed to import DocumentRetriever: {e}")
            return f"Error: Failed to import DocumentRetriever"
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {str(e)}"

    def _load_registry(self) -> Dict[str, Any]:
        """Load the document registry from disk.
        
        Returns:
            Document registry dictionary
        """
        try:
            registry = {
                "documents": {},
                "last_updated": datetime.now().isoformat()
            }
            
            if self.registry_path.exists():
                try:
                    with open(self.registry_path, "r", encoding="utf-8") as f:
                        registry = json.load(f)
                    logger.info(f"Loaded document registry from {self.registry_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error loading document registry: {e}")
                    logger.warning("Creating new registry due to JSON error")
                    # Remove the corrupted file
                    os.remove(self.registry_path)
                    logger.info(f"Removed corrupted registry file: {self.registry_path}")
        except Exception as e:
            logger.error(f"Unexpected error loading registry: {e}")
            registry = {
                "documents": {},
                "last_updated": datetime.now().isoformat()
            }
            logger.warning("Creating new registry due to unexpected error")
        
        return registry
    
    def _save_registry(self) -> None:
        """Save the document registry to disk with proper error handling."""
        try:
            # Ensure data directory exists and is writable
            if not os.access(self.data_dir, os.W_OK):
                raise PermissionError(f"Write access denied to data directory: {self.data_dir}")
            
            # Create a temporary file in the same directory
            temp_path = self.registry_path.with_suffix('.tmp')
            
            # Use atomic file operations with retry logic
            max_retries = 5
            retry_delay = 0.5  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Write to temporary file
                    with open(temp_path, "w", encoding="utf-8", errors='replace') as f:
                        json.dump(self.registry, f, indent=2, default=str)
                    
                    # Try to replace the file
                    if self.registry_path.exists():
                        try:
                            # Try to remove the old file with retry
                            for _ in range(3):
                                try:
                                    os.remove(self.registry_path)
                                    break
                                except PermissionError:
                                    # Wait a bit and try again
                                    time.sleep(retry_delay)
                            else:
                                raise PermissionError(f"Could not remove old registry file after multiple attempts")
                        except PermissionError:
                            # If we can't remove, try to rename
                            try:
                                os.rename(temp_path, self.registry_path)
                                break
                            except PermissionError:
                                # Wait and try again
                                time.sleep(retry_delay)
                                continue
                    else:
                        # If no old file exists, just rename
                        os.rename(temp_path, self.registry_path)
                        break
                    
                    logger.info(f"Successfully saved document registry to {self.registry_path}")
                    break
                
                except PermissionError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Permission error saving document registry after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Permission error, retrying ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                
                except Exception as e:
                    logger.error(f"Error saving document registry: {e}")
                    if temp_path.exists():
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    raise
        
        except Exception as e:
            logger.error(f"Final error saving document registry: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise
        
    def import_document(
        self,
        file_path: Union[str, Path],
        source: str = "upload"
    ) -> str:
        """Import a single document with retry logic.
        
        Args:
            file_path: Path to the document file
            source: Source of the document (default: 'upload')
            
        Returns:
            Document ID
        
        Raises:
            ValueError: If the file is not a supported document type
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
                
            # Check if it's a supported document type
            supported_extensions = {".txt", ".pdf", ".docx", ".md"}
            if file_path.suffix.lower() not in supported_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
            # Copy file to documents directory with retry logic
            target_path = self.documents_dir / file_path.name
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    # Ensure target directory exists and is writable
                    if not os.access(self.documents_dir, os.W_OK):
                        raise PermissionError(f"Write access denied to documents directory: {self.documents_dir}")
                    
                    # Copy file with proper error handling
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Successfully copied document to: {target_path}")
                    break
                
                except PermissionError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Permission error after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Permission error, retrying ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    continue
                
                except Exception as e:
                    logger.error(f"Error copying document: {e}")
                    raise
            
            # Register the document
            return self.register_document(
                file_path=target_path,
                source=source,
                metadata={"original_path": str(file_path)}
            )
            
        except Exception as e:
            logger.error(f"Error importing document {file_path}: {e}")
            raise

    def register_document(
        self,
        file_path: Union[str, Path],
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        relationships: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """Register a document in the registry.
        
        Args:
            file_path: Path to the document
            source: Source of the document (e.g., 'sharepoint', 'upload')
            metadata: Additional metadata
            tags: List of tags to add to the document, each with optional category
                  Format: [{'tag': 'tag1', 'category': 'category1'}, {'tag': 'tag2'}]
            relationships: Dictionary of relationship types and target document IDs
                  Format: {'parent': ['doc_id1'], 'related': ['doc_id2', 'doc_id3']}
            
        Returns:
            Document ID
        """
        file_path = Path(file_path)
        doc_id = f"{source}_{file_path.name}_{int(datetime.now().timestamp())}"
        
        # Create document entry
        doc_entry = {
            "id": doc_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "source": source,
            "size_bytes": file_path.stat().st_size,
            "imported_at": datetime.now().isoformat(),
            "processed": False,
            "chunked": False,
            "embedded": False,
            "metadata": metadata or {}
        }
        
        # Add to registry
        self.registry["documents"][doc_id] = doc_entry
        self._save_registry()
        
        # Register document in relationship manager
        try:
            self.relationship_manager.register_document(
                doc_id=doc_id,
                original_name=file_path.name,
                source_path=str(file_path),
                metadata=metadata
            )
            
            # Add tags if provided
            if tags:
                for tag_info in tags:
                    self.relationship_manager.add_tag(
                        doc_id=doc_id,
                        tag=tag_info['tag'],
                        category=tag_info.get('category')
                    )
            
            # Add relationships if provided
            if relationships:
                for rel_type, target_ids in relationships.items():
                    for target_id in target_ids:
                        self.relationship_manager.add_relationship(
                            source_doc_id=doc_id,
                            target_doc_id=target_id,
                            relationship_type=rel_type
                        )
                        
            logger.info(f"Document {doc_id} registered with relationship manager")
        except Exception as e:
            logger.warning(f"Failed to register document with relationship manager: {e}")
        
        return doc_id
    
    def import_from_local(
        self,
        source_path: Union[str, Path],
        copy_to_documents: bool = True
    ) -> List[str]:
        """Import documents from a local directory.
        
        Args:
            source_path: Path to source directory or file
            copy_to_documents: Whether to copy files to documents directory
            
        Returns:
            List of document IDs
        """
        source_path = Path(source_path)
        doc_ids = []
        
        if source_path.is_file():
            files = [source_path]
        else:
            files = list(source_path.glob("**/*.*"))
            
        for file_path in files:
            # Skip non-document files
            if file_path.suffix.lower() not in [".txt", ".pdf", ".docx", ".md"]:
                continue
                
            # Copy file to documents directory if requested
            if copy_to_documents:
                target_path = self.documents_dir / file_path.name
                shutil.copy2(file_path, target_path)
                file_path = target_path
                
            # Register document
            doc_id = self.register_document(
                file_path=file_path,
                source="local",
                metadata={"original_path": str(source_path)}
            )
            
            doc_ids.append(doc_id)
            
        return doc_ids
    
    def import_from_sharepoint(
        self,
        library_name: str,
        folder_path: str = "",
        file_extensions: Optional[List[str]] = None
    ) -> List[str]:
        """Import documents from SharePoint.
        
        Args:
            library_name: Name of SharePoint document library
            folder_path: Path to folder within library
            file_extensions: List of file extensions to filter by
            
        Returns:
            List of document IDs
        """
        try:
            # Import SharePoint connector
            from src.connectors.sharepoint_connector import get_sharepoint_connector
            
            # Get SharePoint connector
            connector = get_sharepoint_connector()
            
            if not connector:
                logger.error("Failed to initialize SharePoint connector")
                return []
                
            # Connect to SharePoint
            if not connector.connect():
                logger.error("Failed to connect to SharePoint")
                return []
                
            # List documents
            documents = connector.list_documents(
                library_name=library_name,
                folder_path=folder_path,
                file_extensions=file_extensions or [".txt", ".pdf", ".docx", ".md"]
            )
            
            if not documents:
                logger.info("No documents found in SharePoint")
                return []
                
            # Download documents
            downloaded = connector.download_documents(
                documents=documents,
                target_dir=self.staging_dir,
                preserve_path=True
            )
            
            # Register downloaded documents
            doc_ids = []
            for file_path in downloaded:
                # Copy to documents directory
                target_path = self.documents_dir / file_path.name
                shutil.copy2(file_path, target_path)
                
                # Register document
                doc_id = self.register_document(
                    file_path=target_path,
                    source="sharepoint",
                    metadata={
                        "library": library_name,
                        "folder_path": folder_path,
                        "sharepoint_url": next(
                            (doc["url"] for doc in documents if doc["name"] == file_path.name),
                            None
                        )
                    }
                )
                
                doc_ids.append(doc_id)
                
            return doc_ids
            
        except ImportError:
            logger.error("SharePoint connector not available")
            return []
        except Exception as e:
            logger.error(f"Error importing from SharePoint: {e}")
            return []
    
    def process_documents(self, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process documents through the RAG pipeline.
        
        Args:
            doc_ids: List of document IDs to process, or None for all unprocessed
            
        Returns:
            Processing statistics
        """
        # Get documents to process
        if doc_ids is None:
            doc_ids = [
                doc_id for doc_id, doc in self.registry["documents"].items()
                if not doc["processed"]
            ]
            
        if not doc_ids:
            logger.info("No documents to process")
            return {"processed": 0}
            
        # Process documents
        processed_count = 0
        for doc_id in doc_ids:
            if doc_id not in self.registry["documents"]:
                logger.warning(f"Document {doc_id} not found in registry")
                continue
                
            doc = self.registry["documents"][doc_id]
            file_path = Path(doc["file_path"])
            
            if not file_path.exists():
                logger.warning(f"Document file {file_path} not found")
                continue
                
            try:
                # Process document
                result = self.batch_processor.process_document(
                    file_path=file_path,
                    output_dir=self.processed_dir
                )
                
                # Update registry
                doc["processed"] = True
                doc["processed_path"] = str(result["output_path"])
                doc["processed_at"] = datetime.now().isoformat()
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                
        # Save registry
        self._save_registry()
        
        return {"processed": processed_count}
    
    def chunk_documents(self, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Chunk processed documents.
        
        Args:
            doc_ids: List of document IDs to chunk, or None for all processed but not chunked
            
        Returns:
            Chunking statistics
        """
        # Get documents to chunk
        if doc_ids is None:
            doc_ids = [
                doc_id for doc_id, doc in self.registry["documents"].items()
                if doc["processed"] and not doc["chunked"]
            ]
            
        if not doc_ids:
            logger.info("No documents to chunk")
            return {"chunked": 0}
            
        # Chunk documents
        chunked_count = 0
        for doc_id in doc_ids:
            if doc_id not in self.registry["documents"]:
                logger.warning(f"Document {doc_id} not found in registry")
                continue
                
            doc = self.registry["documents"][doc_id]
            
            if not doc["processed"]:
                logger.warning(f"Document {doc_id} not processed yet")
                continue
                
            processed_path = Path(doc["processed_path"])
            
            if not processed_path.exists():
                logger.warning(f"Processed document {processed_path} not found")
                continue
                
            try:
                # Load processed document
                with open(processed_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # Chunk document using the correct method
                chunk_list = self.chunker.chunk_document(document)
                
                # Save chunks
                chunks_path = self.chunks_dir / f"{processed_path.stem}.chunks.json"
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_list, f, ensure_ascii=False, indent=2)
                
                # Create result dictionary similar to what we expect
                chunks = {
                    "output_path": chunks_path,
                    "chunks_count": len(chunk_list)
                }
                
                # Update registry
                doc["chunked"] = True
                doc["chunks_path"] = str(chunks["output_path"])
                doc["chunks_count"] = chunks["chunks_count"]
                doc["chunked_at"] = datetime.now().isoformat()
                
                chunked_count += 1
                
            except Exception as e:
                logger.error(f"Error chunking document {doc_id}: {e}")
                
        
        Returns:
            List of document metadata dictionaries
        """
        documents = []
        for doc_id, doc in self.registry.get('documents', {}).items():
            # Skip documents that have a parent relationship (chunks)
            relationships = self.relationship_manager.get_document_relationships(doc_id)
            if 'parent' in relationships and relationships['parent']:
                continue
            
            documents.append({
                'id': doc_id,
                'file_name': Path(doc['file_path']).name,
                'source': doc['source'],
                'size_bytes': Path(doc['file_path']).stat().st_size if Path(doc['file_path']).exists() else 0,
                'imported_at': doc['imported_at'],
                'processed': doc.get('processed', False),
                'chunked': doc.get('chunked', False),
                'embedded': doc.get('embedded', False)
            })
        return documents

    # Relationship management methods
    
    def add_document_relationship(self, source_doc_id: str, target_doc_id: str, relationship_type: str, bidirectional: bool = False) -> bool:
        """Add a relationship between documents.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship (e.g., 'parent', 'related', 'version_of')
            bidirectional: Whether to add the reverse relationship
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.add_relationship(
            source_doc_id=source_doc_id,
            target_doc_id=target_doc_id,
            relationship_type=relationship_type,
            bidirectional=bidirectional
        )
    
    def remove_document_relationship(self, source_doc_id: str, target_doc_id: str, relationship_type: str, bidirectional: bool = False) -> bool:
        """Remove a relationship between documents.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            bidirectional: Whether to remove the reverse relationship
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.remove_relationship(
            source_doc_id=source_doc_id,
            target_doc_id=target_doc_id,
            relationship_type=relationship_type,
            bidirectional=bidirectional
        )
    
    def get_related_documents(self, doc_id: str, relationship_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get documents related to the specified document.
        
        Args:
            doc_id: Document ID
            relationship_type: Type of relationship to filter by
            
        Returns:
            Dictionary of relationship types and related document IDs
        """
        return self.relationship_manager.get_related_documents(doc_id, relationship_type)
    
    def get_document_with_relationships(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata with relationships.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata with relationships
        """
        doc = self.get_document(doc_id)
        if not doc:
            return {}
            
        # Get relationships
        relationships = self.relationship_manager.get_related_documents(doc_id)
        
        # Get tags
        tags = self.relationship_manager.get_document_tags(doc_id)
        
        # Combine data
        result = doc.copy()
        result["relationships"] = relationships
        result["tags"] = tags
        
        return result
    
    # Tag management methods
    
    def add_document_tag(self, doc_id: str, tag: str, category: Optional[str] = None) -> bool:
        """Add a tag to a document.
        
        Args:
            doc_id: Document ID
            tag: Tag to add
            category: Optional tag category
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.add_tag(doc_id, tag, category)
    
    def remove_document_tag(self, doc_id: str, tag: str, category: Optional[str] = None) -> bool:
        """Remove a tag from a document.
        
        Args:
            doc_id: Document ID
            tag: Tag to remove
            category: Optional tag category
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.remove_tag(doc_id, tag, category)
    
    def get_document_tags(self, doc_id: str, category: Optional[str] = None) -> List[str]:
        """Get tags for a document.
        
        Args:
            doc_id: Document ID
            category: Optional category to filter by
            
        Returns:
            List of tags
        """
        return self.relationship_manager.get_document_tags(doc_id, category)
    
    def get_documents_by_tag(self, tag: str, category: Optional[str] = None) -> List[str]:
        """Get documents with a specific tag.
        
        Args:
            tag: Tag to search for
            category: Optional tag category
            
        Returns:
            List of document IDs
        """
        return self.relationship_manager.get_documents_by_tag(tag, category)
    
    def get_all_tags(self, category: Optional[str] = None) -> List[str]:
        """Get all tags in the registry.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of tags
        """
        return self.relationship_manager.get_all_tags(category)
    
    def get_tag_categories(self) -> Set[str]:
        """Get all tag categories in the registry.
        
        Returns:
            Set of categories
        """
        return self.relationship_manager.get_tag_categories()
    
    # Relationship versioning methods
    
    def get_relationship_versions(self) -> Dict[str, List[str]]:
        """Get available versions of relationship and tag registries.
        
        Returns:
            Dictionary with 'relationships' and 'tags' keys, each containing a list of version filenames
        """
        return self.relationship_manager.get_available_versions()
    
    def load_relationship_version(self, version_filename: str) -> bool:
        """Load a specific version of the relationship registry.
        
        Args:
            version_filename: Filename of the version to load
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.load_relationship_version(version_filename)
    
    def load_tags_version(self, version_filename: str) -> bool:
        """Load a specific version of the tags registry.
        
        Args:
            version_filename: Filename of the version to load
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.load_tags_version(version_filename)
    
    def create_relationship_snapshot(self, snapshot_name: str) -> str:
        """Create a named snapshot of the current relationship and tag registries.
        
        Args:
            snapshot_name: Name for the snapshot (e.g., 'before_reorganization')
            
        Returns:
            Snapshot filename
        """
        return self.relationship_manager.create_snapshot(snapshot_name)
    
    def load_relationship_snapshot(self, snapshot_filename: str) -> bool:
        """Load a named snapshot of relationship and tag registries.
        
        Args:
            snapshot_filename: Filename of the snapshot to load
            
        Returns:
            True if successful, False otherwise
        """
        return self.relationship_manager.load_snapshot(snapshot_filename)
    
    def run_full_pipeline(
        self,
        doc_ids: Optional[List[str]] = None,
        process: bool = True,
        chunk: bool = True,
        embed: bool = False,
        index: bool = False
    ) -> Dict[str, Any]:
        """Run the full document processing pipeline.
        
        Args:
            doc_ids: List of document IDs to process, or None for all eligible
            process: Whether to process documents
            chunk: Whether to chunk documents
            embed: Whether to generate embeddings
            index: Whether to index documents
            
        Returns:
            Processing statistics
        """
        stats = {}
        
        # Process documents
        if process:
            process_stats = self.process_documents(doc_ids)
            stats.update(process_stats)
            
        # Chunk documents
        if chunk:
            chunk_stats = self.chunk_documents(doc_ids)
            stats.update(chunk_stats)
            
        # Generate embeddings and index
        if embed or index:
            # These would be implemented in a similar way to process and chunk
            # but are omitted for brevity
            pass
            
        return stats
