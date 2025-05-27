"""Elasticsearch vector database implementation for the RAG system.

This module provides a vector database implementation using Elasticsearch
for efficient similarity search and retrieval in production environments.
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)


class ElasticsearchVectorDB:
    """Vector database implementation using Elasticsearch."""

    def __init__(
        self,
        index_name: str = "rag_documents",
        es_hosts: Optional[List[str]] = None,
        es_api_key: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        dimension: int = 384,  # Default dimension for all-MiniLM-L6-v2 model
        similarity_metric: str = "cosine"
    ):
        """Initialize the Elasticsearch vector database.

        Args:
            index_name: Name of the Elasticsearch index
            es_hosts: List of Elasticsearch hosts (e.g., ["http://localhost:9200"])
            es_api_key: Elasticsearch API key for authentication
            es_cloud_id: Elasticsearch Cloud ID for cloud deployments
            dimension: Dimension of the embedding vectors
            similarity_metric: Similarity metric to use ('cosine', 'dot_product', 'l2')
        """
        self.index_name = index_name
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        
        # Set up Elasticsearch client
        self.es_hosts = es_hosts or ["http://localhost:9200"]
        self.es_api_key = es_api_key or os.environ.get("ELASTICSEARCH_API_KEY")
        self.es_cloud_id = es_cloud_id or os.environ.get("ELASTICSEARCH_CLOUD_ID")
        
        self.es = self._create_client()
        
        # Create index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self._create_index()
    
    def _create_client(self) -> Elasticsearch:
        """Create an Elasticsearch client.
        
        Returns:
            Elasticsearch client
        """
        es_kwargs = {}
        
        if self.es_cloud_id:
            es_kwargs["cloud_id"] = self.es_cloud_id
            
        if self.es_api_key:
            es_kwargs["api_key"] = self.es_api_key
        else:
            es_kwargs["hosts"] = self.es_hosts
            
        try:
            es = Elasticsearch(**es_kwargs)
            logger.info(f"Connected to Elasticsearch: {es.info()}")
            return es
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise
    
    def _create_index(self):
        """Create the Elasticsearch index with vector search capabilities."""
        # Define the mapping for the index
        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.dimension,
                        "index": True,
                        "similarity": self.similarity_metric
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        }
        
        try:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index '{self.index_name}' with vector search capabilities")
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {str(e)}")
            raise
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """Add a document with embedding to the index.
        
        Args:
            document: Document dict containing 'embedding' and other metadata
            
        Returns:
            ID of the document in the index
        """
        if 'embedding' not in document:
            raise ValueError("Document must contain an 'embedding' field")
            
        # Prepare document for indexing
        doc = {
            "text": document.get("text", ""),
            "metadata": document.get("metadata", {}),
            "embedding": document["embedding"]
        }
        
        try:
            # Index the document
            response = self.es.index(index=self.index_name, document=doc)
            doc_id = response["_id"]
            
            logger.debug(f"Added document to Elasticsearch with ID {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to Elasticsearch: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents with embeddings to the index.
        
        Args:
            documents: List of document dicts, each containing 'embedding' and other metadata
            
        Returns:
            List of document IDs in the index
        """
        if not documents:
            return []
            
        # Prepare bulk indexing actions
        actions = []
        for doc in documents:
            if 'embedding' not in doc:
                logger.warning("Document missing 'embedding' field, skipping")
                continue
                
            action = {
                "_index": self.index_name,
                "_source": {
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc["embedding"]
                }
            }
            actions.append(action)
            
        if not actions:
            return []
            
        try:
            # Bulk index the documents
            results = helpers.bulk(self.es, actions)
            logger.info(f"Bulk indexed {results[0]} documents to Elasticsearch")
            
            # We don't have direct access to the IDs in bulk indexing
            # To get IDs, we would need to search for the documents
            return [f"bulk_indexed_{i}" for i in range(results[0])]
            
        except Exception as e:
            logger.error(f"Error bulk indexing documents to Elasticsearch: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_query: Optional Elasticsearch query to filter results
            
        Returns:
            List of document dicts with similarity scores
        """
        # Prepare the query
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": k * 2
            }
        }
        
        # Add filter if provided
        if filter_query:
            query = {
                "bool": {
                    "must": [query],
                    "filter": filter_query
                }
            }
        
        try:
            # Execute the search
            response = self.es.search(
                index=self.index_name,
                query=query,
                size=k
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc_id = hit["_id"]
                score = hit["_score"]
                
                # Prepare result
                result = {
                    "id": doc_id,
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "similarity": score
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {str(e)}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dict or None if not found
        """
        try:
            response = self.es.get(index=self.index_name, id=doc_id)
            doc = response["_source"]
            
            return {
                "id": doc_id,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "embedding": doc.get("embedding", [])
            }
            
        except Exception as e:
            logger.error(f"Error getting document from Elasticsearch: {str(e)}")
            return None
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index.
        
        Returns:
            Number of documents
        """
        try:
            response = self.es.count(index=self.index_name)
            return response["count"]
            
        except Exception as e:
            logger.error(f"Error getting document count from Elasticsearch: {str(e)}")
            return 0
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document from Elasticsearch: {str(e)}")
            return False
    
    def clear(self):
        """Clear the index by deleting and recreating it."""
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                
            self._create_index()
            logger.info(f"Cleared Elasticsearch index '{self.index_name}'")
            
        except Exception as e:
            logger.error(f"Error clearing Elasticsearch index: {str(e)}")
            raise


def build_es_vector_db_from_embedded_docs(
    input_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\embedded",
    es_hosts: Optional[List[str]] = None,
    es_api_key: Optional[str] = None,
    es_cloud_id: Optional[str] = None,
    index_name: str = "rag_documents",
    similarity_metric: str = "cosine"
) -> Dict[str, Any]:
    """Build an Elasticsearch vector database from embedded documents.
    
    Args:
        input_dir: Directory containing embedded documents
        es_hosts: List of Elasticsearch hosts
        es_api_key: Elasticsearch API key for authentication
        es_cloud_id: Elasticsearch Cloud ID for cloud deployments
        index_name: Name of the Elasticsearch index
        similarity_metric: Similarity metric to use
        
    Returns:
        Dict with statistics about the vector database
    """
    input_dir = Path(input_dir)
    
    # Get the dimension from the first document
    first_file = next(input_dir.glob("*.embedded*.json"), None)
    if not first_file:
        raise FileNotFoundError(f"No embedded documents found in {input_dir}")
        
    with open(first_file, 'r', encoding='utf-8') as f:
        first_doc = json.load(f)
        dimension = len(first_doc.get('embedding', []))
        
    # Initialize vector database
    vector_db = ElasticsearchVectorDB(
        index_name=index_name,
        es_hosts=es_hosts,
        es_api_key=es_api_key,
        es_cloud_id=es_cloud_id,
        dimension=dimension,
        similarity_metric=similarity_metric
    )
    
    # Clear the index to start fresh
    vector_db.clear()
    
    stats = {
        "total_documents": 0,
        "documents_by_topic": {}
    }
    
    # Process each embedded document
    for file_path in input_dir.glob("*.embedded*.json"):
        try:
            # Load the embedded document
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Get document topic
            topic = document.get("metadata", {}).get("topic", "unknown")
            
            # Add to vector database
            doc_id = vector_db.add_document(document)
            
            # Update stats
            stats["total_documents"] += 1
            if topic not in stats["documents_by_topic"]:
                stats["documents_by_topic"][topic] = 0
            stats["documents_by_topic"][topic] += 1
            
            logger.info(f"Added {file_path.name} to Elasticsearch with ID {doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return stats


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build vector database
    try:
        stats = build_es_vector_db_from_embedded_docs()
        
        logger.info(f"Elasticsearch vector database build complete. Added {stats['total_documents']} documents.")
        for topic, count in stats["documents_by_topic"].items():
            logger.info(f"Topic '{topic}': {count} documents")
    except Exception as e:
        logger.error(f"Error building Elasticsearch vector database: {str(e)}")
