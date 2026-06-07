"""Semantic cache for LLM responses in the RAG system.

This module provides a semantic caching mechanism that can match similar queries
even when they're not identical, reducing the need for repeated API calls.
"""
import logging
import time
import json
import hashlib
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from threading import Lock
import pickle
from datetime import datetime, timedelta

# Import the quantized embedding generator for faster semantic similarity
from src.embeddings.quantized_generator import QuantizedEmbeddingGenerator

logger = logging.getLogger(__name__)

class SemanticCache:
    """Semantic cache for LLM responses.
    
    This cache stores LLM responses along with query embeddings to enable
    semantic matching of similar queries, even when they're not identical.
    It uses cosine similarity to find similar queries in the cache.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        similarity_threshold: float = 0.75,
        ttl_hours: int = 24,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the semantic cache.
        
        Args:
            max_size: Maximum number of responses to cache
            similarity_threshold: Threshold for considering queries similar (0.0 to 1.0)
            ttl_hours: Time-to-live for cache entries in hours
            cache_dir: Directory to persist cache to disk (None for in-memory only)
        """
        # Cache storage: maps query hash to cache entry
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        
        # Concurrency control
        self._lock = Lock()
        
        # Initialize quantized embedding generator for faster semantic matching
        self.embedding_generator = QuantizedEmbeddingGenerator()
        
        # Load cache from disk if available
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
            
        logger.info(
            f"Initialized semantic cache with max size {max_size}, "
            f"similarity threshold {similarity_threshold}, "
            f"TTL {ttl_hours} hours"
        )
    
    def _generate_key(self, query: str) -> str:
        """Generate a cache key from the query.
        
        Args:
            query: User's query
            
        Returns:
            Cache key (MD5 hash of the query)
        """
        # Create a deterministic hash of the query
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def _generate_context_key(self, documents: List[Dict[str, Any]]) -> str:
        """Generate a key representing the context documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Context key (MD5 hash of document IDs)
        """
        # Extract document IDs or content hashes
        doc_ids = []
        for doc in documents:
            # Use document ID if available, otherwise hash the text
            if 'id' in doc:
                doc_ids.append(str(doc['id']))
            else:
                text = doc.get('text', '')[:100]  # Use first 100 chars
                doc_ids.append(hashlib.md5(text.encode('utf-8')).hexdigest())
        
        # Sort for deterministic order
        doc_ids.sort()
        
        # Create a hash of the document IDs
        return hashlib.md5(','.join(doc_ids).encode('utf-8')).hexdigest()
    
    def _is_entry_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired.
        
        Args:
            entry: Cache entry
            
        Returns:
            True if the entry has expired, False otherwise
        """
        # Get the creation time of the entry
        created_at = entry.get('created_at', datetime.now())
        
        # Check if the entry has expired
        return datetime.now() > created_at + timedelta(hours=self.ttl_hours)
    
    def _compute_similarity(self, query_embedding: List[float], cached_embedding: List[float]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            query_embedding: Embedding of the current query
            cached_embedding: Embedding of a cached query
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Convert to numpy arrays
        a = np.array(query_embedding)
        b = np.array(cached_embedding)
        
        # Compute cosine similarity: dot(a, b) / (norm(a) * norm(b))
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _find_similar_query(
        self,
        query_embedding: List[float],
        context_key: str
    ) -> Optional[str]:
        """Find a similar query in the cache.
        
        Args:
            query_embedding: Embedding of the current query
            context_key: Key representing the context documents
            
        Returns:
            Cache key of a similar query, or None if no similar query is found
        """
        best_similarity = 0.0
        best_key = None
        
        # Iterate through all cache entries
        for key, entry in self.cache.items():
            # Skip entries with different context
            if entry.get('context_key') != context_key:
                continue
                
            # Skip expired entries
            if self._is_entry_expired(entry):
                continue
                
            # Get the cached query embedding
            cached_embedding = entry.get('query_embedding')
            if not cached_embedding:
                continue
                
            # Compute similarity
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            
            # Update best match if this is better
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        if best_key:
            logger.debug(f"Found similar query with similarity {best_similarity:.4f}")
            
        return best_key
    
    def get(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Get a response from the cache.
        
        Args:
            query: User's query
            documents: Retrieved documents
            system_prompt: System prompt
            
        Returns:
            Cached response or None if not found
        """
        # Generate exact match key
        key = self._generate_key(query)
        
        # Generate context key
        context_key = self._generate_context_key(documents)
        
        with self._lock:
            # Check for exact match
            entry = self.cache.get(key)
            if entry and entry.get('context_key') == context_key and not self._is_entry_expired(entry):
                self.hits += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                return entry.get('response')
            
            # No exact match, try semantic matching
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Find similar query
            similar_key = self._find_similar_query(query_embedding, context_key)
            if similar_key:
                self.semantic_hits += 1
                logger.info(f"Semantic cache hit for query: {query[:50]}...")
                
                # Get the cached response
                similar_entry = self.cache.get(similar_key)
                if similar_entry:
                    return similar_entry.get('response')
            
            # No match found
            self.misses += 1
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
    
    def set(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        response: str,
        system_prompt: Optional[str] = None
    ) -> None:
        """Add a response to the cache.
        
        Args:
            query: User's query
            documents: Retrieved documents
            response: Response to cache
            system_prompt: System prompt
        """
        # Generate keys
        key = self._generate_key(query)
        context_key = self._generate_context_key(documents)
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        with self._lock:
            # If cache is full, remove oldest item (simple LRU strategy)
            if len(self.cache) >= self.max_size:
                # Find oldest entry
                oldest_key = None
                oldest_time = datetime.now()
                
                for k, entry in self.cache.items():
                    created_at = entry.get('created_at', datetime.now())
                    if created_at < oldest_time:
                        oldest_time = created_at
                        oldest_key = k
                
                # Remove oldest entry
                if oldest_key:
                    self.cache.pop(oldest_key)
            
            # Add new entry
            self.cache[key] = {
                'query': query,
                'query_embedding': query_embedding,
                'context_key': context_key,
                'response': response,
                'system_prompt': system_prompt,
                'created_at': datetime.now()
            }
            
            logger.debug(f"Cached response for query: {query[:50]}...")
            
            # Persist cache to disk if configured
            if self.cache_dir:
                self._save_cache()
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.semantic_hits = 0
            
            # Remove cache files if persisting to disk
            if self.cache_dir:
                for cache_file in self.cache_dir.glob("semantic_cache_*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_file}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.semantic_hits + self.misses
            hit_rate = (self.hits + self.semantic_hits) / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "semantic_hits": self.semantic_hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "similarity_threshold": self.similarity_threshold,
                "ttl_hours": self.ttl_hours
            }
    
    def _save_cache(self) -> None:
        """Save the cache to disk."""
        if not self.cache_dir:
            return
            
        try:
            # Create a timestamp with microseconds for uniqueness
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            microseconds = now.microsecond
            unique_id = f"{timestamp}_{microseconds:06d}"
            
            # Define cache file paths
            cache_file = self.cache_dir / f"semantic_cache_{unique_id}.pkl"
            temp_file = self.cache_dir / f"semantic_cache_{unique_id}.tmp"
            
            # Ensure the temp file doesn't exist (should be extremely unlikely with microsecond precision)
            if temp_file.exists():
                temp_file = self.cache_dir / f"semantic_cache_{unique_id}_{os.getpid()}.tmp"
            
            # Save the cache to a temporary file
            with open(temp_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            # Safely rename the temporary file to the final file
            # If the target exists (extremely unlikely), use a different name
            if cache_file.exists():
                cache_file = self.cache_dir / f"semantic_cache_{unique_id}_{os.getpid()}.pkl"
                
            try:
                temp_file.rename(cache_file)
            except OSError:
                # If rename fails (can happen on Windows), try copy and delete
                import shutil
                shutil.copy2(temp_file, cache_file)
                temp_file.unlink()
            
            # Remove old cache files, keeping only the 3 most recent
            # This helps prevent issues with concurrent access
            cache_files = sorted(self.cache_dir.glob("semantic_cache_*.pkl"))
            for old_file in cache_files[:-3]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logger.error(f"Error removing old cache file {old_file}: {str(e)}")
                    
            logger.debug(f"Saved cache to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache to disk: {str(e)}")
            # Log more detailed error information for debugging
            import traceback
            logger.debug(f"Cache save error details: {traceback.format_exc()}")
            # Continue execution - don't let cache errors break the application
    
    def _load_cache(self) -> None:
        """Load the cache from disk."""
        if not self.cache_dir:
            return
            
        try:
            # Find the most recent cache file
            cache_files = sorted(self.cache_dir.glob("semantic_cache_*.pkl"))
            if not cache_files:
                logger.info("No cache file found")
                return
                
            cache_file = cache_files[-1]
            
            # Load the cache from the file
            with open(cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                
            # Update the cache
            self.cache.update(loaded_cache)
            
            # Remove expired entries
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_entry_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.cache.pop(key)
                
            logger.info(f"Loaded {len(loaded_cache)} entries from {cache_file}, "
                       f"removed {len(expired_keys)} expired entries")
            
        except Exception as e:
            logger.error(f"Error loading cache from disk: {str(e)}")


# Create a global instance for easy access
# Use a cache directory in the project root
project_root = Path(__file__).parent.parent.parent
cache_dir = project_root / "data" / "cache" / "semantic"
semantic_cache = SemanticCache(
    max_size=200,  # Increased cache size
    similarity_threshold=0.75,  # Lower threshold for better hit rate
    ttl_hours=48,  # Longer TTL for better cache utilization
    cache_dir=cache_dir
)


def get_response(
    query: str,
    documents: List[Dict[str, Any]],
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """Convenience function to get a response from the cache.
    
    Args:
        query: User's query
        documents: Retrieved documents
        system_prompt: System prompt
        
    Returns:
        Cached response or None if not found
    """
    return semantic_cache.get(query, documents, system_prompt)


def set_response(
    query: str,
    documents: List[Dict[str, Any]],
    response: str,
    system_prompt: Optional[str] = None
) -> None:
    """Convenience function to add a response to the cache.
    
    Args:
        query: User's query
        documents: Retrieved documents
        response: Response to cache
        system_prompt: System prompt
    """
    semantic_cache.set(query, documents, response, system_prompt)


def get_stats() -> Dict[str, Any]:
    """Convenience function to get cache statistics.
    
    Returns:
        Dict with cache statistics
    """
    return semantic_cache.get_stats()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the semantic cache
    cache = SemanticCache()
    
    # Create some test documents
    documents = [
        {"id": 1, "text": "This is document 1"},
        {"id": 2, "text": "This is document 2"},
        {"id": 3, "text": "This is document 3"}
    ]
    
    # Test exact match
    query1 = "What is the capital of France?"
    response1 = "The capital of France is Paris."
    
    cache.set(query1, documents, response1)
    cached_response = cache.get(query1, documents)
    
    print(f"Query: {query1}")
    print(f"Cached response: {cached_response}")
    print(f"Match: {cached_response == response1}")
    
    # Test semantic match
    query2 = "Can you tell me what the capital city of France is?"
    cached_response = cache.get(query2, documents)
    
    print(f"\nQuery: {query2}")
    print(f"Cached response: {cached_response}")
    print(f"Match: {cached_response == response1}")
    
    # Print statistics
    stats = cache.get_stats()
    print(f"\nCache statistics: {stats}")
