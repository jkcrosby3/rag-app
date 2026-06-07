"""Advanced model caching module for the RAG system.

This module provides advanced caching functionality for embedding models
to improve response times and reduce memory usage. Features include:

1. LRU (Least Recently Used) eviction policy for efficient cache management
2. Persistent caching to disk to preserve embeddings between runs
3. Statistics tracking for monitoring cache performance
4. Thread-safe operations for concurrent access
"""
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple, Union, cast

import numpy as np

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """Metaclass for implementing the Singleton pattern."""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """Ensure only one instance of a class exists."""
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


class EmbeddingCache(metaclass=SingletonMeta):
    """Advanced cache for text embeddings to avoid recomputing them.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - Persistent caching to disk
    - Statistics tracking
    - Thread-safe operations
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_hours: int = 24,
                 cache_dir: Optional[Union[str, Path]] = None,
                 model_name: str = "default"):
        """Initialize the advanced embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache in memory
            ttl_hours: Time-to-live for cache entries in hours
            cache_dir: Directory to persist cache to disk (None for in-memory only)
            model_name: Name of the model this cache is for (used for disk storage)
        """
        # Use OrderedDict for LRU functionality
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.model_name = model_name
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir) / "embeddings" / model_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "embedding_cache.pkl"
        else:
            self.cache_dir = None
            self.cache_file = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.total_load_time = 0
        self.total_save_time = 0
        
        # Access timestamps for TTL
        self.access_times = {}
        
        # Concurrency control
        self._lock = threading.Lock()
        
        # Load cache from disk if available
        if self.cache_dir:
            self._load_cache()
            
        logger.info(
            f"Initialized advanced embedding cache for model '{model_name}' with "
            f"max size {max_size}, TTL {ttl_hours} hours"
        )
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash key for the text.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash key for the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_entry_expired(self, key: str) -> bool:
        """Check if a cache entry has expired based on TTL.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if expired, False otherwise
        """
        if key not in self.access_times:
            return False
            
        last_access = self.access_times[key]
        return datetime.now() > last_access + timedelta(hours=self.ttl_hours)
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get an embedding from the cache with LRU update.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._hash_text(text)
        
        with self._lock:
            # Check if in memory cache and not expired
            if key in self.cache and not self._is_entry_expired(key):
                # Update access time
                self.access_times[key] = datetime.now()
                
                # Move to end of OrderedDict (most recently used)
                embedding = self.cache.pop(key)
                self.cache[key] = embedding
                
                self.hits += 1
                return embedding
            
            # If not in memory but we have disk cache, try to load from disk
            if self.cache_dir and key not in self.cache:
                disk_path = self.cache_dir / f"{key}.npy"
                if disk_path.exists():
                    try:
                        start_time = time.time()
                        embedding = np.load(disk_path).tolist()
                        load_time = time.time() - start_time
                        self.total_load_time += load_time
                        
                        # Add to memory cache
                        self.cache[key] = embedding
                        self.access_times[key] = datetime.now()
                        
                        # Enforce max size
                        if len(self.cache) > self.max_size:
                            # Remove oldest item (first in OrderedDict)
                            oldest_key, _ = self.cache.popitem(last=False)
                            if oldest_key in self.access_times:
                                del self.access_times[oldest_key]
                        
                        self.disk_hits += 1
                        logger.debug(f"Loaded embedding from disk in {load_time:.4f}s: {text[:50]}...")
                        return embedding
                    except Exception as e:
                        logger.error(f"Error loading embedding from disk: {str(e)}")
            
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Add an embedding to the cache with LRU tracking.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding to cache
        """
        key = self._hash_text(text)
        
        with self._lock:
            # If already in cache, remove it so it will be added at the end (most recent)
            if key in self.cache:
                self.cache.pop(key)
            
            # If cache is full, remove oldest item (first in OrderedDict)
            if len(self.cache) >= self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                if oldest_key in self.access_times:
                    del self.access_times[oldest_key]
            
            # Add to cache and update access time
            self.cache[key] = embedding
            self.access_times[key] = datetime.now()
            
            # Save to disk if configured
            if self.cache_dir:
                try:
                    start_time = time.time()
                    disk_path = self.cache_dir / f"{key}.npy"
                    np.save(disk_path, np.array(embedding))
                    save_time = time.time() - start_time
                    self.total_save_time += save_time
                    logger.debug(f"Saved embedding to disk in {save_time:.4f}s: {text[:50]}...")
                except Exception as e:
                    logger.error(f"Error saving embedding to disk: {str(e)}")
    
    def clear(self, clear_disk: bool = False) -> None:
        """Clear the cache.
        
        Args:
            clear_disk: Whether to also clear the disk cache
        """
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
            self.disk_hits = 0
            
            # Clear disk cache if requested
            if clear_disk and self.cache_dir:
                try:
                    for cache_file in self.cache_dir.glob("*.npy"):
                        cache_file.unlink()
                    logger.info(f"Cleared disk cache in {self.cache_dir}")
                except Exception as e:
                    logger.error(f"Error clearing disk cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """Load the cache from disk."""
        if not self.cache_dir or not self.cache_file.exists():
            return
            
        try:
            # Load the cache metadata
            with open(self.cache_file, 'rb') as f:
                metadata = pickle.load(f)
                self.access_times = metadata.get('access_times', {})
                
            # Only load the most recent max_size items
            # Sort by access time (most recent first)
            sorted_keys = sorted(
                self.access_times.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_size]
            
            # Load embeddings for the most recent items
            loaded_count = 0
            for key, _ in sorted_keys:
                disk_path = self.cache_dir / f"{key}.npy"
                if disk_path.exists():
                    try:
                        embedding = np.load(disk_path).tolist()
                        self.cache[key] = embedding
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Error loading embedding {key}: {str(e)}")
            
            logger.info(f"Loaded {loaded_count} embeddings from disk cache")
            
        except Exception as e:
            logger.error(f"Error loading cache from disk: {str(e)}")
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        if not self.cache_dir or not self.cache_file:
            return
            
        try:
            # Save metadata (access times)
            metadata = {
                'access_times': self.access_times,
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.debug(f"Saved cache metadata to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits + self.disk_hits) / total_requests if total_requests > 0 else 0
            memory_hit_rate = self.hits / total_requests if total_requests > 0 else 0
            disk_hit_rate = self.disk_hits / total_requests if total_requests > 0 else 0
            
            # Count expired entries
            expired_count = sum(1 for key in self.cache if self._is_entry_expired(key))
            
            # Estimate memory usage (rough approximation)
            # Each float is typically 8 bytes, plus overhead
            avg_embedding_size = 384  # Typical embedding dimension
            estimated_memory_kb = len(self.cache) * avg_embedding_size * 8 / 1024
            
            return {
                "model_name": self.model_name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "disk_hits": self.disk_hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "memory_hit_rate": memory_hit_rate,
                "disk_hit_rate": disk_hit_rate,
                "expired_entries": expired_count,
                "ttl_hours": self.ttl_hours,
                "estimated_memory_kb": estimated_memory_kb,
                "avg_load_time": self.total_load_time / (self.disk_hits if self.disk_hits > 0 else 1),
                "avg_save_time": self.total_save_time / (len(self.cache) if len(self.cache) > 0 else 1),
                "persistent": self.cache_dir is not None
            }
    
    def cleanup(self) -> None:
        """Remove expired entries and save metadata."""
        with self._lock:
            # Find expired keys
            expired_keys = [key for key in list(self.cache.keys()) if self._is_entry_expired(key)]
            
            # Remove expired entries
            for key in expired_keys:
                self.cache.pop(key)
                if key in self.access_times:
                    del self.access_times[key]
            
            # Save metadata
            if self.cache_dir:
                self._save_metadata()
                
            if expired_keys:
                logger.info(f"Removed {len(expired_keys)} expired entries from cache")


# Constants for default values
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MAX_CACHE_SIZE = 1000
DEFAULT_TTL_HOURS = 24

# Create the project cache directory
project_root = Path(__file__).parent.parent.parent
cache_dir = project_root / "data" / "cache"

# Create global embedding cache instances for different models
default_embedding_cache = EmbeddingCache(
    max_size=DEFAULT_MAX_CACHE_SIZE,
    ttl_hours=DEFAULT_TTL_HOURS,
    cache_dir=cache_dir,
    model_name=DEFAULT_MODEL_NAME
)

# Function to get the appropriate cache for a model
def get_cache_for_model(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingCache:
    """Get the embedding cache for a specific model.
    
    This function provides access to the appropriate cache instance for a given model.
    Currently, it returns the default cache for all models, but it's designed to be
    extended in the future to support multiple model-specific caches.
    
    Args:
        model_name: Name of the embedding model to get the cache for
        
    Returns:
        EmbeddingCache instance for the model
    """
    # Currently just returns the default cache
    # In the future, this could be expanded to support multiple model caches
    return default_embedding_cache


# Convenience function to get an embedding from the cache
def get_embedding(text: str, model_name: str = DEFAULT_MODEL_NAME) -> Optional[List[float]]:
    """Get an embedding from the cache.
    
    This function checks if an embedding for the given text already exists in the cache.
    If found, it returns the cached embedding, avoiding the need to regenerate it.
    
    Args:
        text: Text to get embedding for
        model_name: Name of the embedding model
        
    Returns:
        Cached embedding vector as a list of floats, or None if not found in cache
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided to get_embedding")
        return None
        
    cache = get_cache_for_model(model_name)
    return cache.get(text)


# Convenience function to set an embedding in the cache
def set_embedding(text: str, embedding: List[float], model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Set an embedding in the cache.
    
    This function stores an embedding vector in the cache for future retrieval,
    which helps avoid regenerating embeddings for texts that have been processed before.
    
    Args:
        text: Text to cache embedding for
        embedding: Embedding vector to cache (list of floats)
        model_name: Name of the embedding model that generated the embedding
        
    Returns:
        None
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided to set_embedding")
        return
        
    if not embedding or not isinstance(embedding, list):
        logger.warning("Invalid embedding provided to set_embedding")
        return
        
    cache = get_cache_for_model(model_name)
    cache.set(text, embedding)


# Convenience function to get cache statistics
def get_stats(model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """Get detailed cache statistics.
    
    This function provides performance metrics and usage statistics for the cache,
    which are useful for monitoring and optimization.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Dict with detailed cache statistics including:
        - size: Current number of entries in the cache
        - max_size: Maximum allowed entries
        - hit_rate: Percentage of successful cache lookups
        - memory_hit_rate: Percentage of in-memory cache hits
        - disk_hit_rate: Percentage of disk cache hits
        - estimated_memory_kb: Estimated memory usage in KB
        - and more performance metrics
    """
    cache = get_cache_for_model(model_name)
    return cache.get_stats()
