"""Embedding generator for the RAG system.

This module provides functionality to generate embeddings for text chunks
using various embedding models.
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
import threading
import importlib.util

# Apply huggingface_hub compatibility patch before importing sentence-transformers
if importlib.util.find_spec("huggingface_hub"):
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        # Apply monkey patch for cached_download
        try:
            from src.utils.hf_compatibility import cached_download
            huggingface_hub.cached_download = cached_download
            logging.getLogger(__name__).info("Applied huggingface_hub compatibility patch for cached_download")
        except ImportError:
            logging.getLogger(__name__).warning("Could not import hf_compatibility module")

# Import the enhanced model cache
from src.embeddings.model_cache import SingletonMeta, get_embedding, set_embedding, get_stats

logger = logging.getLogger(__name__)


class EmbeddingGenerator(metaclass=SingletonMeta):
    """Generates embeddings for text chunks using various models.
    
    This class uses the Singleton pattern to ensure only one instance exists,
    keeping the model loaded in memory between queries.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the EmbeddingGenerator.

        Args:
            model_name: Name of the embedding model to use
                Default is 'all-MiniLM-L6-v2', a lightweight model from sentence-transformers
            device: Device to run the model on ('cpu', 'cuda', etc.)
                If None, will use CUDA if available, otherwise CPU
            cache_dir: Directory to cache downloaded models
                If None, will use the default cache directory
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        
        # Lazy-load the model when first needed
        
    def _load_model(self):
        """Load the embedding model if not already loaded."""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Set up kwargs for model loading
            kwargs = {}
            if self.cache_dir:
                kwargs['cache_folder'] = str(self.cache_dir)
            if self.device:
                kwargs['device'] = self.device
                
            self.model = SentenceTransformer(self.model_name, **kwargs)
            
            logger.info(f"Model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers package not installed. "
                         "Please install it with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return a zero vector of appropriate dimension
            # This is a fallback and not ideal for semantic search
            return [0.0] * self._get_embedding_dimension()
        
        # Check if embedding is in enhanced cache
        cached_embedding = get_embedding(text, self.model_name)
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for text of length {len(text)}")
            return cached_embedding
            
        # Load model if not already loaded
        start_time = time.time()
        self._load_model()
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            
            # Cache the embedding using the enhanced cache
            set_embedding(text, embedding, self.model_name)
            
            generation_time = time.time() - start_time
            logger.debug(f"Generated embedding in {generation_time:.2f} seconds")
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors (each a list of floats)
        """
        if not texts:
            logger.warning("Empty list of texts provided for embedding")
            return []
            
        start_time = time.time()
        self._load_model()
        
        try:
            # Check which texts are already in cache
            cached_embeddings = {}
            texts_to_embed = []
            
            # First pass: Check cache and identify texts that need embedding
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    # Handle empty text
                    cached_embeddings[i] = [0.0] * self._get_embedding_dimension()
                    continue
                    
                # Try to get from enhanced cache
                cached_embedding = get_embedding(text, self.model_name)
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                else:
                    texts_to_embed.append((i, text))
            
            # Generate embeddings for texts not in cache
            if texts_to_embed:
                # Use multithreading for better performance with large batches
                if len(texts_to_embed) > 10:  # Only use threading for larger batches
                    self._generate_embeddings_parallel(texts_to_embed, cached_embeddings)
                else:
                    self._generate_embeddings_sequential(texts_to_embed, cached_embeddings)
            
            # Reconstruct the results in the original order
            results = [cached_embeddings[i] for i in range(len(texts))]
            
            generation_time = time.time() - start_time
            logger.debug(f"Generated {len(texts)} embeddings in {generation_time:.2f} seconds")
            
            # Log cache statistics periodically
            if len(texts) > 5:  # Only log stats for larger batches
                cache_stats = get_stats(self.model_name)
                logger.debug(f"Cache stats: size={cache_stats['size']}, hit_rate={cache_stats['hit_rate']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
            
    def _generate_embeddings_sequential(self, texts_to_embed, cached_embeddings):
        """Generate embeddings sequentially.
        
        Args:
            texts_to_embed: List of (index, text) tuples to embed
            cached_embeddings: Dict to store results in
        """
        indices, texts_to_process = zip(*texts_to_embed)
        
        # Generate embeddings in batch
        batch_embeddings = self.model.encode(
            list(texts_to_process),
            convert_to_numpy=True
        ).tolist()
        
        # Cache the new embeddings
        for i, (idx, text) in enumerate(texts_to_embed):
            embedding = batch_embeddings[i]
            set_embedding(text, embedding, self.model_name)
            cached_embeddings[idx] = embedding
            
    def _generate_embeddings_parallel(self, texts_to_embed, cached_embeddings):
        """Generate embeddings using parallel processing for large batches.
        
        This method splits the texts into smaller batches and processes them
        in parallel to improve performance for large batches.
        
        Args:
            texts_to_embed: List of (index, text) tuples to embed
            cached_embeddings: Dict to store results in
        """
        # Define optimal batch size (tuned for performance)
        batch_size = 32
        
        # Split texts into batches
        batches = []
        current_batch = []
        
        for item in texts_to_embed:
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
                
        if current_batch:  # Add the last partial batch
            batches.append(current_batch)
            
        # Define worker function for threading
        def process_batch(batch):
            indices, texts = zip(*batch)
            embeddings = self.model.encode(list(texts), convert_to_numpy=True).tolist()
            
            # Store results in shared dict with lock
            for i, (idx, text) in enumerate(batch):
                embedding = embeddings[i]
                # Cache the embedding
                set_embedding(text, embedding, self.model_name)
                # Update the shared results dict
                with results_lock:
                    cached_embeddings[idx] = embedding
        
        # Process batches in parallel
        threads = []
        results_lock = threading.Lock()
        
        for batch in batches:
            thread = threading.Thread(target=process_batch, args=(batch,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def process_chunk_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a chunk file and add embedding.
        
        Args:
            file_path: Path to the chunk file
            
        Returns:
            Dict containing the chunk data with added embedding
        """
        file_path = Path(file_path)
        
        try:
            # Load chunk data
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                
            # Generate embedding for the chunk text
            text = chunk_data.get('text', '')
            embedding = self.generate_embedding(text)
            
            # Add embedding to chunk data
            chunk_data['embedding'] = embedding
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Error processing chunk file {file_path}: {str(e)}")
            raise


def process_chunks_with_embeddings(
    input_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\chunked",
    output_dir: Union[str, Path] = "D:\\Development\\rag-app\\data\\embedded",
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, int]:
    """Process all chunk files and generate embeddings.
    
    Args:
        input_dir: Directory containing chunked documents
        output_dir: Directory to save documents with embeddings
        model_name: Name of the embedding model to use
        
    Returns:
        Dict with statistics about the embedding process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name=model_name)
    
    stats = {
        "total_chunks_processed": 0,
        "chunks_by_topic": {}
    }
    
    # Process each chunk file
    for file_path in input_dir.glob("*.chunk*.json"):
        try:
            # Extract topic from filename
            file_name = file_path.name
            topic = "unknown"
            
            # Try to extract topic from filename
            if "glass_steagall" in file_name:
                topic = "glass_steagall"
            elif "new_deal" in file_name:
                topic = "new_deal"
            elif "sec" in file_name:
                topic = "sec"
            
            # Update stats
            stats["total_chunks_processed"] += 1
            if topic not in stats["chunks_by_topic"]:
                stats["chunks_by_topic"][topic] = 0
            stats["chunks_by_topic"][topic] += 1
            
            # Process chunk and add embedding
            chunk_with_embedding = generator.process_chunk_file(file_path)
            
            # Save to output directory
            output_file = output_dir / file_path.name.replace(".chunk", ".embedded")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_with_embedding, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Processed {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return stats


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process chunks and generate embeddings
    stats = process_chunks_with_embeddings()
    
    logger.info(f"Embedding generation complete. Processed {stats['total_chunks_processed']} chunks.")
    for topic, count in stats["chunks_by_topic"].items():
        logger.info(f"Topic '{topic}': {count} chunks processed")
