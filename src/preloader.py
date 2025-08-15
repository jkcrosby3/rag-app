"""Preloader for the RAG system.

This module provides functionality to preload models and clients
to reduce cold start times for the first query.
"""
import logging
import threading
import time
from typing import Optional, Union, Literal, Dict, Any

from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.quantized_generator import QuantizedEmbeddingGenerator
from src.llm.llm_factory import get_llm_client, get_available_llm_types

logger = logging.getLogger(__name__)

class ModelPreloader:
    """Preloads models and clients to reduce cold start times."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance of ModelPreloader exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelPreloader, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ModelPreloader."""
        if self._initialized:
            return
            
        self.embedding_generator = None
        self.quantized_embedding_generator = None
        self.llm_client = None
        self.preload_thread = None
        self._initialized = True
        
        logger.info("ModelPreloader initialized")
    
    def preload_models(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "claude-3-5-sonnet-20241022",
        use_quantized_embeddings: bool = False,
        quantization_type: Literal["int8", "int4"] = "int8"
    ) -> None:
        """Preload models in a background thread.
        
        Args:
            embedding_model_name: Name of the embedding model to preload
            llm_model_name: Name of the LLM model to preload
            use_quantized_embeddings: Whether to preload a quantized embedding model
            quantization_type: Type of quantization to use (int8 or int4)
        """
        if self.preload_thread and self.preload_thread.is_alive():
            logger.info("Models are already being preloaded")
            return
            
        self.preload_thread = threading.Thread(
            target=self._preload_models_thread,
            args=(embedding_model_name, llm_model_name, use_quantized_embeddings, quantization_type),
            daemon=True
        )
        self.preload_thread.start()
        logger.info("Started model preloading in background thread")
    
    def _preload_models_thread(
        self,
        embedding_model_name: str,
        llm_model_name: str,
        use_quantized_embeddings: bool,
        quantization_type: str,
        llm_type: str = "claude",
        llm_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Thread function to preload models.
        
        Args:
            embedding_model_name: Name of the embedding model to preload
            llm_model_name: Name of the LLM model to preload
            use_quantized_embeddings: Whether to preload a quantized embedding model
            quantization_type: Type of quantization to use
            llm_type: Type of LLM to use (e.g., 'claude')
            llm_config: Additional LLM configuration
        """
        try:
            start_time = time.time()
            
            # Preload embedding model
            if use_quantized_embeddings:
                logger.info(f"Preloading quantized embedding model: {embedding_model_name} ({quantization_type})")
                self.quantized_embedding_generator = QuantizedEmbeddingGenerator(
                    model_name=embedding_model_name,
                    quantization_type=quantization_type
                )
                # Force model loading
                self.quantized_embedding_generator.generate_embedding("Preload test")
            else:
                logger.info(f"Preloading standard embedding model: {embedding_model_name}")
                self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
                # Force model loading
                self.embedding_generator.generate_embedding("Preload test")
            
            # Preload LLM client using factory
            logger.info(f"Preloading LLM client: {llm_type} ({llm_model_name})")
            llm_config = llm_config or {}
            llm_config['model_name'] = llm_model_name
            self.llm_client = get_llm_client(llm_type, **llm_config)
            
            preload_time = time.time() - start_time
            logger.info(f"Models preloaded successfully in {preload_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error preloading models: {str(e)}")
            raise
    
    def get_embedding_generator(self) -> Optional[EmbeddingGenerator]:
        """Get the preloaded embedding generator.
        
        Returns:
            Preloaded embedding generator or None if not loaded
        """
        return self.embedding_generator
    
    def get_quantized_embedding_generator(self) -> Optional[QuantizedEmbeddingGenerator]:
        """Get the preloaded quantized embedding generator.
        
        Returns:
            Preloaded quantized embedding generator or None if not loaded
        """
        return self.quantized_embedding_generator
    
    def get_llm_client(self) -> Optional:
        """Get the preloaded LLM client.
        
        Returns:
            Preloaded LLM client or None if not loaded
        """
        return self.llm_client
    
    def is_preloading_complete(self) -> bool:
        """Check if preloading is complete.
        
        Returns:
            True if preloading is complete, False otherwise
        """
        # Basic preloading is complete if we have the standard embedding generator and LLM client
        return (
            self.embedding_generator is not None and
            self.llm_client is not None
        )
    
    def is_quantized_preloading_complete(self) -> bool:
        """Check if quantized preloading is complete.
        
        Returns:
            True if quantized preloading is complete, False otherwise
        """
        # Quantized preloading is complete if we have the quantized embedding generator
        return self.quantized_embedding_generator is not None
    
    def wait_for_preloading(self, timeout: float = 30.0) -> bool:
        """Wait for preloading to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if preloading completed, False if timed out
        """
        start_time = time.time()
        while not self.is_preloading_complete():
            if time.time() - start_time > timeout:
                logger.warning(f"Timed out waiting for model preloading after {timeout} seconds")
                return False
            time.sleep(0.1)
        
        return True


# Global instance
preloader = ModelPreloader()


def preload_models(
    embedding_model_name: str = "all-MiniLM-L6-v2",
    llm_model_name: str = "claude-3-5-sonnet-20241022",
    use_quantized_embeddings: bool = True,
    quantization_type: str = "int8",
    llm_type: str = "claude",
    llm_config: Optional[Dict[str, Any]] = None
) -> None:
    """Preload models to reduce cold start times.
    
    Args:
        embedding_model_name: Name of the embedding model to preload
        llm_model_name: Name of the LLM model to preload
        use_quantized_embeddings: Whether to preload a quantized embedding model
        quantization_type: Type of quantization to use (int8 or int4)
    """
    preloader.preload_models(
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        use_quantized_embeddings=use_quantized_embeddings,
        quantization_type=quantization_type,
        llm_type=llm_type,
        llm_config=llm_config
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Preload models
    preload_models()
    
    # Wait for preloading to complete
    preloader.wait_for_preloading()
    
    logger.info("Preloading complete, models are ready for use")
