"""
Embedding generation for vector representations of text.
"""
import time
from typing import Any, Dict, List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from osyllabi.ai.client import OllamaClient
from osyllabi.utils.decorators.singleton import singleton
from osyllabi.utils.utils import check_for_ollama
from osyllabi.utils.log import log

@singleton
class EmbeddingGenerator:
    """
    Generate embeddings for text using Ollama's embedding API.
    
    This class manages embedding generation for converting text chunks into
    vector representations suitable for similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        cache_embeddings: bool = True,
        max_cache_size: int = 1000
    ):
        """
        Initialize the embedding generator with a model.
        
        Args:
            model_name: Name of the model to use for embeddings
            cache_embeddings: Whether to cache embeddings to avoid duplicate processing
            max_cache_size: Maximum number of embeddings to cache
            
        Raises:
            RuntimeError: If Ollama embedding API is not available
        """
        # Ensure Ollama is available - will raise RuntimeError if not
        check_for_ollama()
        
        self.model_name = model_name
        self.client = OllamaClient()
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size
        
        # Initialize embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Verify model exists and can generate embeddings
        try:
            self.client.embed("test", model=self.model_name)
            log.info(f"Successfully connected to Ollama embedding API with model {self.model_name}")
        except Exception as e:
            log.error(f"Failed to generate embeddings with model {self.model_name}: {e}")
            raise RuntimeError(f"Model {self.model_name} is not available or cannot generate embeddings")
        
        # Determine embedding dimensions by testing
        try:
            self._embedding_dim = len(self.client.embed("test", model=self.model_name))
            log.debug(f"Embedding dimension: {self._embedding_dim}")
        except Exception:
            # Use default dimension
            self._embedding_dim = 4096
            log.warning(f"Couldn't determine embedding dimension, using default: {self._embedding_dim}")
            
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception))
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []
        
        start_time = time.time()
        result = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first if enabled
        if self.cache_embeddings:
            for i, text in enumerate(texts):
                # Use a truncated version of the text as cache key
                cache_key = text[:1000]
                if cache_key in self._embedding_cache:
                    result.append(self._embedding_cache[cache_key])
                    self._cache_hits += 1
                else:
                    result.append([])  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                embeddings = self.client.embed_batch(uncached_texts, model=self.model_name)
                
                # Update results and cache
                for i, embedding in zip(uncached_indices, embeddings):
                    result[i] = embedding
                    
                    # Add to cache if enabled
                    if self.cache_embeddings and uncached_texts[uncached_indices.index(i)]:
                        cache_key = uncached_texts[uncached_indices.index(i)][:1000]
                        self._embedding_cache[cache_key] = embedding
                        
                        # Prune cache if it exceeds max size
                        if len(self._embedding_cache) > self.max_cache_size:
                            self._prune_cache()
                            
            except Exception as e:
                log.error(f"Error generating embeddings with Ollama: {e}")
                raise RuntimeError(f"Failed to generate embeddings: {e}")
        
        # Log performance stats
        duration = time.time() - start_time
        log.debug(f"Generated {len(uncached_texts)} embeddings in {duration:.2f}s " +
                 f"(cache hits: {self._cache_hits}, misses: {self._cache_misses})")
        
        return result
            
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception))
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        # Check cache first if enabled
        if self.cache_embeddings:
            cache_key = text[:1000]
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            self._cache_misses += 1
            
        try:
            embedding = self.client.embed(text, model=self.model_name)
            
            # Add to cache if enabled
            if self.cache_embeddings:
                cache_key = text[:1000]
                self._embedding_cache[cache_key] = embedding
                
                # Prune cache if it exceeds max size
                if len(self._embedding_cache) > self.max_cache_size:
                    self._prune_cache()
                    
            return embedding
            
        except Exception as e:
            log.error(f"Error generating embedding with Ollama: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
        
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self._embedding_dim
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_embeddings:
            return {"enabled": False}
            
        return {
            "enabled": True,
            "size": len(self._embedding_cache),
            "max_size": self.max_cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def _prune_cache(self) -> None:
        """Prune the embedding cache to stay within size limits."""
        # Simple LRU-like pruning - just remove oldest entries
        overflow = len(self._embedding_cache) - self.max_cache_size
        if overflow > 0:
            keys_to_remove = list(self._embedding_cache.keys())[:overflow]
            for key in keys_to_remove:
                del self._embedding_cache[key]
            log.debug(f"Pruned {overflow} entries from embedding cache")
