"""
Embedding generation for RAG capabilities in Osyllabi.

This module provides functionality for generating embeddings from text
using Ollama's embedding API, to support vector-based retrieval.
"""
from typing import List

from osyllabi.utils.log import log
from osyllabi.ai.client import OllamaClient


class EmbeddingGenerator:
    """
    Generate embeddings for text using Ollama's embedding API.
    
    This class manages embedding generation and provides methods to
    generate vector representations of text chunks.
    """
    
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the embedding generator with a model.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        self.model_name = model_name
        self.client = OllamaClient()
        
        # Test if Ollama is available
        self.ollama_available = self._test_ollama()
        if not self.ollama_available:
            log.warning(f"Ollama embedding API not available. Using fallback embedding method.")
        
    def _test_ollama(self) -> bool:
        """Test if Ollama's embedding API is available."""
        try:
            # Try to get an embedding for a simple test text
            self.client.embed("test", model=self.model_name)
            log.info(f"Successfully connected to Ollama embedding API with model {self.model_name}")
            return True
        except Exception as e:
            log.error(f"Failed to connect to Ollama embedding API: {e}")
            return False
            
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        if self.ollama_available:
            # Use Ollama embedding API
            try:
                return self.client.embed_batch(texts, model=self.model_name)
            except Exception as e:
                log.error(f"Ollama embedding failed: {e}. Falling back to hash-based method.")
                return [self._fallback_embed(text) for text in texts]
        else:
            # Fallback to simple hash-based embedding
            return [self._fallback_embed(text) for text in texts]
            
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        if self.ollama_available:
            try:
                return self.client.embed(text, model=self.model_name)
            except Exception as e:
                log.error(f"Ollama embedding failed: {e}. Falling back to hash-based method.")
                return self._fallback_embed(text)
        else:
            return self._fallback_embed(text)
            
    def _fallback_embed(self, text: str) -> List[float]:
        """
        Generate a simple fallback embedding when Ollama is unavailable.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple placeholder embedding
        """
        import hashlib
        import numpy as np
        
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode("utf-8"))
        hash_bytes = hash_obj.digest()
        
        # Convert hash to array of 16 integers
        hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
        
        # Expand to 4096 dimensions (typical for LLM embeddings)
        expanded = np.zeros(4096, dtype=np.float32)
        for i, val in enumerate(hash_array):
            idx = i * (4096 // len(hash_array))
            expanded[idx:idx + (4096 // len(hash_array))] = val / 255.0
            
        # Normalize
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded = expanded / norm
            
        return expanded.tolist()
        
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        if self.ollama_available:
            # Get a sample embedding to determine dimension
            try:
                sample = self.embed_text("test")
                return len(sample)
            except:
                pass
        
        # Default dimension for Ollama embeddings
        return 4096
