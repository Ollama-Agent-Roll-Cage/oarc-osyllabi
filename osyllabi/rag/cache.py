"""
Caching functionality for RAG operations.

This module provides caching mechanisms for RAG operations
to improve performance by reducing redundant computations.
"""
import time
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
import re

from osyllabi.utils.log import log


class Cache:
    """
    Base cache implementation using LRU strategy.
    
    This class implements a simple least-recently-used (LRU) cache
    with a maximum size limit.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max(max_size, 100)  # Minimum size of 100
        self.cache = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached item or None if not found
        """
        if key in self.cache:
            # Move to end to mark as recently used
            value = self.cache.pop(key)
            self.cache[key] = value
            self.stats["hits"] += 1
            return value
            
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.
        
        Args:
            key: Cache key
            value: Item to cache
        """
        # If key exists, remove it first to update its position
        if key in self.cache:
            self.cache.pop(key)
            
        # Add new item
        self.cache[key] = value
        
        # Enforce size limit - remove oldest items
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate
        }
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache


class QueryCache(Cache):
    """
    Cache for query results with time-to-live support.
    
    This cache is optimized for storing query results with automatic
    expiration based on a configurable TTL.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of queries to cache
            ttl: Time-to-live in seconds for cached entries
        """
        super().__init__(max_size)
        self.ttl = ttl  # Time to live in seconds
        self.timestamps = {}
    
    def add(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Add query results to the cache.
        
        Args:
            query: The query string
            results: List of result items
        """
        # Normalize query to improve cache hit rate
        key = self._normalize_query(query)
        
        # Store with timestamp for TTL checking
        self.timestamps[key] = time.time()
        self.set(key, results)
    
    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get results for a query from the cache.
        
        Args:
            query: The query string
            
        Returns:
            List of result items or None if not found or expired
        """
        key = self._normalize_query(query)
        
        # Check if entry exists and is not expired
        if key in self.cache:
            timestamp = self.timestamps.get(key, 0)
            if time.time() - timestamp <= self.ttl:
                return super().get(key)
            else:
                # Expired - remove from cache
                self.cache.pop(key)
                self.timestamps.pop(key)
        
        return None
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query string for consistent cache keys.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Remove extra whitespace and convert to lowercase
        query = re.sub(r'\s+', ' ', query.strip().lower())
        return query
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        super().clear()
        self.timestamps.clear()


class LRUOrderedDict(OrderedDict):
    """OrderedDict with a size limit that removes oldest items."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize with max size."""
        super().__init__()
        self.max_size = max_size
        
    def __setitem__(self, key, value):
        """Add an item, removing oldest if at capacity."""
        if key not in self and len(self) >= self.max_size:
            # Remove oldest item (first in ordered dict)
            self.popitem(last=False)
        super().__setitem__(key, value)


class DocumentCache:
    """Cache for document chunks and embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the document cache."""
        self.max_size = max(max_size, 100)  # Minimum size of 100
        self.cache = OrderedDict()  # Use OrderedDict directly
        self.embeddings_cache = OrderedDict()  # Use OrderedDict directly
        self.stats = {
            "hits": 0,
            "misses": 0
        }

    def add(self, source: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Add document chunks and embeddings to the cache."""
        if source in self.cache:
            self.remove(source)
        self.cache[source] = chunks
        self.embeddings_cache[source] = embeddings

        # Repeatedly remove oldest items if over capacity
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self.remove(oldest_key)

    def get_chunks(self, source: str) -> Optional[List[str]]:
        """Get cached chunks for a document."""
        # Simple lookup without moving to end
        if source in self.cache:
            self.stats["hits"] += 1
            return self.cache[source]
        self.stats["misses"] += 1
        return None
    
    def get_embeddings(self, source: str) -> Optional[List[List[float]]]:
        """Get cached embeddings for a document."""
        # Also simple lookup
        if source in self.embeddings_cache and source in self.cache:
            return self.embeddings_cache[source]
        return None

    def remove(self, source: str) -> None:
        """
        Remove a document from the cache.
        
        Args:
            source: Document source identifier
        """
        if source in self.cache:
            self.cache.pop(source)
            
        if source in self.embeddings_cache:
            self.embeddings_cache.pop(source)
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        self.embeddings_cache.clear()


# For backward compatibility if older code uses ResultCache
ResultCache = QueryCache
