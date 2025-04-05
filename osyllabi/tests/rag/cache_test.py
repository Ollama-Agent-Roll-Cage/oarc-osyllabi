"""
Unit tests for the RAG cache functionality.
"""
import unittest
import time
import warnings
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import all classes from cache to see what's available
from osyllabi.rag.cache import *

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestQueryCache(unittest.TestCase):
    """Test cases for the QueryCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of QueryCache."""
        # Initialize cache
        cache = QueryCache(max_size=100, ttl=3600)
        
        # Verify initialization
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(cache.ttl, 3600)
        self.assertEqual(len(cache.cache), 0)
    
    def test_add_and_get_with_expiration(self):
        """Test adding and retrieving items with TTL."""
        # Create cache with short TTL for testing
        cache = QueryCache(max_size=100, ttl=0.1)  # 100ms TTL
        
        # Add item to cache
        cache.add("test_query", [{"text": "Result 1"}, {"text": "Result 2"}])
        
        # Verify we can get it immediately
        results = cache.get("test_query")
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)
        
        # Wait for TTL to expire
        time.sleep(0.15)  # Wait longer than TTL
        
        # Verify the item has expired
        results = cache.get("test_query")
        self.assertIsNone(results)
    
    def test_normalize_query(self):
        """Test query normalization for cache keys."""
        cache = QueryCache()
        
        # Test normalization of different queries that should map to same key
        query1 = "  Python programming  "
        query2 = "python programming"
        query3 = "Python  Programming"
        
        # Add item with first query
        cache.add(query1, ["result"])
        
        # Verify we can get it with all variations
        self.assertIsNotNone(cache.get(query1))
        self.assertIsNotNone(cache.get(query2))
        self.assertIsNotNone(cache.get(query3))


class TestDocumentCache(unittest.TestCase):
    """Test cases for the DocumentCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of DocumentCache."""
        # Initialize cache
        cache = DocumentCache(max_size=100)
        
        # Verify initialization
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(len(cache.cache), 0)
    
    def test_add_and_get_document_chunks(self):
        """Test storing and retrieving document chunks."""
        cache = DocumentCache(max_size=100)
        
        # Add document chunks
        source = "test_doc.txt"
        chunks = ["Chunk 1", "Chunk 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        cache.add(source, chunks, embeddings)
        
        # Get chunks
        retrieved_chunks = cache.get_chunks(source)
        self.assertIsNotNone(retrieved_chunks)
        self.assertEqual(len(retrieved_chunks), 2)
        self.assertEqual(retrieved_chunks[0], "Chunk 1")
        self.assertEqual(retrieved_chunks[1], "Chunk 2")
        
        # Get embeddings
        retrieved_embeddings = cache.get_embeddings(source)
        self.assertIsNotNone(retrieved_embeddings)
        self.assertEqual(len(retrieved_embeddings), 2)
        self.assertEqual(retrieved_embeddings[0], [0.1, 0.2])
        self.assertEqual(retrieved_embeddings[1], [0.3, 0.4])
    
    def test_document_not_in_cache(self):
        """Test retrieving document not in cache."""
        cache = DocumentCache(max_size=100)
        
        # Get non-existent document
        chunks = cache.get_chunks("non_existent.txt")
        embeddings = cache.get_embeddings("non_existent.txt")
        
        self.assertIsNone(chunks)
        self.assertIsNone(embeddings)
    
    def test_remove_document(self):
        """Test removing document from cache."""
        cache = DocumentCache(max_size=100)
        
        # Add document
        source = "test_doc.txt"
        chunks = ["Chunk 1", "Chunk 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        cache.add(source, chunks, embeddings)
        
        # Verify document exists
        self.assertIsNotNone(cache.get_chunks(source))
        
        # Remove document
        cache.remove(source)
        
        # Verify document was removed
        self.assertIsNone(cache.get_chunks(source))


if __name__ == '__main__':
    unittest.main()
