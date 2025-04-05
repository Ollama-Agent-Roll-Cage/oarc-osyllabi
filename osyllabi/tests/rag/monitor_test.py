"""
Unit tests for the RAGMonitor class.
"""
import unittest
import time
import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

from osyllabi.rag.monitor import RAGMonitor

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestRAGMonitor(unittest.TestCase):
    """Test cases for the RAGMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of RAGMonitor."""
        # Initialize monitor with default settings
        monitor = RAGMonitor()
        
        # Verify basic initialization
        self.assertIsInstance(monitor, RAGMonitor)
        self.assertIsInstance(monitor.metrics, dict)
        self.assertIn("retrieval", monitor.metrics)
        self.assertIn("embedding", monitor.metrics)
        self.assertIn("documents", monitor.metrics)
    
    def test_start_retrieval(self):
        """Test starting retrieval operation."""
        monitor = RAGMonitor()
        
        # Start retrieval operation
        retrieval_id = monitor.start_retrieval()
        
        # Verify retrieval was started
        self.assertIsInstance(retrieval_id, int)
        self.assertEqual(monitor.metrics["retrieval"]["count"], 1)
        self.assertEqual(len(monitor.query_history), 1)
    
    def test_record_retrieval(self):
        """Test recording retrieval metrics."""
        monitor = RAGMonitor()
        
        # First start retrieval operation
        retrieval_id = monitor.start_retrieval()
        
        # Record retrieval results
        monitor.record_retrieval(
            retrieval_id=retrieval_id,
            query="test query",
            results=[{"text": "Result 1", "similarity": 0.9}, {"text": "Result 2", "similarity": 0.8}],
            duration=0.25
        )
        
        # Verify metrics were updated
        self.assertEqual(monitor.metrics["retrieval"]["count"], 1)
        self.assertGreaterEqual(monitor.metrics["retrieval"]["total_time"], 0.25)
        self.assertEqual(monitor.metrics["retrieval"]["chunk_count"], 2)
        self.assertEqual(monitor.metrics["retrieval"]["hit_count"], 1)  # Since results were provided
    
    def test_record_embedding(self):
        """Test recording embedding metrics."""
        monitor = RAGMonitor()
        
        # Record embedding operation
        monitor.record_embedding(
            chunk_count=5,
            duration=0.3
        )
        
        # Verify metrics were updated
        self.assertEqual(monitor.metrics["embedding"]["count"], 1)
        self.assertGreaterEqual(monitor.metrics["embedding"]["total_time"], 0.3)
        self.assertEqual(monitor.metrics["embedding"]["chunk_count"], 5)
    
    def test_record_document_addition(self):
        """Test recording document addition metrics."""
        monitor = RAGMonitor()
        
        # Record document addition
        monitor.record_document_addition(chunk_count=10)
        
        # Verify metrics were updated
        self.assertEqual(monitor.metrics["documents"]["count"], 1)
        self.assertEqual(monitor.metrics["documents"]["total_chunks"], 10)
    
    def test_get_metrics(self):
        """Test getting metrics summary."""
        monitor = RAGMonitor()
        
        # Record some activity
        retrieval_id = monitor.start_retrieval()
        monitor.record_retrieval(
            retrieval_id=retrieval_id,
            query="test query",
            results=[{"text": "Result 1"}],
            duration=0.2
        )
        monitor.record_embedding(chunk_count=5, duration=0.3)
        
        # Get metrics
        metrics = monitor.get_metrics()
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn("retrieval", metrics)
        self.assertIn("embedding", metrics)
        self.assertIn("retrieval", metrics)
    
    def test_get_recent_queries(self):
        """Test getting recent query history."""
        monitor = RAGMonitor()
        
        # Add queries to history
        for i in range(5):
            retrieval_id = monitor.start_retrieval()
            monitor.record_retrieval(
                retrieval_id=retrieval_id,
                query=f"query {i}",
                results=[{"text": f"Result for query {i}"}],
                duration=0.1
            )
        
        # Get recent queries (default should be up to 10)
        queries = monitor.get_recent_queries(count=3)
        
        # Verify recent queries
        self.assertEqual(len(queries), 3)
        # Most recent should be first (if sorted by start_time in descending order)
        self.assertIn("query", queries[0].get("query", ""))
    
    def test_save_metrics(self):
        """Test saving metrics to file."""
        log_path = self.base_dir / "metrics.json"
        monitor = RAGMonitor(log_path=log_path)
        
        # Record some activity
        retrieval_id = monitor.start_retrieval()
        monitor.record_retrieval(
            retrieval_id=retrieval_id,
            query="test query",
            results=[{"text": "Result 1"}],
            duration=0.2
        )
        
        # Save metrics by forcing a call to _save_metrics
        monitor._save_metrics()
        
        # Verify file was created
        self.assertTrue(log_path.exists())
        
        # Verify file contains valid JSON
        with open(log_path, 'r') as f:
            metrics_json = json.load(f)
            
        # Check content structure
        self.assertIsInstance(metrics_json, dict)
        self.assertIn("metrics", metrics_json)
    
    def test_reset(self):
        """Test resetting metrics."""
        monitor = RAGMonitor()
        
        # Record some activity
        retrieval_id = monitor.start_retrieval()
        monitor.record_retrieval(
            retrieval_id=retrieval_id,
            query="test query",
            results=[{"text": "Result 1"}],
            duration=0.2
        )
        
        # Reset metrics
        monitor.reset()
        
        # Verify metrics were reset
        self.assertEqual(monitor.metrics["retrieval"]["count"], 0)
        self.assertEqual(monitor.metrics["retrieval"]["total_time"], 0.0)
        self.assertEqual(len(monitor.query_history), 0)


if __name__ == '__main__':
    unittest.main()
