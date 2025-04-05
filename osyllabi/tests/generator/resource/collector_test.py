"""
Tests for resource collection functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import List, Dict, Any

from osyllabi.generator.resource.collector import CollectorABC, ResourceCollector
from osyllabi.generator.resource.web import WebResourceCollector
from osyllabi.generator.resource.file import FileResourceCollector  # Keep this import

class MockCollector(CollectorABC):
    """Mock implementation of collector for testing the abstract base class."""
    
    def __init__(self):
        self.sources_processed = []
        self.stats = {"processed": 0, "failed": 0}
        
    def collect(self, sources: List[str]) -> Dict[str, Any]:
        self.sources_processed.extend(sources)
        self.stats["processed"] += len(sources)
        return {"results": sources, "stats": self.stats}
        
    def get_stats(self) -> Dict[str, Any]:
        return self.stats


class TestCollectorABC(unittest.TestCase):
    """Test cases for the collector abstract base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = MockCollector()
        
    def test_collect_method(self):
        """Test the basic collection method implementation."""
        sources = ["source1", "source2"]
        result = self.collector.collect(sources)
        
        self.assertEqual(self.collector.sources_processed, sources)
        self.assertEqual(result["results"], sources)
        self.assertEqual(result["stats"]["processed"], 2)
        
    def test_get_stats_method(self):
        """Test retrieving statistics."""
        # First collect some data to generate stats
        self.collector.collect(["source1", "source2"])
        
        stats = self.collector.get_stats()
        self.assertEqual(stats["processed"], 2)
        self.assertEqual(stats["failed"], 0)


class TestResourceCollector(unittest.TestCase):
    """Test cases for the main ResourceCollector class."""
    
    @patch('osyllabi.generator.resource.web.WebResourceCollector')
    @patch('osyllabi.generator.resource.file.FileResourceCollector')  # Updated patch path
    def setUp(self, mock_file_collector_class, mock_web_collector_class):
        """Set up test fixtures with mocked collectors."""
        # Configure mocks
        self.mock_web_collector = mock_web_collector_class.return_value
        self.mock_file_collector = mock_file_collector_class.return_value
        
        # Configure mock statistics
        self.mock_web_collector.get_stats.return_value = {
            "urls_processed": 2,
            "urls_failed": 0,
            "total_content_size": 1000
        }
        self.mock_file_collector.get_stats.return_value = {
            "files_processed": 3,
            "files_failed": 1,
            "total_content_size": 2000
        }
        
        # Initialize collector with mocks already injected by the patch
        self.collector = ResourceCollector(
            max_concurrent_requests=5,
            max_file_size_mb=10.0
        )
        
        # Override the collectors with our mocks
        self.collector.web_collector = self.mock_web_collector
        self.collector.file_collector = self.mock_file_collector
        
    def test_initialization(self):
        """Test collector initialization."""
        self.assertEqual(self.collector.stats["total_urls"], 0)
        self.assertEqual(self.collector.stats["total_files"], 0)
        self.assertEqual(self.collector.stats["total_content_size"], 0)
        self.assertEqual(self.collector.stats["success_rate"], 0.0)
        
    def test_collect_urls_only(self):
        """Test collecting only from URLs."""
        # Configure mock collector responses
        self.mock_web_collector.collect.return_value = {
            "urls": {
                "http://example.com": {"content": "Example content", "title": "Example"}
            },
            "metadata": {
                "keywords": ["example", "test"],
                "sources": ["example.com"]
            },
            "stats": {"urls_processed": 1, "total_content_size": 14}
        }
        
        # Call collect with only URLs
        urls = ["http://example.com"]
        result = self.collector.collect(urls, [])
        
        # Verify web collector was called correctly
        self.mock_web_collector.collect.assert_called_once_with(urls)
        self.mock_file_collector.collect.assert_not_called()
        
        # Verify results are processed correctly
        self.assertEqual(len(result["urls"]), 1)
        self.assertIn("http://example.com", result["urls"])
        self.assertEqual(self.collector.stats["total_urls"], 1)
        
    def test_collect_files_only(self):
        """Test collecting only from files."""
        # Configure mock collector responses
        self.mock_file_collector.collect.return_value = {
            "files": {
                "test.txt": {"content": "File content", "title": "Test File"}
            },
            "metadata": {
                "keywords": ["file", "test"]
            },
            "stats": {"files_processed": 1, "total_content_size": 11}
        }
        
        # Call collect with only file paths
        paths = ["test.txt"]
        result = self.collector.collect([], paths)
        
        # Verify file collector was called correctly
        self.mock_file_collector.collect.assert_called_once_with(paths)
        self.mock_web_collector.collect.assert_not_called()
        
        # Verify results are processed correctly
        self.assertEqual(len(result["files"]), 1)
        self.assertIn("test.txt", result["files"])
        self.assertEqual(self.collector.stats["total_files"], 1)
        
    def test_collect_both_urls_and_files(self):
        """Test collecting from both URLs and files."""
        # Configure mock collector responses
        self.mock_web_collector.collect.return_value = {
            "urls": {
                "http://example.com": {"content": "Example content", "title": "Example"}
            },
            "metadata": {
                "keywords": ["example", "test"],
                "sources": ["example.com"]
            },
            "stats": {"urls_processed": 1, "total_content_size": 14}
        }
        
        self.mock_file_collector.collect.return_value = {
            "files": {
                "test.txt": {"content": "File content", "title": "Test File"}
            },
            "metadata": {
                "keywords": ["file", "test"]
            },
            "stats": {"files_processed": 1, "total_content_size": 11}
        }
        
        # Call collect with both URLs and file paths
        urls = ["http://example.com"]
        paths = ["test.txt"]
        result = self.collector.collect(urls, paths)
        
        # Verify both collectors were called correctly
        self.mock_web_collector.collect.assert_called_once_with(urls)
        self.mock_file_collector.collect.assert_called_once_with(paths)
        
        # Verify results are processed correctly
        self.assertEqual(len(result["urls"]), 1)
        self.assertEqual(len(result["files"]), 1)
        self.assertIn("http://example.com", result["urls"])
        self.assertIn("test.txt", result["files"])
        
        # Check that keywords are combined and deduplicated
        self.assertIn("example", result["metadata"]["keywords"])
        self.assertIn("file", result["metadata"]["keywords"])
        self.assertIn("test", result["metadata"]["keywords"])
        
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Configure mock collector responses with specific stats for this test
        self.mock_web_collector.get_stats.return_value = {
            "urls_processed": 8,
            "urls_failed": 2,
            "total_content_size": 1000
        }
        
        self.mock_file_collector.get_stats.return_value = {
            "files_processed": 6,
            "files_failed": 4,
            "total_content_size": 2000
        }
        
        # Call collect to trigger stats calculation
        self.collector.collect(["http://example.com"], ["test.txt"])
        
        # Calculate expected success rate: (8 + 6) / (8 + 2 + 6 + 4) = 14/20 = 0.7
        expected_success_rate = 0.7
        
        # Verify success rate is calculated correctly
        self.assertAlmostEqual(self.collector.stats["success_rate"], expected_success_rate, places=2)
        
    def test_get_stats(self):
        """Test retrieving collector statistics."""
        # Setup some stats by collecting data
        self.collector.stats = {
            "total_urls": 5,
            "total_files": 3,
            "total_content_size": 8000,
            "success_rate": 0.8
        }
        
        # Get stats
        stats = self.collector.get_stats()
        
        # Verify stats are returned correctly
        self.assertEqual(stats["total_urls"], 5)
        self.assertEqual(stats["total_files"], 3)
        self.assertEqual(stats["total_content_size"], 8000)
        self.assertEqual(stats["success_rate"], 0.8)


if __name__ == '__main__':
    unittest.main()
