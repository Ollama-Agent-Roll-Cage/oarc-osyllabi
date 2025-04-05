"""
Tests for resource management functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import List, Dict, Any

from osyllabi.generator.resource.manager import ResourceManager


class TestResourceManager(unittest.TestCase):
    """Test cases for the ResourceManager class."""
    
    @patch('osyllabi.generator.resource.manager.ResourceCollector')
    def setUp(self, mock_collector_class):
        """Set up test fixtures with mocked collector."""
        # Set up the mock collector
        self.mock_collector = mock_collector_class.return_value
        
        # Initialize ResourceManager with default settings
        self.manager = ResourceManager(
            max_concurrent_requests=5,
            max_file_size_mb=10.0,
            max_content_length=10000
        )
        
        # Replace the collector with our mock
        self.manager.collector = self.mock_collector
        
    def test_initialization(self):
        """Test initialization of ResourceManager."""
        self.assertEqual(self.manager.max_content_length, 10000)
        self.assertEqual(self.manager.stats["sources_processed"], 0)
        self.assertEqual(self.manager.stats["total_content_size"], 0)
        self.assertEqual(self.manager.stats["keywords_extracted"], 0)
        
    def test_process_sources_with_urls_and_paths(self):
        """Test processing sources with both URLs and file paths."""
        # Configure mock collector response
        self.mock_collector.collect.return_value = {
            "urls": {
                "http://example.com": {"title": "Example", "content": "Example content"}
            },
            "files": {
                "test_file.py": {"title": "Test File", "content": "Test content"}
            },
            "metadata": {
                "keywords": ["example", "test", "content"],
                "sources": ["example.com"]
            },
            "stats": {
                "total_content_size": 25,
                "total_urls": 1,
                "total_files": 1
            }
        }
        
        # Call process_sources
        urls = ["http://example.com"]
        paths = ["test_file.py"]
        result = self.manager.process_sources(urls=urls, paths=paths)
        
        # Verify the collector was called with the right parameters
        self.mock_collector.collect.assert_called_once_with(urls, paths)
        
        # Verify results
        self.assertEqual(len(result["urls"]), 1)
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(len(result["metadata"]["keywords"]), 3)
        
        # Verify stats were updated
        self.assertEqual(self.manager.stats["sources_processed"], 2)
        self.assertEqual(self.manager.stats["total_content_size"], 25)
        self.assertEqual(self.manager.stats["keywords_extracted"], 3)
        
    def test_process_sources_with_empty_inputs(self):
        """Test processing sources with empty inputs."""
        # Configure mock collector response
        self.mock_collector.collect.return_value = {
            "urls": {},
            "files": {},
            "metadata": {"keywords": [], "sources": []},
            "stats": {"total_content_size": 0}
        }
        
        # Call process_sources with empty lists
        result = self.manager.process_sources(urls=[], paths=[])
        
        # Verify the collector was called with empty lists
        self.mock_collector.collect.assert_called_once_with([], [])
        
        # Verify results
        self.assertEqual(len(result["urls"]), 0)
        self.assertEqual(len(result["files"]), 0)
        self.assertEqual(len(result["metadata"]["keywords"]), 0)
        
    def test_deduplication(self):
        """Test deduplication of similar content."""
        # Create resources with duplicate content
        resources = {
            "urls": {
                "http://example.com/1": {"title": "Example 1", "content": "Same content here"},
                "http://example.com/2": {"title": "Example 2", "content": "Same content here"},
                "http://example.com/3": {"title": "Example 3", "content": "Different content"}
            },
            "files": {
                "file1.txt": {"title": "File 1", "content": "Same content here"},
                "file2.txt": {"title": "File 2", "content": "Unique file content"}
            },
            "metadata": {"keywords": [], "sources": []},
            "stats": {"total_content_size": 100}
        }
        
        # Configure mock collector to return our resources
        self.mock_collector.collect.return_value = resources
        
        # Process with deduplication
        result = self.manager.process_sources(
            urls=["http://example.com/1", "http://example.com/2", "http://example.com/3"],
            paths=["file1.txt", "file2.txt"],
            deduplicate=True
        )
        
        # Verify deduplication (should have 2 unique URLs and 1 unique file)
        self.assertEqual(len(result["urls"]), 2)
        self.assertIn("http://example.com/1", result["urls"])  # First instance of duplicate content
        self.assertIn("http://example.com/3", result["urls"])  # Unique content
        self.assertNotIn("http://example.com/2", result["urls"])  # Duplicate should be removed
        self.assertEqual(len(result["files"]), 1)  # Only unique file content should remain
        
    def test_content_truncation(self):
        """Test truncation of long content."""
        # Create a simpler test that directly tests the truncation method
        long_content = "x" * 20000  # Content longer than max_content_length (10000)
        
        # Setup a test resources dictionary
        resources = {
            "urls": {
                "http://example.com": {"title": "Example", "content": long_content}
            },
            "files": {
                "long_file.txt": {"title": "Long File", "content": long_content}
            },
            "metadata": {"keywords": [], "sources": []},
            "stats": {"total_content_size": 40000}
        }
        
        # Directly call the truncation method - no mocking needed
        truncated = self.manager._truncate_content(resources)
        
        # Verify URL content was truncated
        self.assertLess(len(truncated["urls"]["http://example.com"]["content"]), 20000)
        self.assertIn("[content truncated]", truncated["urls"]["http://example.com"]["content"])
        
        # Verify file content was truncated 
        self.assertIn("long_file.txt", truncated["files"])
        self.assertLess(len(truncated["files"]["long_file.txt"]["content"]), 20000)
        self.assertIn("[content truncated]", truncated["files"]["long_file.txt"]["content"])
        
        # Verify truncation to max_content_length
        self.assertLessEqual(len(truncated["urls"]["http://example.com"]["content"]), 
                            self.manager.max_content_length + len("... [content truncated]"))
        self.assertLessEqual(len(truncated["files"]["long_file.txt"]["content"]),
                            self.manager.max_content_length + len("... [content truncated]"))
        
    def test_extract_context(self):
        """Test context extraction for prompts."""
        # Create sample resources
        resources = {
            "urls": {
                "http://example.com/1": {"title": "Example 1", "content": "Example content 1"},
                "http://example.com/2": {"title": "Example 2", "content": "Example content 2"}
            },
            "files": {
                "file1.txt": {"title": "File 1", "content": "File content 1"},
                "file2.txt": {"title": "File 2", "content": "File content 2"}
            },
            "metadata": {"keywords": ["example", "file", "content"]},
            "stats": {}
        }
        
        # Extract context
        context = self.manager.extract_context(resources, "Test Topic", max_items=2)
        
        # Verify context contains expected sections
        self.assertIn("Web Resources", context)
        self.assertIn("Local Resources", context)
        self.assertIn("Keywords", context)
        self.assertIn("Example 1", context)
        self.assertIn("Example content 1", context)
        self.assertIn("File 1", context)
        self.assertIn("File content 1", context)
        self.assertIn("example", context)
        self.assertIn("file", context)
        self.assertIn("content", context)
        
    def test_get_stats(self):
        """Test retrieving manager statistics."""
        # Set up some stats
        self.manager.stats = {
            "sources_processed": 10,
            "total_content_size": 5000,
            "keywords_extracted": 25
        }
        
        # Configure mock collector stats
        self.mock_collector.get_stats.return_value = {
            "total_urls": 5,
            "total_files": 5,
            "success_rate": 0.9
        }
        
        # Get stats
        stats = self.manager.get_stats()
        
        # Verify combined stats
        self.assertEqual(stats["sources_processed"], 10)
        self.assertEqual(stats["total_content_size"], 5000)
        self.assertEqual(stats["keywords_extracted"], 25)
        self.assertEqual(stats["total_urls"], 5)
        self.assertEqual(stats["total_files"], 5)
        self.assertEqual(stats["success_rate"], 0.9)


if __name__ == '__main__':
    unittest.main()
