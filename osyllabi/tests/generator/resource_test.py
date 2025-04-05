"""
Tests for resource collection management functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

from osyllabi.generator.resource import ResourceCollectionManager


class TestResourceCollectionManager(unittest.TestCase):
    """Test cases for the ResourceCollectionManager class."""
    
    @patch('osyllabi.generator.resource.ResourceManager')
    def setUp(self, mock_resource_manager_class):
        """Set up test fixtures with mocked ResourceManager."""
        # Configure the mock resource manager
        self.mock_resource_manager = mock_resource_manager_class.return_value
        
        # Initialize the ResourceCollectionManager
        self.manager = ResourceCollectionManager(
            max_concurrent_requests=5,
            max_file_size_mb=10.0,
            max_content_length=10000
        )
        
        # Replace the internal manager with our mock
        self.manager.manager = self.mock_resource_manager
        
    def test_initialization(self):
        """Test initialization of ResourceCollectionManager."""
        # Verify dependency flags are set
        self.assertIsInstance(self.manager.bs4_available, bool)
        self.assertIsInstance(self.manager.markdown_available, bool)
        self.assertIsInstance(self.manager.pandas_available, bool)
        self.assertIsInstance(self.manager.pymupdf_available, bool)
        self.assertIsInstance(self.manager.docx_available, bool)
        
    def test_collect_resources(self):
        """Test collecting resources."""
        # Configure mock resource manager
        self.mock_resource_manager.process_sources.return_value = {
            "urls": {"http://example.com": {"content": "Example content"}},
            "files": {"test.py": {"content": "Test content"}},
            "stats": {"total_content_size": 25}
        }
        
        # Call collect_resources
        urls = ["http://example.com"]
        paths = ["test.py"]
        result = self.manager.collect_resources(urls=urls, paths=paths)
        
        # Verify process_sources was called with correct arguments
        self.mock_resource_manager.process_sources.assert_called_once_with(
            urls=urls,
            paths=paths,
            extract_keywords=True,
            deduplicate=True
        )
        
        # Verify result is passed through unchanged
        self.assertEqual(len(result["urls"]), 1)
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(result["stats"]["total_content_size"], 25)
        
    def test_collect_resources_with_custom_options(self):
        """Test collecting resources with custom options."""
        # Configure mock resource manager
        self.mock_resource_manager.process_sources.return_value = {
            "urls": {},
            "files": {},
            "stats": {}
        }
        
        # Call collect_resources with custom options
        urls = ["http://example.com"]
        paths = ["test.py"]
        self.manager.collect_resources(
            urls=urls,
            paths=paths,
            extract_keywords=False,  # Custom option
            deduplicate=False  # Custom option
        )
        
        # Verify process_sources was called with custom arguments
        self.mock_resource_manager.process_sources.assert_called_once_with(
            urls=urls,
            paths=paths,
            extract_keywords=False,
            deduplicate=False
        )
        
    def test_extract_context_for_prompt_without_markdown(self):
        """Test context extraction without markdown available."""
        # Set markdown as unavailable
        self.manager.markdown_available = False
        
        # Mock resources
        resources = {
            "urls": {"http://example.com": {"title": "Example", "content": "Example content"}},
            "files": {"test.py": {"title": "Test File", "content": "Test content"}}
        }
        
        # Call extract_context_for_prompt
        self.manager.extract_context_for_prompt(resources, "Test Topic")
        
        # Verify extract_context was called on the internal manager
        self.mock_resource_manager.extract_context.assert_called_once_with(
            resources, "Test Topic", 5
        )
        
    @patch('markdown.markdown')
    def test_extract_context_for_prompt_with_markdown(self, mock_markdown):
        """Test enhanced context extraction with markdown available."""
        # Set markdown as available
        self.manager.markdown_available = True
        
        # Configure mock markdown function
        mock_markdown.return_value = "<h1>Heading</h1><p>Content</p>"
        
        # Mock resources
        resources = {
            "urls": {
                "http://example.com": {"title": "Example", "content": "Example content"}
            },
            "files": {
                "test.py": {"title": "Test File", "content": "Test content"}
            },
            "metadata": {
                "keywords": ["python", "test", "example"]
            }
        }
        
        # Call extract_context_for_prompt
        context = self.manager.extract_context_for_prompt(resources, "Test Topic")
        
        # Verify markdown-enhanced output
        self.assertIn("## Web Resources", context)
        self.assertIn("### Example", context)
        self.assertIn("## Local Resources", context)
        self.assertIn("### Test File", context)
        self.assertIn("## Keywords", context)
        self.assertIn("[python](#python)", context)
        
        # Verify manager's extract_context was not called (we're using enhanced version)
        self.mock_resource_manager.extract_context.assert_not_called()
        
    def test_get_stats(self):
        """Test retrieving statistics."""
        # Configure mock resource manager stats
        self.mock_resource_manager.get_stats.return_value = {
            "sources_processed": 10,
            "total_content_size": 5000,
            "success_rate": 0.9
        }
        
        # Get stats
        stats = self.manager.get_stats()
        
        # Verify stats are passed through from ResourceManager
        self.assertEqual(stats["sources_processed"], 10)
        self.assertEqual(stats["total_content_size"], 5000)
        self.assertEqual(stats["success_rate"], 0.9)


if __name__ == '__main__':
    unittest.main()
