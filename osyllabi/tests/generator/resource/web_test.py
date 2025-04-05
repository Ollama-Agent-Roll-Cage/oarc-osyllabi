"""
Tests for web resource collection functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock, ANY
import requests
from pathlib import Path
import json

from osyllabi.generator.resource.web import WebResourceCollector, HTMLExtractor, JSONExtractor, TextExtractor


class TestWebResourceCollector(unittest.TestCase):
    """Test cases for the WebResourceCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = WebResourceCollector(
            max_concurrent_requests=3,
            max_content_length=5000,
            user_agent="Test User Agent"
        )
        
    def test_initialization(self):
        """Test initialization of WebResourceCollector."""
        self.assertEqual(self.collector.max_concurrent_requests, 3)
        self.assertEqual(self.collector.max_content_length, 5000)
        self.assertEqual(self.collector.user_agent, "Test User Agent")
        self.assertIn("html", self.collector.extractors)
        self.assertIn("json", self.collector.extractors)
        self.assertIn("text", self.collector.extractors)
        self.assertEqual(self.collector.stats["urls_processed"], 0)
        self.assertEqual(self.collector.stats["urls_failed"], 0)
        self.assertEqual(self.collector.stats["total_content_size"], 0)
        
    @patch('osyllabi.generator.resource.web.concurrent.futures.ThreadPoolExecutor')
    def test_collect_empty_urls(self, mock_executor_class):
        """Test collecting from an empty list of URLs."""
        result = self.collector.collect([])
        
        self.assertEqual(len(result["urls"]), 0)
        self.assertEqual(len(result["metadata"]["keywords"]), 0)
        self.assertEqual(result["stats"]["urls_processed"], 0)
        
        # Verify executor wasn't created
        mock_executor_class.assert_not_called()
        
    @patch('osyllabi.generator.resource.web.requests.get')
    def test_collect_single_url(self, mock_get):
        """Rewrite to avoid concurrency hangs."""
        # Mock a successful response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.text = "<html><head><title>Mocked Page</title></head><body>Mocked content</body></html>"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response

        # Expected result after extraction
        expected_data = {
            "title": "Mocked Page",
            "content": "Mocked content",
            "content_type": "html",
            "domain": "example.com",
            "url": "http://example.com"
        }

        # Collect from a single URL
        url = "http://example.com"
        result = self.collector.collect([url])

        # Verify requests.get was called
        mock_get.assert_called_once_with(
            url,
            headers={'User-Agent': 'Test User Agent'},
            timeout=10
        )

        # Verify that we have one URL in results
        self.assertEqual(len(result["urls"]), 1)
        self.assertIn(url, result["urls"])
        # Check a few keys from the extracted data
        self.assertEqual(result["urls"][url]["title"], expected_data["title"])
        self.assertIn(expected_data["domain"], result["metadata"]["sources"])
        
    @patch('osyllabi.generator.resource.web.requests.get')
    def test_extract_url_content_html(self, mock_get):
        """Test extracting content from an HTML URL."""
        # Setup mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.text = "<html><head><title>Test Page</title></head><body><h1>Test</h1><p>Content</p></body></html>"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response
        
        # Mock the HTML extractor
        mock_html_extractor = MagicMock()
        mock_html_extractor.supports.return_value = True
        mock_html_extractor.extract.return_value = {
            "title": "Test Page",
            "domain": "example.com",
            "content": "Test\n\nContent",
            "content_type": "html",
            "metadata": {}
        }
        
        # Replace HTML extractor with our mock
        original_html_extractor = self.collector.extractors["html"]
        self.collector.extractors["html"] = mock_html_extractor
        
        try:
            # Extract content
            result = self.collector.extract_url_content("http://example.com")
            
            # Verify requests.get was called correctly
            mock_get.assert_called_once_with(
                "http://example.com",
                headers={'User-Agent': 'Test User Agent'},
                timeout=10
            )
            
            # Verify HTML extractor was called
            mock_html_extractor.extract.assert_called_once_with(mock_response)
            
            # Verify results
            self.assertEqual(result["title"], "Test Page")
            self.assertEqual(result["content"], "Test\n\nContent")
            self.assertEqual(result["content_type"], "html")
            self.assertEqual(result["url"], "http://example.com")
        finally:
            # Restore the original extractor
            self.collector.extractors["html"] = original_html_extractor
            
    @patch('osyllabi.generator.resource.web.requests.get')
    def test_extract_url_content_json(self, mock_get):
        """Test extracting content from a JSON URL."""
        # Setup mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.return_value = {"name": "Test", "value": 42}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.url = "http://example.com/api"
        mock_get.return_value = mock_response
        
        # Mock the JSON extractor
        mock_json_extractor = MagicMock()
        mock_json_extractor.supports.return_value = True
        mock_json_extractor.extract.return_value = {
            "title": "API Response",
            "domain": "example.com",
            "content": '{"name": "Test", "value": 42}',
            "content_type": "json",
            "metadata": {"json_keys": ["name", "value"]}
        }
        
        # Replace JSON extractor with our mock
        original_json_extractor = self.collector.extractors["json"]
        self.collector.extractors["json"] = mock_json_extractor
        
        try:
            # Extract content
            result = self.collector.extract_url_content("http://example.com/api")
            
            # Verify JSON extractor was called
            mock_json_extractor.extract.assert_called_once_with(mock_response)
            
            # Verify results
            self.assertEqual(result["content_type"], "json")
            self.assertEqual(result["url"], "http://example.com/api")
        finally:
            # Restore the original extractor
            self.collector.extractors["json"] = original_json_extractor
            
    @patch('osyllabi.generator.resource.web.requests.get')
    def test_request_retry_on_failure(self, mock_get):
        """Test retry logic when requests fail."""
        # Fix 1: Create response objects first
        error_response = requests.RequestException("Connection error")
        success_response = MagicMock(spec=requests.Response)
        success_response.headers = {'Content-Type': 'text/plain'}
        success_response.url = "http://example.com"
        success_response.text = "Plain text content"
        
        # Then set side_effect
        mock_get.side_effect = [error_response, success_response]

        # Mock the text extractor
        mock_text_extractor = MagicMock()
        mock_text_extractor.supports.return_value = True
        mock_text_extractor.extract.return_value = {
            "title": "example.com",
            "domain": "example.com",
            "content": "Plain text content",
            "content_type": "text",
            "metadata": {}
        }
        
        # Replace text extractor with our mock
        original_text_extractor = self.collector.extractors["text"]
        self.collector.extractors["text"] = mock_text_extractor
        
        try:
            # Extract content - should retry and succeed
            result = self.collector.extract_url_content("http://example.com")
            
            # Verify get was called twice (once for initial failure, once for retry)
            self.assertEqual(mock_get.call_count, 2)
            
            # Verify results from successful retry
            self.assertEqual(result["content"], "Plain text content")
            self.assertEqual(result["content_type"], "text")
        finally:
            # Restore the original extractor
            self.collector.extractors["text"] = original_text_extractor
        
    def test_get_extractor_for_content_type(self):
        """Test selecting the appropriate extractor based on content type."""
        # Test HTML content type - make sure our test is accurate by checking the actual objects
        extractor = self.collector._get_extractor_for_content_type('text/html; charset=utf-8')
        self.assertIsInstance(extractor, HTMLExtractor)
        
        # Test JSON content type
        extractor = self.collector._get_extractor_for_content_type('application/json')
        self.assertIsInstance(extractor, JSONExtractor)
        
        # Test plain text content type
        extractor = self.collector._get_extractor_for_content_type('text/plain')
        self.assertIsInstance(extractor, TextExtractor)
        
        # Test unknown content type - should default to HTML
        extractor = self.collector._get_extractor_for_content_type('application/octet-stream')
        self.assertIsInstance(extractor, HTMLExtractor)
        
    def test_extract_keywords(self):
        """Test keyword extraction functionality."""
        text = "This is a test document with important keywords and significant terms"
        
        # Extract keywords
        keywords = self.collector._extract_keywords(text)
        
        # Verify results
        self.assertTrue(len(keywords) <= 20)  # Default max_keywords is 20
        self.assertIn("document", keywords)
        self.assertIn("important", keywords)
        self.assertIn("significant", keywords)
        self.assertIn("keywords", keywords)
        self.assertIn("terms", keywords)
        
        # Common stop words should be excluded
        self.assertNotIn("this", keywords)
        self.assertNotIn("with", keywords)
        
    def test_get_stats(self):
        """Test retrieving collector statistics."""
        # Set up some stats
        self.collector.stats = {
            "urls_processed": 5,
            "urls_failed": 2,
            "total_content_size": 3000
        }
        
        # Get stats
        stats = self.collector.get_stats()
        
        # Verify stats are returned correctly
        self.assertEqual(stats["urls_processed"], 5)
        self.assertEqual(stats["urls_failed"], 2)
        self.assertEqual(stats["total_content_size"], 3000)


class TestHTMLExtractor(unittest.TestCase):
    """Test cases for the HTMLExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HTMLExtractor()
        
    def test_supports_method(self):
        """Test supports method for HTML responses."""
        # Test with HTML response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
        self.assertTrue(self.extractor.supports(mock_response))
        
        # Test with non-HTML response
        mock_response.headers = {'Content-Type': 'application/json'}
        self.assertFalse(self.extractor.supports(mock_response))
        
    @patch('osyllabi.generator.resource.web.BeautifulSoup')
    def test_extract_html_content(self, mock_bs4_class):
        """Test extracting content from HTML response."""
        # Create a mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.text = "<html><head><title>Test Page</title></head><body><h1>Test Heading</h1><p>Test content</p></body></html>"
        mock_response.url = "http://example.com"
        mock_response.headers = {'Content-Type': 'text/html', 'Content-Length': '100'}
        
        # Create a simple BeautifulSoup mock to avoid complex patching
        # Configure mock soup methods
        mock_soup = MagicMock()
        mock_bs4_class.return_value = mock_soup
        
        # Setup soup title attribute
        mock_title = MagicMock()
        mock_title.string = "Test Page"
        mock_soup.title = mock_title
        
        # Configure soup.find_all to return elements for text extraction
        mock_h1 = MagicMock()
        mock_h1.name = "h1"
        mock_h1.get_text.return_value = "Test Heading"
        
        mock_p = MagicMock()
        mock_p.name = "p" 
        mock_p.get_text.return_value = "Test content"
        
        mock_soup.find_all.return_value = [mock_h1, mock_p]
        
        # Fix 2: Setup mock meta tag correctly
        mock_meta = MagicMock()
        mock_meta.get.return_value = 'Test description'  # Use get() instead of attrs
        mock_soup.find.return_value = mock_meta
        
        # Extract content
        result = self.extractor.extract(mock_response)
        
        # Verify BeautifulSoup was called correctly
        mock_bs4_class.assert_called_with(mock_response.text, ANY)
        
        # Verify results - only basic checks to avoid dependency on specific implementation
        self.assertEqual(result["title"], "Test Page")
        self.assertEqual(result["domain"], "example.com")
        self.assertEqual(result["content_type"], "html")
        self.assertIsInstance(result["content"], str)
        self.assertIn("Content-Type", result["metadata"])


class TestJSONExtractor(unittest.TestCase):
    """Test cases for the JSONExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = JSONExtractor()
        
    def test_supports_method(self):
        """Test supports method for JSON responses."""
        # Test with JSON response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.headers = {'Content-Type': 'application/json'}
        self.assertTrue(self.extractor.supports(mock_response))
        
        # Test with non-JSON response
        mock_response.headers = {'Content-Type': 'text/html'}
        self.assertFalse(self.extractor.supports(mock_response))
        
    def test_extract_json_content(self):
        """Test extracting content from JSON response."""
        # Create a mock response with JSON data
        mock_response = MagicMock(spec=requests.Response)
        mock_json_data = {
            "name": "Test Item",
            "description": "This is a test",
            "values": [1, 2, 3]
        }
        mock_response.json.return_value = mock_json_data
        mock_response.url = "http://example.com/api"
        mock_response.headers = {'Content-Type': 'application/json'}
        
        # Extract content
        result = self.extractor.extract(mock_response)
        
        # Verify JSON was processed correctly
        self.assertEqual(result["domain"], "example.com")  # Domain extracted from URL
        self.assertEqual(result["content_type"], "json")   # Content type is set correctly
        
        # Verify JSON data is in the content
        self.assertIn("Test Item", result["content"])
        self.assertIn("values", result["content"])
        
        # Check that metadata includes JSON keys
        for key in mock_json_data.keys():
            self.assertIn(key, result["metadata"]["json_keys"])
        
    def test_extract_invalid_json(self):
        """Test handling of invalid JSON responses."""
        # Fix 3: Use JSONDecodeError instead of ValueError
        from json import JSONDecodeError
        
        # Create a mock response that raises JSONDecodeError
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.side_effect = JSONDecodeError("Invalid JSON", "", 0)  # Proper JSONDecodeError
        mock_response.text = "Invalid JSON content"
        mock_response.url = "http://example.com/api"
        mock_response.headers = {'Content-Type': 'application/json'}
        
        try:
            # Extract content
            result = self.extractor.extract(mock_response)
            
            # Verify fallback to text handling
            self.assertEqual(result["content_type"], "text")
            self.assertEqual(result["content"], "Invalid JSON content")
            self.assertIn("parse_error", result["metadata"])
        except Exception as e:
            self.fail(f"Should handle JSON decode error gracefully, but got: {e}")


class TestTextExtractor(unittest.TestCase):
    """Test cases for the TextExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()
        
    def test_supports_method(self):
        """Test supports method for text responses."""
        # Test with text response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.headers = {'Content-Type': 'text/plain'}
        self.assertTrue(self.extractor.supports(mock_response))
        
        # Test with non-text response
        mock_response.headers = {'Content-Type': 'application/json'}
        self.assertFalse(self.extractor.supports(mock_response))
        
    def test_extract_text_content(self):
        """Test extracting content from text response."""
        # Create a mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.text = "This is plain text content for testing."
        mock_response.url = "http://example.com/text"
        mock_response.headers = {'Content-Type': 'text/plain'}
        
        # Extract content
        result = self.extractor.extract(mock_response)
        
        # Verify text was handled correctly
        self.assertEqual(result["domain"], "example.com")
        self.assertEqual(result["content"], "This is plain text content for testing.")
        self.assertEqual(result["content_type"], "text")


if __name__ == '__main__':
    unittest.main()
