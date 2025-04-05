"""
Tests for content extraction functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, List

from osyllabi.generator.resource.extractor import ContentExtractorABC


class MockExtractor(ContentExtractorABC):
    """Mock implementation of extractor for testing the abstract base class."""
    
    def extract(self, resource: Any) -> Dict[str, Any]:
        if isinstance(resource, str):
            return {
                "content": f"Extracted from: {resource}",
                "content_type": "text",
                "metadata": {"length": len(resource)}
            }
        return {"content": "Unknown resource", "content_type": "unknown", "metadata": {}}
    
    def supports(self, resource: Any) -> bool:
        return isinstance(resource, str)


class TestContentExtractorABC(unittest.TestCase):
    """Test cases for the content extractor abstract base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MockExtractor()
        
    def test_extract_method(self):
        """Test the basic extraction method implementation."""
        resource = "test resource"
        result = self.extractor.extract(resource)
        
        self.assertEqual(result["content"], "Extracted from: test resource")
        self.assertEqual(result["content_type"], "text")
        self.assertEqual(result["metadata"]["length"], len(resource))
        
    def test_supports_method(self):
        """Test the supports method for resource validation."""
        self.assertTrue(self.extractor.supports("string resource"))
        self.assertFalse(self.extractor.supports(123))  # Not a string
        self.assertFalse(self.extractor.supports({}))   # Not a string
        
    def test_extract_keywords(self):
        """Test keyword extraction functionality."""
        text = "This is a test document with important keywords and significant terms"
        
        # Extract keywords with default settings
        keywords = self.extractor.extract_keywords(text)
        
        # Verify basic extraction behavior
        self.assertTrue(len(keywords) <= 20)  # Default max_keywords is 20
        self.assertIn("document", keywords)
        self.assertIn("important", keywords)
        self.assertIn("significant", keywords)
        self.assertIn("keywords", keywords)
        self.assertIn("terms", keywords)
        
        # Common stop words should be excluded
        self.assertNotIn("this", keywords)
        self.assertNotIn("with", keywords)
        
        # Test with custom min_length
        keywords_longer = self.extractor.extract_keywords(text, min_length=6)
        self.assertIn("important", keywords_longer)
        self.assertIn("significant", keywords_longer)
        self.assertNotIn("terms", keywords_longer)  # "terms" is only 5 chars
        
        # Test with custom max_keywords limit
        keywords_limited = self.extractor.extract_keywords(text, max_keywords=2)
        self.assertEqual(len(keywords_limited), 2)
        
    def test_extract_keywords_with_frequencies(self):
        """Test keyword extraction with repeated terms."""
        text = "Python Python Python is a programming language. Python is used for programming."
        
        keywords = self.extractor.extract_keywords(text)
        
        # "python" should be first in the list as it appears most frequently
        self.assertEqual(keywords[0], "python")
        self.assertIn("programming", keywords)
        self.assertIn("language", keywords)
        
        # Common stop words should be excluded
        self.assertNotIn("is", keywords)
        self.assertNotIn("for", keywords)
        
    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text."""
        keywords = self.extractor.extract_keywords("")
        self.assertEqual(keywords, [])
        
        keywords = self.extractor.extract_keywords("   ")
        self.assertEqual(keywords, [])


if __name__ == '__main__':
    unittest.main()
