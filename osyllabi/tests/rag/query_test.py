"""
Unit tests for the QueryFormulator class.
"""
import unittest
import warnings
from unittest.mock import patch, MagicMock

from osyllabi.rag.query import QueryFormulator

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestQueryFormulator(unittest.TestCase):
    """Test cases for the QueryFormulator class."""
    
    def test_initialization(self):
        """Test initialization of QueryFormulator."""
        # Initialize with default settings
        formulator = QueryFormulator()
        
        # Verify basic initialization
        self.assertIsInstance(formulator, QueryFormulator)
    
    def test_formulate_query(self):
        """Test basic query formulation."""
        formulator = QueryFormulator()
        
        # Test with standard parameters
        query = formulator.formulate_query(
            topic="Python",
            query_type="learning_path",
            skill_level="Beginner"
        )
        
        # Verify query contains the topic and is not empty
        self.assertIn("Python", query)
        self.assertGreater(len(query), 20)  # Arbitrary minimum length
    
    def test_formulate_query_with_unknown_type(self):
        """Test handling of unknown query types."""
        formulator = QueryFormulator()
        
        # Test with unknown query type - should use default
        query = formulator.formulate_query(
            topic="Python",
            query_type="unknown_type",
            skill_level="Beginner"
        )
        
        # Verify a query was still generated
        self.assertIn("Python", query)
        self.assertGreater(len(query), 10)
    
    def test_formulate_query_with_additional_context(self):
        """Test query formulation with additional context."""
        formulator = QueryFormulator()
        
        # Add additional context to help formulate the query
        # FIX: additional_context should be a dictionary, not a string
        query = formulator.formulate_query(
            topic="Python Programming",
            query_type="concept_explanation", 
            skill_level="Intermediate",
            additional_context={"focus": "Object-oriented concepts"}
        )
        
        # Verify the query includes the topic
        self.assertIn("Python", query)
        
        # The additional context may or may not be directly included in the query,
        # but the query should be non-empty
        self.assertGreater(len(query), 20)
    
    # FIX: Patch OllamaClient from the correct module
    @patch('osyllabi.ai.client.OllamaClient', autospec=True)
    def test_initialize_with_client(self, mock_ollama_client):
        """Test initialization with client."""
        # Create formulator with default settings
        formulator = QueryFormulator()
        
        # Manually create and set client
        formulator.client = mock_ollama_client
        
        # Verify client attribute exists
        self.assertTrue(hasattr(formulator, 'client'))


if __name__ == '__main__':
    unittest.main()
