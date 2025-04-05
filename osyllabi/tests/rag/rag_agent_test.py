"""
Unit tests for the RAGAgent class.
"""
import unittest
import warnings
from unittest.mock import patch, MagicMock, AsyncMock

from osyllabi.rag.rag_agent import RAGAgent
from osyllabi.rag.engine import RAGEngine
from osyllabi.rag.context import ContextAssembler
from osyllabi.rag.query import QueryFormulator
from osyllabi.ai.client import OllamaClient

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestRAGAgent(unittest.TestCase):
    """Test cases for the RAGAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset any singletons or shared state
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_initialization(self, mock_client_class):
        """Test initialization of RAGAgent."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Initialize agent
        agent = RAGAgent(
            name="test_agent",
            model="test-model",
            temperature=0.5,
            max_tokens=500
        )
        
        # Verify initialization
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.model, "test-model")
        self.assertEqual(agent.temperature, 0.5)
        self.assertEqual(agent.max_tokens, 500)
        self.assertIsNone(agent.rag_engine)
        self.assertIsInstance(agent.context_assembler, ContextAssembler)
        self.assertIsInstance(agent.query_formulator, QueryFormulator)
    
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_initialization_with_components(self, mock_client_class):
        """Test initialization of RAGAgent with custom components."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_engine = MagicMock(spec=RAGEngine)
        mock_context_assembler = MagicMock(spec=ContextAssembler)
        mock_query_formulator = MagicMock(spec=QueryFormulator)
        
        # Initialize agent with custom components
        agent = RAGAgent(
            name="test_agent",
            model="test-model",
            rag_engine=mock_engine,
            context_assembler=mock_context_assembler,
            query_formulator=mock_query_formulator,
            monitor_performance=False
        )
        
        # Verify custom components were set
        self.assertEqual(agent.rag_engine, mock_engine)
        self.assertEqual(agent.context_assembler, mock_context_assembler)
        self.assertEqual(agent.query_formulator, mock_query_formulator)
    
    @patch('osyllabi.ai.client.AsyncClient')
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_set_rag_engine(self, mock_client_class, mock_async_client):
        """Test setting RAG engine."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent
        agent = RAGAgent(name="test_agent")
        
        # Create mock engine and set it
        mock_engine = MagicMock(spec=RAGEngine)
        mock_engine.run_id = "test-run-id"
        
        agent.set_rag_engine(mock_engine)
        
        # Verify engine was set
        self.assertEqual(agent.rag_engine, mock_engine)
    
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_retrieve_context(self, mock_client_class):
        """Test context retrieval."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create mock RAG engine and components
        mock_engine = MagicMock(spec=RAGEngine)
        mock_engine.retrieve.return_value = [
            {"text": "Context snippet 1", "similarity": 0.9},
            {"text": "Context snippet 2", "similarity": 0.8}
        ]
        
        mock_query_formulator = MagicMock(spec=QueryFormulator)
        mock_query_formulator.formulate_query.return_value = "Formulated query about Python"
        
        mock_context_assembler = MagicMock(spec=ContextAssembler)
        mock_context_assembler.assemble_context.return_value = "Assembled context about Python"
        
        # Create agent with mocked components
        agent = RAGAgent(
            name="test_agent",
            rag_engine=mock_engine,
            query_formulator=mock_query_formulator,
            context_assembler=mock_context_assembler
        )
        
        # Test retrieving context - use the parameters the actual implementation expects
        context = agent.retrieve_context(
            topic="Python",
            query_type="learning_path",
            skill_level="Beginner",
            top_k=2
        )
        
        # Verify interactions - use the parameter names the actual implementation expects
        mock_query_formulator.formulate_query.assert_called_with(
            topic="Python",
            query_type="learning_path",
            skill_level="Beginner",
            additional_context=None
        )
        
        # Fix: Match the actual API - query is a positional argument, not keyword
        mock_engine.retrieve.assert_called_with(
            "Formulated query about Python",  # positional argument
            top_k=2,
            threshold=0.0
        )
        
        self.assertEqual(context, "Assembled context about Python")
    
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_create_enhanced_prompt(self, mock_client_class):
        """Test creating enhanced prompt with context."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent with mock retrieve_context method
        agent = RAGAgent(name="test_agent")
        agent.retrieve_context = MagicMock(return_value="Relevant context about Python")
        
        # Test creating enhanced prompt with a simple template string
        base_prompt = "Create a Python tutorial"
        prompt = agent.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic="Python",
            query_type="tutorial",
            skill_level="Intermediate"
        )
        
        # Verify enhanced prompt includes context
        self.assertIn("Relevant context about Python", prompt)
        self.assertIn(base_prompt, prompt)
        
        # Verify retrieve_context was called
        agent.retrieve_context.assert_called_once()
    
    @patch('osyllabi.rag.rag_agent.OllamaClient')
    def test_process_invalid_input(self, mock_client_class):
        """Test processing with invalid input data."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent
        agent = RAGAgent(name="test_agent")
        
        # Test with missing required fields
        invalid_input = {
            "skill_level": "Beginner",
            "base_prompt": "Create a tutorial"
            # Missing 'topic' field
        }
        
        # Fix: Catch the ValueError instead of expecting an error message return value
        with self.assertRaises(ValueError) as cm:
            agent.process(invalid_input)
        
        # Verify the error message contains the missing field
        self.assertIn("topic", str(cm.exception).lower())


if __name__ == '__main__':
    unittest.main()
