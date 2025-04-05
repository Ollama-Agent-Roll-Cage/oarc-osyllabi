"""
Unit tests for the RAGEngine class.
"""
import unittest
import json
import tempfile
import warnings
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from osyllabi.rag.engine import RAGEngine
from osyllabi.rag.database import VectorDatabase
from osyllabi.rag.embedding import EmbeddingGenerator
from osyllabi.rag.chunking import TextChunker

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestRAGEngine(unittest.TestCase):
    """Test cases for the RAGEngine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
        # Reset any singletons
        if hasattr(EmbeddingGenerator, '_instance'):
            delattr(EmbeddingGenerator, '_instance')
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_initialization(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, mock_db_class):
        """Test initialization of RAGEngine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_embedder = MagicMock()
        mock_embedding_class.return_value = mock_embedder
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Create directory for config file
        config_dir = self.base_dir / "test-run" / "vectors"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a real config file for the test
        config_path = config_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump({
                "run_id": "test-run",
                "embedding_model": "test-model",
                "chunk_size": 256,
                "chunk_overlap": 25,
                "created_at": 12345
            }, f)
        
        # Initialize engine
        engine = RAGEngine(
            run_id="test-run",
            base_dir=self.base_dir,
            embedding_model="test-model",
            chunk_size=256,
            chunk_overlap=25
        )
        
        # Verify initialization
        self.assertEqual(engine.run_id, "test-run")
        self.assertEqual(engine.config["embedding_model"], "test-model")
        self.assertEqual(engine.config["chunk_size"], 256)
        self.assertEqual(engine.config["chunk_overlap"], 25)
        
        # Verify component creation
        mock_check_ollama.assert_called_once_with(raise_error=True)
        mock_db_class.assert_called_once()
        mock_embedding_class.assert_called_once_with(model_name="test-model")
        mock_chunker_class.assert_called_once_with(chunk_size=256, overlap=25)
        
        # Verify config file was created
        self.assertTrue(config_path.exists())
    
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_add_document(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, mock_db_class):
        """Test adding a document to the RAG engine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        
        mock_db = MagicMock()
        mock_db.add_document.return_value = [1, 2, 3]  # Mock chunk IDs
        mock_db_class.return_value = mock_db
        
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embedding_class.return_value = mock_embedder
        
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_chunker_class.return_value = mock_chunker
        
        # Initialize engine
        engine = RAGEngine(
            run_id="test-run",
            base_dir=self.base_dir
        )
        
        # Test adding document
        result = engine.add_document(
            text="Test document content",
            metadata={"source_type": "test"},
            source="test_document.txt"
        )
        
        # Verify calls
        mock_chunker.chunk_text.assert_called_once_with("Test document content")
        mock_embedder.embed_texts.assert_called_once_with(["Chunk 1", "Chunk 2"])

        # Fix: Match how the actual implementation creates the metadata
        expected_metadata = {"source_type": "test", "source": "test_document.txt"}
        mock_db.add_document.assert_called_once_with(
            ["Chunk 1", "Chunk 2"],
            [[0.1, 0.2], [0.3, 0.4]],
            expected_metadata,
            source="test_document.txt"
        )
        
        # Verify result
        self.assertEqual(result, 2)  # Number of chunks
    
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_retrieve(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, mock_db_class):
        """Test retrieving content from the RAG engine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        
        mock_db = MagicMock()
        mock_db.search.return_value = [
            {"text": "Result 1", "similarity": 0.9},
            {"text": "Result 2", "similarity": 0.8}
        ]
        mock_db_class.return_value = mock_db
        
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_embedding_class.return_value = mock_embedder
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Initialize engine
        engine = RAGEngine(
            run_id="test-run",
            base_dir=self.base_dir
        )
        
        # Test retrieval
        results = engine.retrieve(
            query="Test query",
            top_k=2,
            threshold=0.5,
            source_filter="test_source"
        )
        
        # Verify calls
        mock_embedder.embed_text.assert_called_once_with("Test query")
        mock_db.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            top_k=2,
            threshold=0.5,
            source_filter="test_source"
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Result 1")
        self.assertEqual(results[1]["text"], "Result 2")
    
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_get_stats(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, mock_db_class):
        """Test getting statistics from the RAG engine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        
        mock_db = MagicMock()
        mock_db.get_stats.return_value = {
            "document_count": 5,
            "chunk_count": 20
        }
        mock_db_class.return_value = mock_db
        
        mock_embedder = MagicMock()
        mock_embedder.get_embedding_dimension.return_value = 1536
        mock_embedding_class.return_value = mock_embedder
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Initialize engine
        engine = RAGEngine(
            run_id="test-run",
            base_dir=self.base_dir
        )
        
        # Test getting stats
        stats = engine.get_stats()
        
        # Verify calls
        mock_db.get_stats.assert_called_once()
        mock_embedder.get_embedding_dimension.assert_called_once()
        
        # Verify stats
        self.assertEqual(stats["run_id"], "test-run")
        self.assertEqual(stats["document_count"], 5)
        self.assertEqual(stats["chunk_count"], 20)
        self.assertEqual(stats["embedding_dimension"], 1536)
    
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_purge(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, mock_db_class):
        """Test purging the RAG engine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        mock_embedder = MagicMock()
        mock_embedding_class.return_value = mock_embedder
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Initialize engine
        engine = RAGEngine(
            run_id="test-run",
            base_dir=self.base_dir
        )
        
        # Test purging
        engine.purge()
        
        # Verify calls
        mock_db.close.assert_called_once()
        # Verify a new DB was created
        self.assertEqual(mock_db_class.call_count, 2)
    
    @patch('osyllabi.rag.engine.Path')
    @patch('osyllabi.rag.engine.VectorDatabase')
    @patch('osyllabi.rag.engine.EmbeddingGenerator')
    @patch('osyllabi.rag.engine.TextChunker')
    @patch('osyllabi.rag.engine.check_for_ollama')
    def test_load(self, mock_check_ollama, mock_chunker_class, mock_embedding_class, 
                 mock_db_class, mock_path):
        """Test loading an existing RAG engine."""
        # Setup mocks
        mock_check_ollama.return_value = True
        
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        mock_embedder = MagicMock()
        mock_embedding_class.return_value = mock_embedder
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Mock path operations correctly
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Fix: Properly mock the file open and read operation
        config_json = {
            "run_id": "existing-run",
            "embedding_model": "existing-model",
            "chunk_size": 128,
            "chunk_overlap": 10
        }
        
        # Use patch to mock the open function
        with patch("builtins.open", mock_open(read_data=json.dumps(config_json))) as mock_file:
            # Test loading
            engine = RAGEngine.load("existing-run", base_dir=self.base_dir)
            
            # Verify configuration
            self.assertEqual(engine.run_id, "existing-run")
            self.assertEqual(engine.config["embedding_model"], "existing-model")
            self.assertEqual(engine.config["chunk_size"], 128)
            self.assertEqual(engine.config["chunk_overlap"], 10)

if __name__ == '__main__':
    unittest.main()
