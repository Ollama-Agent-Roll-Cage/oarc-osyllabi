"""
Tests for LlamaIndex integration.
"""
import unittest
import warnings
import os
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
from pathlib import Path

# Filter out NumPy deprecation warning
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

from osyllabi.rag.llama import (
    set_test_mode, LlamaDocumentLoader, TextProcessor,
    setup_llama_index, is_llama_index_available
)
from osyllabi.rag.engine import RAGEngine
from llama_index.core.schema import TextNode, Document as LlamaDocument
from llama_index.readers.file.docs import PDFReader, DocxReader  # Import the readers

# Enable test mode before running tests
set_test_mode(True)

class TestLlamaIndexIntegration(unittest.TestCase):
    """Test cases for LlamaIndex integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Ensure test environment is set up
        os.environ["IS_TESTING"] = "True"
        
        self.mock_db = MagicMock()
        self.mock_db.data = pd.DataFrame({
            'chunk_id': [1, 2],
            'doc_id': [1, 1],
            'text': ['Sample text 1', 'Sample text 2'],
            'source': ['test.txt', 'test.txt'],
            'metadata': ['{"type":"test"}', '{"type":"test"}'],
            'embedding': [[0.1, 0.2], [0.3, 0.4]]
        })
        
        # Mock Path.exists() for document tests
        self.patcher = patch('pathlib.Path.exists')
        self.mock_exists = self.patcher.start()
        self.mock_exists.return_value = True
        
    def tearDown(self):
        """Clean up test environment.""" 
        self.patcher.stop()
    
    @patch('osyllabi.rag.llama.LlamaDocumentLoader._convert_to_osyllabi_format')
    @patch('osyllabi.rag.llama.LlamaDocumentLoader._call_reader_method')
    def test_document_loader_pdf(self, mock_call_method, mock_convert):
        """Test PDF document loading."""
        # Setup mocks
        mock_docs = [MagicMock()]
        mock_call_method.return_value = mock_docs
        
        # Setup mock conversion
        mock_converted_docs = [{"content": "Sample PDF content", "source": "test.pdf"}]
        mock_convert.return_value = mock_converted_docs
        
        loader = LlamaDocumentLoader()
        result = loader.load(Path("test.pdf"))
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Sample PDF content")
        self.assertEqual(result[0]["source"], "test.pdf")
        mock_call_method.assert_called_once()
    
    @patch('osyllabi.rag.llama.LlamaDocumentLoader._convert_to_osyllabi_format')
    @patch('osyllabi.rag.llama.LlamaDocumentLoader._call_reader_method')
    def test_document_loader_docx(self, mock_call_method, mock_convert):
        """Test DOCX document loading."""
        # Setup mocks
        mock_docs = [MagicMock()]
        mock_call_method.return_value = mock_docs
        
        # Setup mock conversion
        mock_converted_docs = [{"content": "Sample DOCX content", "source": "test.docx"}]
        mock_convert.return_value = mock_converted_docs
        
        loader = LlamaDocumentLoader()
        result = loader.load(Path("test.docx"))
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Sample DOCX content")
        self.assertEqual(result[0]["source"], "test.docx")
        mock_call_method.assert_called_once()
    
    def test_document_loader_unsupported_format(self):
        """Test handling of unsupported file formats."""
        loader = LlamaDocumentLoader()
        with self.assertRaises(ValueError):
            loader.load(Path("test.txt"))
    
    def test_setup_llama_index(self):
        """Test LlamaIndex setup configuration."""
        with patch('osyllabi.rag.llama.Settings') as MockSettings:
            # Call the function
            result = setup_llama_index()
            
            # Verify result
            self.assertTrue(result)
            
            # Verify Settings assignments
            self.assertEqual(MockSettings.chunk_size, 512)
            self.assertEqual(MockSettings.chunk_overlap, 50)
            self.assertIsNone(MockSettings.embed_model)
    
    def test_is_llama_index_available(self):
        """Test LlamaIndex availability checking."""
        # Test when LlamaIndex is available
        with patch('importlib.import_module', return_value=MagicMock()):
            self.assertTrue(is_llama_index_available())
            
        # Test when LlamaIndex is not available
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            self.assertFalse(is_llama_index_available())
    
    def test_create_nodes_from_documents(self):
        """Test creating LlamaIndex nodes from document dictionaries."""
        loader = LlamaDocumentLoader()
        
        # Create sample documents
        documents = [
            {
                "content": "Document 1 content",
                "metadata": {"source": "doc1.pdf", "page": 1},
                "source": "doc1.pdf",
                "content_type": "text"
            },
            {
                "content": "Document 2 content",
                "metadata": {"source": "doc2.pdf", "page": 2},
                "source": "doc2.pdf", 
                "content_type": "text"
            }
        ]
        
        # Convert to nodes
        nodes = loader.create_nodes_from_documents(documents)
        
        # Verify conversion
        self.assertEqual(len(nodes), 2)
        self.assertIsInstance(nodes[0], TextNode)
        self.assertEqual(nodes[0].text, "Document 1 content")
        self.assertEqual(nodes[0].metadata["source"], "doc1.pdf")
        self.assertEqual(nodes[0].metadata["page"], 1)
        self.assertEqual(nodes[1].text, "Document 2 content")
    
    def test_text_processor_split_text(self):
        """Test text splitting functionality."""
        with patch('llama_index.core.node_parser.SimpleNodeParser.get_nodes_from_documents') as mock_get_nodes:
            # Mock node parser return values
            node1 = MagicMock()
            node1.get_content.return_value = "Chunk 1"
            node2 = MagicMock()
            node2.get_content.return_value = "Chunk 2"
            mock_get_nodes.return_value = [node1, node2]
            
            # Create processor and split text
            processor = TextProcessor(chunk_size=256, chunk_overlap=25)
            chunks = processor.split_text("This is a long text that should be split into chunks.")
            
            # Verify results
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0], "Chunk 1")
            self.assertEqual(chunks[1], "Chunk 2")
            mock_get_nodes.assert_called_once()
    
    @patch('llama_index.core.ingestion.IngestionPipeline')
    def test_create_ingestion_pipeline(self, mock_pipeline_class):
        """Test creation of ingestion pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Patch the specific module where IngestionPipeline is imported in our code
        with patch('osyllabi.rag.llama.IngestionPipeline', mock_pipeline_class):
            processor = TextProcessor()
            pipeline = processor.create_ingestion_pipeline()
            
            # Verify pipeline creation
            self.assertEqual(pipeline, mock_pipeline)
            mock_pipeline_class.assert_called_once()
    
    def test_document_loader_integration(self):
        """Test end-to-end document loading flow."""
        # Create mock documents
        mock_docs = [
            LlamaDocument(text="Document 1", metadata={"source": "doc1.pdf"}),
            LlamaDocument(text="Document 2", metadata={"source": "doc1.pdf"})
        ]
        
        # Setup mocks - patch the internal _call_reader_method
        with patch.object(LlamaDocumentLoader, '_call_reader_method', return_value=mock_docs):
            with patch.object(LlamaDocumentLoader, '_convert_to_osyllabi_format') as mock_convert:
                # Setup mock conversion result
                mock_convert.return_value = [
                    {"content": "Document 1", "source": "doc1.pdf"},
                    {"content": "Document 2", "source": "doc1.pdf"}
                ]
                
                # Load document
                loader = LlamaDocumentLoader()
                result = loader.load("test.pdf")
                
                # Verify conversion
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0]["content"], "Document 1")
                self.assertEqual(result[0]["source"], "doc1.pdf")
                self.assertEqual(result[1]["content"], "Document 2")
    
    # Add a test for the new _call_reader_method
    def test_call_reader_method(self):
        """Test the fallback mechanism for calling reader methods."""
        # Create a mock reader with a specific method
        mock_reader = MagicMock()
        mock_reader.parse_file = MagicMock(return_value=["document1"])
        
        # Test with the reader
        loader = LlamaDocumentLoader()
        result = loader._call_reader_method(mock_reader, "test_file.pdf")
        
        # Verify correct method was called
        self.assertEqual(result, ["document1"])
        mock_reader.parse_file.assert_called_once_with("test_file.pdf")
        
        # Test fallback behavior
        mock_reader2 = MagicMock()
        # No parse_file, but has load_data
        mock_reader2.parse_file = None
        mock_reader2.load_data = MagicMock(return_value=["document2"])
        
        result2 = loader._call_reader_method(mock_reader2, "test_file.pdf")
        self.assertEqual(result2, ["document2"])
        mock_reader2.load_data.assert_called_once_with("test_file.pdf")
        
        # Test error case when no methods are available
        mock_reader3 = MagicMock()
        # Remove all methods
        mock_reader3.parse_file = None
        mock_reader3.load_data = None
        mock_reader3.load = None
        
        with self.assertRaises(AttributeError):
            loader._call_reader_method(mock_reader3, "test_file.pdf")


if __name__ == '__main__':
    unittest.main()
