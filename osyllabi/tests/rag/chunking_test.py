"""
Unit tests for the text chunking functionality.
"""
import unittest
import warnings
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from osyllabi.rag.chunking import TextChunker

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestTextChunker(unittest.TestCase):
    """Test cases for the TextChunker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of TextChunker."""
        # Default initialization
        chunker = TextChunker()
        self.assertEqual(chunker.chunk_size, 512)
        self.assertEqual(chunker.overlap, 50)
        
        # Custom initialization
        chunker = TextChunker(chunk_size=256, overlap=20)
        self.assertEqual(chunker.chunk_size, 256)
        self.assertEqual(chunker.overlap, 20)
    
    def test_chunk_text_simple(self):
        """Test chunking simple text."""
        chunker = TextChunker(chunk_size=100, overlap=0)
        
        # Simple text shorter than chunk size
        text = "This is a simple text."
        chunks = chunker.chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
        
        # Text slightly longer than chunk size
        long_text = "A" * 120
        chunks = chunker.chunk_text(long_text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 100)
        self.assertEqual(len(chunks[1]), 20)
    
    def test_chunk_text_with_overlap(self):
        """Test chunking with overlap."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        # Create text that should split into multiple chunks with overlap
        text = "A" * 250
        chunks = chunker.chunk_text(text)
        
        # Should create 3 chunks: 0-100, 80-180, 160-250
        self.assertEqual(len(chunks), 3)
        
        # Check chunk sizes
        self.assertEqual(len(chunks[0]), 100)
        self.assertEqual(len(chunks[1]), 100)
        self.assertEqual(len(chunks[2]), 90)  # Last chunk might be smaller
        
        # Check overlap: end of first chunk should match start of second chunk
        self.assertEqual(chunks[0][-20:], chunks[1][:20])
        self.assertEqual(chunks[1][-20:], chunks[2][:20])
    
    def test_chunk_text_with_paragraphs(self):
        """Test chunking text with paragraphs."""
        chunker = TextChunker(chunk_size=100, overlap=0)
        
        # Create text with multiple paragraphs
        paragraphs = [
            "This is paragraph one.",
            "This is paragraph two.",
            "This is paragraph three.",
            "This is paragraph four which is a bit longer than the others.",
            "This is paragraph five."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = chunker.chunk_text(text)
        
        # Verify that we got the right number of chunks
        self.assertEqual(len(chunks), 2)  # Based on text length and chunk size
        
        # Verify that the chunker tries to respect paragraph boundaries
        # First chunk should contain complete paragraphs
        self.assertIn("paragraph one", chunks[0])
        self.assertIn("paragraph two", chunks[0])
        
        # Last chunk should contain the remaining paragraphs
        self.assertIn("paragraph four", chunks[1])
        self.assertIn("paragraph five", chunks[1])
    
    @patch('osyllabi.rag.chunking.CharacterTextSplitter')
    def test_recursive_character_chunking(self, mock_splitter_class):
        """Test recursive character text chunking."""
        # Setup mock for CharacterTextSplitter
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
        mock_splitter_class.return_value = mock_splitter
        
        # Create chunker
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        # Test chunking
        text = "This is sample text for testing recursive character chunking."
        chunks = chunker.chunk_text(text)
        
        # Verify the text splitter was used correctly
        mock_splitter_class.assert_called_once_with(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            separators=['\n\n', '\n', '. ', ', ', ' ', '']
        )
        mock_splitter.split_text.assert_called_once_with(text)
        
        # Verify chunks
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Chunk 1")
        self.assertEqual(chunks[1], "Chunk 2")
        self.assertEqual(chunks[2], "Chunk 3")
    
    def test_empty_text(self):
        """Test handling empty text."""
        chunker = TextChunker()
        
        chunks = chunker.chunk_text("")
        self.assertEqual(len(chunks), 0)
        
        chunks = chunker.chunk_text(None)
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_document_with_sections(self):
        """Test chunking a document with sections."""
        chunker = TextChunker(chunk_size=150, overlap=0)
        
        # Create a document with marked sections
        document = """
        # Introduction
        This is the introduction section of the document.
        
        # Methodology
        This section describes the methodology used.
        It contains multiple paragraphs of information.
        
        This is the second paragraph in the methodology section.
        
        # Results
        Here are the results of the analysis.
        """
        
        chunks = chunker.chunk_text(document)
        
        # Verify we got appropriate chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that section headers are preserved in chunks
        header_chunks = [chunk for chunk in chunks if "# " in chunk]
        self.assertGreaterEqual(len(header_chunks), 1)
    
    def test_get_optimal_chunk_size(self):
        """Test getting optimal chunk size based on text length."""
        chunker = TextChunker()
        
        # Short text
        optimal_size = chunker._get_optimal_chunk_size("Short text", max_chunks=5)
        self.assertLessEqual(optimal_size, chunker.chunk_size)
        
        # Medium text
        medium_text = "A" * 2000
        optimal_size = chunker._get_optimal_chunk_size(medium_text, max_chunks=5)
        self.assertGreaterEqual(optimal_size, len(medium_text) // 5)
        
        # Long text
        long_text = "A" * 10000
        optimal_size = chunker._get_optimal_chunk_size(long_text, max_chunks=10)
        self.assertGreaterEqual(optimal_size, len(long_text) // 10)


if __name__ == '__main__':
    unittest.main()
