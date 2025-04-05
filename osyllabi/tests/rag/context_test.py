"""
Unit tests for the context assembly functionality.
"""
import unittest
import warnings
from unittest.mock import patch, MagicMock

from osyllabi.rag.context import ContextAssembler

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestContextAssembler(unittest.TestCase):
    """Test cases for the ContextAssembler class."""
    
    def test_initialization(self):
        """Test initialization of ContextAssembler."""
        # Default initialization
        assembler = ContextAssembler()
        self.assertEqual(assembler.max_tokens, 4000)
        
        # Custom initialization
        assembler = ContextAssembler(max_tokens=2000, format_style="markdown")
        self.assertEqual(assembler.max_tokens, 2000)
        self.assertEqual(assembler.format_style, "markdown")
    
    def test_assemble_context_basic(self):
        """Test basic context assembly."""
        assembler = ContextAssembler()
        
        # Create sample retrieved chunks
        chunks = [
            {"text": "First chunk of information.", "similarity": 0.95},
            {"text": "Second chunk with different info.", "similarity": 0.85},
            {"text": "Third chunk with more details.", "similarity": 0.75}
        ]
        
        # Assemble context
        context = assembler.assemble_context(chunks)
        
        # Verify all chunks are included
        self.assertIn("First chunk", context)
        self.assertIn("Second chunk", context)
        self.assertIn("Third chunk", context)
        
        # Verify formatting includes similarity scores
        self.assertIn("0.95", context)
        self.assertIn("0.85", context)
        self.assertIn("0.75", context)
    
    def test_assemble_context_with_metadata(self):
        """Test context assembly with metadata."""
        assembler = ContextAssembler()
        
        # Create sample retrieved chunks with metadata
        chunks = [
            {
                "text": "Information about Python.", 
                "similarity": 0.95,
                "metadata": {"source": "python_docs.txt", "page": 10}
            },
            {
                "text": "Python is a programming language.", 
                "similarity": 0.85,
                "metadata": {"source": "intro_guide.pdf", "page": 5}
            }
        ]
        
        # Assemble context
        context = assembler.assemble_context(chunks)
        
        # Verify chunks are included with metadata
        self.assertIn("Information about Python", context)
        self.assertIn("Python is a programming language", context)
        self.assertIn("python_docs.txt", context)
        self.assertIn("intro_guide.pdf", context)
    
    def test_format_context_markdown(self):
        """Test markdown formatting of context."""
        assembler = ContextAssembler(format_style="markdown")
        
        # Create sample chunk
        chunk = {
            "text": "This is sample text.",
            "similarity": 0.9,
            "metadata": {"source": "sample.txt"}
        }
        
        # Format chunk
        formatted = assembler._format_chunk(chunk, 1)
        
        # Verify markdown formatting
        self.assertIn("## Relevant Context 1", formatted)
        self.assertIn("This is sample text.", formatted)
        self.assertIn("**Source:** sample.txt", formatted)
        self.assertIn("**Similarity:** 0.90", formatted)
    
    def test_format_context_plain(self):
        """Test plain text formatting of context."""
        assembler = ContextAssembler(format_style="plain")
        
        # Create sample chunk
        chunk = {
            "text": "This is sample text.",
            "similarity": 0.9,
            "metadata": {"source": "sample.txt"}
        }
        
        # Format chunk
        formatted = assembler._format_chunk(chunk, 1)
        
        # Verify plain formatting
        self.assertIn("RELEVANT CONTEXT 1:", formatted)
        self.assertIn("This is sample text.", formatted)
        self.assertIn("Source: sample.txt", formatted)
        self.assertIn("Similarity: 0.90", formatted)
    
    def test_deduplication(self):
        """Test deduplication of similar chunks."""
        assembler = ContextAssembler()
        
        # Create chunks with similar content
        chunks = [
            {"text": "Python is a programming language.", "similarity": 0.95},
            {"text": "Python is a high-level programming language.", "similarity": 0.9},
            {"text": "Completely different information.", "similarity": 0.8}
        ]
        
        # Assemble with deduplication
        context = assembler.assemble_context(chunks, deduplicate=True)
        
        # Verify chunks with similar content are deduplicated
        # This test assumes the deduplication logic works on similarity of content
        # Exact implementation may vary
        occurrences = context.lower().count("python is a")
        self.assertEqual(occurrences, 1)  # Only one of the similar chunks should remain
        self.assertIn("Completely different", context)
    
    def test_truncation(self):
        """Test truncation based on token limit."""
        # Create assembler with small token limit
        assembler = ContextAssembler(max_tokens=50)
        
        # Create many chunks that would exceed the token limit
        chunks = [
            {"text": f"This is chunk {i} with some content.", "similarity": 0.9 - (i * 0.1)}
            for i in range(10)
        ]
        
        # Assemble context
        context = assembler.assemble_context(chunks)
        
        # Verify context has been truncated (not all chunks included)
        # Assuming approx 6-7 tokens per chunk, only ~7 chunks should fit
        for i in range(5):  # Check first few chunks are included
            self.assertIn(f"chunk {i}", context)
        
        # The last chunks should be omitted due to token limit
        self.assertNotIn("chunk 9", context)
    
    def test_empty_chunks(self):
        """Test assembly with empty chunk list."""
        assembler = ContextAssembler()
        
        # Assemble with empty list
        context = assembler.assemble_context([])
        
        # Verify we get a reasonable empty context message
        self.assertIn("No relevant context", context)
    
    def test_intro_and_instructions(self):
        """Test introduction and instructions in context."""
        assembler = ContextAssembler()
        
        chunks = [{"text": "Sample content.", "similarity": 0.9}]
        
        # Assemble context
        context = assembler.assemble_context(chunks)
        
        # Verify intro and instructions are included
        self.assertIn("following relevant information", context.lower())
        self.assertIn("improve your response", context.lower())

    def test_disclaimer_included(self):
        assembler = ContextAssembler()
        chunks = [{"text": "Sample info.", "similarity": 0.9}]
        result = assembler.assemble_context(chunks)
        self.assertIn("Disclaimer: This context", result)
        self.assertIn("Sample info.", result)


class TestContext(unittest.TestCase):
    """Test suite for context-related functionalities."""

    def test_context_creation(self):
        """Example test that checks basic instantiation or setup."""
        self.assertTrue(True, "Context creation test not implemented yet.")

    def test_context_behavior(self):
        """Example test that verifies context behavior or methods."""
        self.assertTrue(True, "Context behavior test not implemented yet.")


if __name__ == '__main__':
    unittest.main()
