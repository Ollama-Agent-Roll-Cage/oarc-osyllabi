"""
Text chunking for RAG capabilities in Osyllabi.

This module provides functionality for splitting text into chunks
suitable for embedding and retrieval in the RAG system.
"""
import re
import importlib.util
from typing import List, Dict, Any
from dataclasses import dataclass

from osyllabi.utils.log import log

from langchain.text_splitter import RecursiveCharacterTextSplitter
CharacterTextSplitter = RecursiveCharacterTextSplitter

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    index: int  # Position in the original document
    source_text: str  # Source of the chunk
    char_start: int  # Starting character index in original text
    char_end: int  # Ending character index in original text
    is_paragraph_boundary: bool = False  # Whether this chunk starts at a paragraph boundary


class TextChunker:
    """
    Split text into chunks for embedding and retrieval.
    
    This class provides functionality to divide documents into
    smaller, overlapping chunks suitable for vector embedding.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        respect_paragraphs: bool = True
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            overlap: Overlap between consecutive chunks (characters)
            respect_paragraphs: Try to preserve paragraph boundaries
            
        Raises:
            ImportError: If LangChain is not available
        """
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2)
        self.respect_paragraphs = respect_paragraphs
        self.text_splitter = None
        
        # Initialize LangChain text splitter - required
        self._initialize_splitter()
        
    def _initialize_splitter(self) -> None:
        """
        Initialize the LangChain text splitter.
        
        Raises:
            ImportError: If LangChain or its text_splitter module is not available
        """
        # Check if langchain is available
        langchain_available = importlib.util.find_spec("langchain") is not None
        text_splitters_available = importlib.util.find_spec("langchain.text_splitter") is not None
        
        if not langchain_available or not text_splitters_available:
            raise ImportError("LangChain is required for text chunking. Please install with 'pip install langchain'")
            
        try:
            # Use the already imported global CharacterTextSplitter instead of re-importing
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]
            
            self.text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                separators=separators
            )
            log.info("Using LangChain's RecursiveCharacterTextSplitter for text chunking")
        except Exception as e:
            log.error(f"Failed to initialize LangChain text splitter: {e}")
            raise ImportError(f"Failed to initialize LangChain text splitter: {e}")
            
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
            
        Raises:
            RuntimeError: If text chunking fails
        """
        if not text or not text.strip():
            return []
            
        # Clean and normalize the text
        clean_text = self._clean_text(text)
        
        try:
            chunks = self.text_splitter.split_text(clean_text)
            log.debug(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            log.error(f"LangChain text splitting failed: {e}")
            raise RuntimeError(f"Text chunking failed: {e}")
    
    def chunk_text_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of dictionaries with chunk text and metadata
            
        Raises:
            RuntimeError: If text chunking fails
        """
        if not text or not text.strip():
            return []
            
        # Clean and normalize the text
        clean_text = self._clean_text(text)
        
        try:
            # Use LangChain for basic chunking
            chunks = self.text_splitter.split_text(clean_text)
            
            # Add metadata to each chunk
            result = []
            for i, chunk in enumerate(chunks):
                # Find position in original text (approximate)
                start_pos = clean_text.find(chunk[:min(50, len(chunk))])
                if start_pos == -1:
                    start_pos = 0
                end_pos = start_pos + len(chunk)
                
                # Check if this is a paragraph boundary
                is_paragraph = False
                if start_pos > 0 and clean_text[start_pos-1:start_pos+1].count('\n') >= 1:
                    is_paragraph = True
                
                result.append({
                    "text": chunk,
                    "metadata": ChunkMetadata(
                        index=i,
                        source_text=text[:50] + "..." if len(text) > 50 else text,
                        char_start=start_pos,
                        char_end=end_pos,
                        is_paragraph_boundary=is_paragraph
                    )
                })
                
            log.debug(f"Split text into {len(result)} chunks with metadata")
            return result
            
        except Exception as e:
            log.error(f"Text chunking with metadata failed: {e}")
            raise RuntimeError(f"Text chunking failed: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        # Replace multiple newlines with just two (for paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Normalize whitespace (no consecutive spaces)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Estimated token count
            
        Raises:
            ImportError: If tiktoken is not available
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")  # Default for many models
            return len(encoding.encode(text))
        except ImportError:
            raise ImportError("tiktoken is required for token counting. Please install with 'pip install tiktoken'")
    
    def _get_optimal_chunk_size(self, text: str, max_chunks: int = 5) -> int:
        length = len(text)
        if length == 0:
            return 0
        chunk_size = length // max_chunks
        return chunk_size if chunk_size > 0 else 1
