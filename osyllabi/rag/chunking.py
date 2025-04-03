"""
Text chunking for RAG capabilities in Osyllabi.

This module provides functionality for splitting text into chunks
suitable for embedding and retrieval in the RAG system, using
LangChain's text splitters for optimal results.
"""
from typing import List, Dict, Any, Optional, Union
import importlib.util

from osyllabi.utils.log import log


class TextChunker:
    """
    Split text into chunks for embedding using LangChain.
    
    This class provides functionality to divide documents into
    smaller, overlapping chunks suitable for vector embedding.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            overlap: Overlap between consecutive chunks (characters)
        """
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2)
        self.text_splitter = None
        self._initialize_splitter()
        
    def _initialize_splitter(self) -> None:
        """Initialize the LangChain text splitter or fallback to custom splitting."""
        # Check if langchain is available
        langchain_available = importlib.util.find_spec("langchain") is not None
        text_splitters_available = importlib.util.find_spec("langchain.text_splitter") is not None
        
        if langchain_available and text_splitters_available:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                log.info("Using LangChain's RecursiveCharacterTextSplitter for text chunking")
            except Exception as e:
                log.error(f"Failed to initialize LangChain text splitter: {e}")
                self.text_splitter = None
        else:
            log.warning("LangChain not available. Using fallback text chunking.")
            self.text_splitter = None
            
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
            
        # Use LangChain if available
        if self.text_splitter:
            try:
                return self.text_splitter.split_text(text)
            except Exception as e:
                log.error(f"LangChain text splitting failed: {e}. Falling back to custom chunking.")
                return self._fallback_chunk_text(text)
        else:
            return self._fallback_chunk_text(text)
    
    def _fallback_chunk_text(self, text: str) -> List[str]:
        """
        Fallback chunking when LangChain is not available.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) < self.chunk_size // 2:
            return [text]
            
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Try to chunk by paragraphs first
        chunks = self._chunk_by_paragraphs(cleaned_text)
        
        # If paragraph chunking yields too few or too large chunks, 
        # fall back to character-based chunking
        if not chunks or any(len(chunk) > self.chunk_size * 2 for chunk in chunks):
            chunks = self._chunk_by_characters(cleaned_text)
            
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        import re
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace Unicode quotes with ASCII quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('’', "'").replace('’', "'")
        
        return text
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text into chunks by paragraphs."""
        import re
        # Split by paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return []
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the chunk size,
            # finish the current chunk
            if current_length + len(paragraph) > self.chunk_size and current_length > 0:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk, potentially overlapping with the end of the previous chunk
                overlap_paragraphs = []
                overlap_length = 0
                
                # Add paragraphs from the end until we reach the desired overlap
                for p in reversed(current_chunk):
                    if overlap_length + len(p) <= self.overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_length += len(p) + 1  # +1 for the space
                    else:
                        break
                        
                current_chunk = overlap_paragraphs
                current_length = overlap_length
            
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1  # +1 for the space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """Split text into chunks by characters with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Adjust end to avoid splitting words
            if end < len(text):
                # Look for a space to break at
                while end > start and text[end] != ' ':
                    end -= 1
                    
                # If no space found, just break at chunk_size
                if end == start:
                    end = start + self.chunk_size
            else:
                end = len(text)
                
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move the start position with overlap
            start = end - self.overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Estimated token count
        """
        # Try to use tiktoken if available for accurate counting
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")  # Default for many models
            return len(encoding.encode(text))
        except ImportError:
            # Simple approximation: count words and punctuation
            import re
            return len(re.findall(r'\w+|[^\w\s]', text))
