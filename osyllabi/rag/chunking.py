"""
Text chunking for RAG capabilities in Osyllabi.

This module provides functionality for splitting text into chunks
suitable for embedding and retrieval in the RAG system.
"""
import re
from typing import List, Dict, Any, Optional

from osyllabi.utils.log import log


class TextChunker:
    """
    Split text into chunks for embedding.
    
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
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.chunk_size // 2:
            return [text] if text else []
            
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
        """
        Clean and normalize text before chunking.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace Unicode quotes with ASCII quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks by paragraphs.
        
        Args:
            text: Cleaned text to split
            
        Returns:
            List of paragraph-based chunks
        """
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
        """
        Split text into chunks by characters with overlap.
        
        Args:
            text: Cleaned text to split
            
        Returns:
            List of character-based chunks
        """
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
        
        This is a simple approximation as actual tokenization
        depends on the specific model.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Estimated token count
        """
        # Simple approximation: count words and punctuation
        return len(re.findall(r'\w+|[^\w\s]', text))
