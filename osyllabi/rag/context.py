"""
Context assembly functionality for RAG-enhanced prompts.

This module provides utilities for formatting and assembling retrieved
context to enhance prompts for curriculum generation.
"""
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from osyllabi.utils.log import log


@dataclass
class FormatSettings:
    """Settings for context formatting."""
    include_sources: bool = True
    include_relevance: bool = True
    include_separators: bool = True
    include_headers: bool = True
    highlight_keywords: bool = True
    max_length_per_chunk: int = 1000
    keywords: Set[str] = field(default_factory=set)


class ContextAssembler:
    """
    Assembles retrieved context for prompt enhancement.
    
    This class provides methods to format and organize retrieved chunks
    into a coherent context that can be added to prompts.
    """
    
    def __init__(
        self,
        max_tokens: int = 1500,
        format_style: str = "detailed",
        deduplication_threshold: float = 0.9,
        preserve_order: bool = False
    ):
        """
        Initialize the context assembler.
        
        Args:
            max_tokens: Maximum approximate tokens to include in assembled context
            format_style: Style of formatting ("simple", "detailed", "markdown")
            deduplication_threshold: Similarity threshold for deduplication (0-1)
            preserve_order: Whether to preserve chunk order from retrieval
        """
        self.max_tokens = max_tokens
        self.format_style = format_style
        self.deduplication_threshold = deduplication_threshold
        self.preserve_order = preserve_order
        self._max_context_chars = max_tokens * 4  # Rough approximation
        
    def assemble_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        deduplicate: bool = True,
        settings: Optional[FormatSettings] = None
    ) -> str:
        """
        Assemble retrieved chunks into a formatted context.
        
        Args:
            retrieved_chunks: List of chunks from RAG retrieval
            query: Optional query for contextual formatting
            deduplicate: Whether to remove similar content
            settings: Optional format settings to override defaults
            
        Returns:
            str: Assembled context ready for prompt inclusion
        """
        if not retrieved_chunks:
            return ""
            
        # Use default settings if none provided
        settings = settings or FormatSettings()
        
        # Extract keywords from query for highlighting
        if query and settings.highlight_keywords:
            settings.keywords = self._extract_keywords(query)
            
        # Sort by similarity if available and not preserving order
        chunks = retrieved_chunks
        if not self.preserve_order:
            chunks = sorted(
                retrieved_chunks, 
                key=lambda x: x.get("similarity", 0), 
                reverse=True
            )
        
        # Deduplicate if requested
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
        
        # Assemble based on format style
        if self.format_style == "simple":
            return self._assemble_simple(chunks, settings)
        elif self.format_style == "markdown":
            return self._assemble_markdown(chunks, settings)
        else:  # detailed is default
            return self._assemble_detailed(chunks, settings)
    
    def _extract_keywords(self, text: str, min_length: int = 4) -> Set[str]:
        """Extract important keywords from the query."""
        # Simple implementation - extract words longer than min_length
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) >= min_length and w not in self._get_stopwords()}
    
    def _get_stopwords(self) -> Set[str]:
        """Get common stopwords to ignore for highlighting."""
        return {
            'the', 'and', 'that', 'have', 'for', 'not', 'this', 'with', 'you', 'which',
            'from', 'they', 'will', 'would', 'there', 'their', 'what', 'about', 'when',
            'how', 'some', 'can', 'like', 'into', 'than', 'more', 'your', 'has'
        }
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate or very similar chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List[Dict[str, Any]]: Deduplicated chunks
        """
        if not chunks:
            return []
            
        # Improved content-based deduplication
        unique_chunks = []
        content_signatures = set()
        
        for chunk in chunks:
            # Get text and clean it for comparison
            text = chunk.get("text", "").strip().lower()
            if not text:
                continue
                
            # Create a content signature based on first N chars + middle N chars
            n = 80  # Signature size
            if len(text) <= n * 2:
                signature = text
            else:
                mid_start = max(0, (len(text) // 2) - (n // 2))
                signature = text[:n] + text[mid_start:mid_start + n]
                
            # Remove punctuation and extra whitespace for better matching
            signature = re.sub(r'[^\w\s]', '', signature)
            signature = re.sub(r'\s+', ' ', signature).strip()
            
            # Check if we've seen very similar content
            if signature not in content_signatures:
                unique_chunks.append(chunk)
                content_signatures.add(signature)
        
        chunk_reduction = len(chunks) - len(unique_chunks)
        if chunk_reduction > 0:
            log.debug(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks (-{chunk_reduction})")
            
        return unique_chunks
    
    def _assemble_simple(self, chunks: List[Dict[str, Any]], settings: FormatSettings) -> str:
        """Create a simple context format with just the content."""
        result = []
        total_length = 0
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            # Limit chunk length if needed
            if len(text) > settings.max_length_per_chunk:
                text = text[:settings.max_length_per_chunk] + "..."
                
            # Add to result if within max context size
            if total_length + len(text) > self._max_context_chars:
                break
                
            result.append(text)
            total_length += len(text)
        
        # Join with double newline for separation
        return "\n\n".join(result)
    
    def _assemble_detailed(self, chunks: List[Dict[str, Any]], settings: FormatSettings) -> str:
        """Create a detailed context with source information."""
        result = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", chunk.get("metadata", {}).get("source", "unknown source"))
            similarity = chunk.get("similarity", 0)
            
            if not text:
                continue
                
            # Limit chunk length if needed
            if len(text) > settings.max_length_per_chunk:
                text = text[:settings.max_length_per_chunk] + "..."
                
            # Format the chunk with source information
            header = f"Context {i+1}"
            if settings.include_sources:
                header += f" from {source}"
            if settings.include_relevance:
                header += f" (relevance: {similarity:.2f})"
                
            content = f"{header}\n{text}" if settings.include_headers else text
            
            # Check if adding this would exceed max context length
            if total_length + len(content) > self._max_context_chars:
                break
                
            # Add separator if requested
            if settings.include_separators and result:
                result.append("-" * 40)
                
            result.append(content)
            total_length += len(content)
        
        return "\n\n".join(result)
    
    def _assemble_markdown(self, chunks: List[Dict[str, Any]], settings: FormatSettings) -> str:
        """Create a markdown-formatted context."""
        result = ["## Relevant Context\n"]
        total_length = 4  # Starting with the header length
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", chunk.get("metadata", {}).get("source", "unknown source"))
            similarity = chunk.get("similarity", 0)
            
            if not text:
                continue
                
            # Highlight keywords if requested
            if settings.highlight_keywords and settings.keywords:
                for keyword in settings.keywords:
                    # Use regex to match whole word only
                    text = re.sub(
                        fr'\b({re.escape(keyword)})\b',
                        r'**\1**',
                        text,
                        flags=re.IGNORECASE
                    )
                
            # Limit chunk length if needed
            if len(text) > settings.max_length_per_chunk:
                text = text[:settings.max_length_per_chunk] + "..."
                
            # Format header with source and relevance
            header = f"### Source {i+1}"
            if settings.include_sources:
                header += f": {source}"
            if settings.include_relevance:
                header += f" (relevance: {similarity:.2f})"
                
            content = f"{header}\n\n{text}" if settings.include_headers else text
            
            # Check if adding this would exceed max context length
            if total_length + len(content) > self._max_context_chars:
                break
                
            result.append(content)
            total_length += len(content)
        
        return "\n\n".join(result)
