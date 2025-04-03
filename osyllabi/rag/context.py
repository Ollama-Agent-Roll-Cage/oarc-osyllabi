"""
Context assembly functionality for RAG-enhanced prompts.

This module provides utilities for formatting and assembling retrieved
context to enhance prompts for curriculum generation.
"""
from typing import List, Dict, Any, Optional

from osyllabi.utils.log import log


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
        deduplication_threshold: float = 0.9
    ):
        """
        Initialize the context assembler.
        
        Args:
            max_tokens: Maximum approximate tokens to include in assembled context
            format_style: Style of formatting ("simple", "detailed", "markdown")
            deduplication_threshold: Similarity threshold for deduplication (0-1)
        """
        self.max_tokens = max_tokens
        self.format_style = format_style
        self.deduplication_threshold = deduplication_threshold
    
    def assemble_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        deduplicate: bool = True
    ) -> str:
        """
        Assemble retrieved chunks into a formatted context.
        
        Args:
            retrieved_chunks: List of chunks from RAG retrieval
            query: Optional query for contextual sorting
            deduplicate: Whether to remove similar content
            
        Returns:
            str: Assembled context ready for prompt inclusion
        """
        if not retrieved_chunks:
            return ""
            
        # Sort by similarity if available
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
            return self._assemble_simple(chunks)
        elif self.format_style == "markdown":
            return self._assemble_markdown(chunks)
        else:  # detailed is default
            return self._assemble_detailed(chunks)
    
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
            
        # Simple content-based deduplication
        unique_chunks = []
        content_hashes = set()
        
        for chunk in chunks:
            # Create a simplified representation of the content
            text = chunk.get("text", "").strip().lower()
            if not text:
                continue
            
            # Get first 100 chars as a simple signature
            signature = text[:100]
            
            # Check if we've seen very similar content
            if signature not in content_hashes:
                unique_chunks.append(chunk)
                content_hashes.add(signature)
        
        log.debug(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks
    
    def _assemble_simple(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a simple context format with just the content."""
        result = []
        token_count = 0
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            # Rough token estimation (words + punctuation)
            estimated_tokens = len(text.split()) + text.count('.') + text.count(',')
            
            if token_count + estimated_tokens > self.max_tokens:
                break
                
            result.append(text)
            token_count += estimated_tokens
        
        return "\n\n".join(result)
    
    def _assemble_detailed(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a detailed context with source information."""
        result = []
        token_count = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", "unknown source")
            similarity = chunk.get("similarity", 0)
            
            if not text:
                continue
                
            # Format the chunk with source information
            header = f"Context {i+1}: From {source} (relevance: {similarity:.2f})"
            content = f"{header}\n{text}"
            
            # Rough token estimation
            estimated_tokens = len(content.split()) + content.count('.') + content.count(',')
            
            if token_count + estimated_tokens > self.max_tokens:
                break
                
            result.append(content)
            token_count += estimated_tokens
        
        return "\n\n".join(result)
    
    def _assemble_markdown(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a markdown-formatted context."""
        result = []
        token_count = 0
        
        result.append("## Relevant Context\n")
        token_count += 3  # rough estimate for the header
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", "unknown source")
            similarity = chunk.get("similarity", 0)
            
            if not text:
                continue
                
            # Format the chunk with markdown
            header = f"### Source: {source} (relevance: {similarity:.2f})"
            content = f"{header}\n\n{text}"
            
            # Rough token estimation
            estimated_tokens = len(content.split()) + content.count('.') + content.count(',')
            
            if token_count + estimated_tokens > self.max_tokens:
                break
                
            result.append(content)
            token_count += estimated_tokens
        
        return "\n\n".join(result)
