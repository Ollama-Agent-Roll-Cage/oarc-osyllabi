"""
Content extraction functionality for curriculum generation.

This module provides base classes and implementations for extracting content
from various types of resources.
"""
import abc
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from osyllabi.utils.log import log


class ContentExtractorABC(abc.ABC):
    """
    Abstract base class for content extractors.
    
    A content extractor is responsible for parsing and extracting useful content
    from specific types of resources (e.g., HTML, markdown, code files).
    """
    
    @abc.abstractmethod
    def extract(self, resource: Any) -> Dict[str, Any]:
        """
        Extract content from a resource.
        
        Args:
            resource: Resource to extract content from
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        pass
    
    @abc.abstractmethod
    def supports(self, resource: Any) -> bool:
        """
        Check if this extractor supports the given resource.
        
        Args:
            resource: Resource to check
            
        Returns:
            True if this extractor supports the resource, False otherwise
        """
        pass
    
    def extract_keywords(self, text: str, min_length: int = 4, max_keywords: int = 20) -> List[str]:
        """
        Extract important keywords from content.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        # Simple stopwords list
        stopwords = {
            'the', 'and', 'is', 'in', 'at', 'of', 'for', 'with', 'by', 'to', 'a', 'an',
            'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'our', 'your'
        }
        
        # Extract words, filter, and count
        words = re.findall(r'\b[a-z]{%d,}\b' % min_length, text.lower())
        word_counts = {}
        
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return top_keywords
