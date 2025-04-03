"""
Resource collection and management for curriculum generation.

This module provides the high-level resource management functionality
for collecting and processing resources from various sources.
"""
import importlib.util
from typing import List, Dict, Any, Optional
from pathlib import Path

from osyllabi.utils.log import log
from osyllabi.generator.resource.manager import ResourceManager


class ResourceCollectionManager:
    """
    High-level manager for resource collection in curriculum generation.
    
    This class provides a convenient entry point to the resource collection
    functionality, delegating to the modular resource management system.
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 5,
        max_file_size_mb: float = 10.0,
        max_content_length: int = 10000
    ):
        """
        Initialize the resource collection manager.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent web requests
            max_file_size_mb: Maximum file size to process in MB
            max_content_length: Maximum content length to extract
        """
        # Create the underlying resource manager
        self.manager = ResourceManager(
            max_concurrent_requests=max_concurrent_requests,
            max_file_size_mb=max_file_size_mb,
            max_content_length=max_content_length
        )
        
        # Check for optional dependencies for enhanced functionality
        self.bs4_available = importlib.util.find_spec("bs4") is not None
        self.markdown_available = importlib.util.find_spec("markdown") is not None
        self.pandas_available = importlib.util.find_spec("pandas") is not None
        self.pymupdf_available = importlib.util.find_spec("fitz") is not None
        self.docx_available = importlib.util.find_spec("docx") is not None
        
        if not self.bs4_available:
            log.warning("BeautifulSoup4 not available - HTML parsing will be limited")
        if not self.markdown_available:
            log.warning("Markdown library not available - Markdown processing will be basic")
        if not self.pandas_available:
            log.warning("Pandas not available - Data file processing will be limited")
        if not self.pymupdf_available:
            log.warning("PyMuPDF not available - PDF processing will not be available")
        if not self.docx_available:
            log.warning("python-docx not available - DOCX processing will not be available")
        
    def collect_resources(
        self,
        urls: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        extract_keywords: bool = True,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Collect resources from URLs and file paths.
        
        Args:
            urls: List of URLs to collect from
            paths: List of file/directory paths to collect from
            extract_keywords: Whether to extract keywords from content
            deduplicate: Whether to deduplicate similar content
            
        Returns:
            Dictionary of collected resources with metadata
        """
        log.info(f"Collecting resources from {len(urls or [])} URLs and {len(paths or [])} paths")
        
        # Process sources using the resource manager
        resources = self.manager.process_sources(
            urls=urls,
            paths=paths,
            extract_keywords=extract_keywords,
            deduplicate=deduplicate
        )
        
        log.info(f"Resource collection completed: " +
                f"{len(resources.get('urls', {}))} web sources, " +
                f"{len(resources.get('files', {}))} file sources")
                
        return resources
        
    def extract_context_for_prompt(
        self,
        resources: Dict[str, Any],
        topic: str,
        max_items: int = 5
    ) -> str:
        """
        Extract a formatted context string for use in prompts.
        
        Args:
            resources: Collected resources
            topic: Main topic for relevance
            max_items: Maximum number of resources to include
            
        Returns:
            Formatted context string for prompt
        """
        # Use enhanced context extraction if markdown is available
        if self.markdown_available:
            try:
                import markdown
                
                # Extract most relevant content
                context_parts = []
                
                # Include most relevant URL content
                urls = list(resources.get("urls", {}).items())
                if urls:
                    context_parts.append("## Web Resources")
                    for i, (url, data) in enumerate(urls[:max_items]):
                        title = data.get("title", url)
                        content = data.get("content", "")
                        if len(content) > 500:
                            content = content[:500] + "..."
                        context_parts.append(f"### {title}\n{content}")
                
                # Include most relevant file content
                files = list(resources.get("files", {}).items())
                if files:
                    context_parts.append("## Local Resources")
                    for i, (path, data) in enumerate(files[:max_items]):
                        title = data.get("title", Path(path).name)
                        content = data.get("content", "")
                        if len(content) > 500:
                            content = content[:500] + "..."
                        context_parts.append(f"### {title}\n{content}")
                
                # Include keywords as markdown
                keywords = resources.get("metadata", {}).get("keywords", [])
                if keywords:
                    keyword_links = [f"[{kw}](#{kw.replace(' ', '-')})" for kw in keywords[:20]]
                    context_parts.append(f"## Keywords\n{', '.join(keyword_links)}")
                    
                # Return markdown context
                return "\n\n".join(context_parts)
            except Exception as e:
                log.warning(f"Error using markdown for context extraction: {e}")
        
        # Fall back to basic context extraction
        return self.manager.extract_context(resources, topic, max_items)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get resource collection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.manager.get_stats()
