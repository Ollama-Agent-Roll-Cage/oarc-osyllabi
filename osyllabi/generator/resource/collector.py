"""
Resource collection functionality for curriculum generation.

This module provides base classes and implementations for resource collection
from various sources.
"""
import abc
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from osyllabi.utils.log import log
from osyllabi.utils.paths import find_source_files
from osyllabi.generator.resource.extractor import ContentExtractorABC


class CollectorABC(abc.ABC):
    """
    Abstract base class for resource collectors.
    
    A resource collector is responsible for gathering content from
    specific types of sources (e.g., web, files, APIs).
    """
    
    @abc.abstractmethod
    def collect(self, sources: List[str]) -> Dict[str, Any]:
        """
        Collect resources from the provided sources.
        
        Args:
            sources: List of source identifiers appropriate for this collector
            
        Returns:
            Dictionary of collected resources
        """
        pass
    
    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary of collection statistics
        """
        pass


class ResourceCollector:
    """
    Coordinates resource collection from multiple sources.
    
    This class handles the collection of resources from various sources
    such as URLs and file paths, delegating to specialized collectors.
    """
    
    def __init__(
        self, 
        max_concurrent_requests: int = 5,
        max_file_size_mb: float = 10.0
    ):
        """
        Initialize the resource collector.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent web requests
            max_file_size_mb: Maximum file size to process in MB
        """
        # Import specialized collectors here to avoid circular imports
        from osyllabi.generator.resource.web import WebResourceCollector
        from osyllabi.generator.resource.file import FileResourceCollector
        
        # Initialize specialized collectors
        self.web_collector = WebResourceCollector(
            max_concurrent_requests=max_concurrent_requests
        )
        
        self.file_collector = FileResourceCollector(
            max_file_size_mb=max_file_size_mb
        )
        
        # Collection statistics
        self.stats = {
            "total_urls": 0,
            "total_files": 0,
            "total_content_size": 0,
            "success_rate": 0.0
        }
    
    def collect(self, urls: List[str], paths: List[str]) -> Dict[str, Any]:
        """
        Collect resources from URLs and file paths.
        
        Args:
            urls: List of URLs to collect from
            paths: List of file/directory paths to collect from
            
        Returns:
            Dictionary of collected resources organized by source
        """
        resources = {
            "urls": {},
            "files": {},
            "stats": {},
            "metadata": {
                "keywords": [],
                "sources": []
            }
        }
        
        # Collect from URLs
        if urls:
            log.info(f"Processing {len(urls)} URLs")
            url_resources = self.web_collector.collect(urls)
            resources["urls"] = url_resources.get("urls", {})
            
            # Update metadata and stats
            resources["metadata"]["sources"].extend(url_resources.get("metadata", {}).get("sources", []))
            resources["metadata"]["keywords"].extend(url_resources.get("metadata", {}).get("keywords", []))
            self.stats["total_urls"] += len(urls)
            self.stats["total_content_size"] += url_resources.get("stats", {}).get("total_content_size", 0)
        
        # Collect from file paths
        if paths:
            log.info(f"Processing {len(paths)} file/directory paths")
            file_resources = self.file_collector.collect(paths)
            resources["files"] = file_resources.get("files", {})
            
            # Update metadata and stats
            resources["metadata"]["keywords"].extend(file_resources.get("metadata", {}).get("keywords", []))
            self.stats["total_files"] += file_resources.get("stats", {}).get("files_processed", 0)
            self.stats["total_content_size"] += file_resources.get("stats", {}).get("total_content_size", 0)
        
        # Deduplicate and sort keywords
        resources["metadata"]["keywords"] = sorted(list(set(resources["metadata"]["keywords"])))
        resources["metadata"]["sources"] = sorted(list(set(resources["metadata"]["sources"])))
        
        # Calculate success rate
        total_attempted = (
            self.web_collector.get_stats().get("urls_processed", 0) + 
            self.web_collector.get_stats().get("urls_failed", 0) +
            self.file_collector.get_stats().get("files_processed", 0) + 
            self.file_collector.get_stats().get("files_failed", 0)
        )
        
        total_succeeded = (
            self.web_collector.get_stats().get("urls_processed", 0) + 
            self.file_collector.get_stats().get("files_processed", 0)
        )
        
        if total_attempted > 0:
            self.stats["success_rate"] = total_succeeded / total_attempted
        
        # Add overall statistics
        resources["stats"] = self.stats.copy()
        
        return resources
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        return self.stats
