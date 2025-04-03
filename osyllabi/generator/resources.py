"""
Resource collection functionality for curriculum generation.
"""
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Union
from urllib.parse import urlparse

from osyllabi.utils.log import log, with_context
from osyllabi.utils.paths import find_source_files


class ResourceCollector:
    """
    Collects and processes resources for curriculum generation.
    
    This class handles collecting content from URLs and local files,
    extracting relevant information for use in curriculum generation.
    """
    
    def __init__(self, max_url_content_length: int = 10000):
        """
        Initialize the resource collector.
        
        Args:
            max_url_content_length: Maximum length of content to extract from URLs
        """
        self.max_url_content_length = max_url_content_length
    
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
            "files": {}
        }
        
        # Process URLs
        for url in urls:
            try:
                url_content = self.extract_url_content(url)
                if url_content:
                    resources["urls"][url] = url_content
            except Exception as e:
                log.error(f"Failed to extract content from URL {url}: {e}")
        
        # Process paths
        for path in paths:
            path_obj = Path(path)
            if path_obj.is_file():
                try:
                    file_content = self.extract_file_content(path_obj)
                    if file_content:
                        resources["files"][str(path_obj)] = file_content
                except Exception as e:
                    log.error(f"Failed to extract content from file {path}: {e}")
            elif path_obj.is_dir():
                source_files = find_source_files([path_obj])
                for source_file in source_files:
                    try:
                        file_content = self.extract_file_content(source_file)
                        if file_content:
                            resources["files"][str(source_file)] = file_content
                    except Exception as e:
                        log.error(f"Failed to extract content from file {source_file}: {e}")
        
        return resources
    
    def extract_url_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extracted information
            
        Raises:
            requests.RequestException: On request errors
        """
        log.info(f"Extracting content from URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Get basic information
            domain = urlparse(url).netloc
            title = self._extract_title(response.text) or domain
            
            # Extract main content (simplified approach)
            content = self._clean_html_content(response.text)
            if len(content) > self.max_url_content_length:
                content = content[:self.max_url_content_length] + "... [content truncated]"
                
            return {
                "title": title,
                "domain": domain,
                "content": content,
                "url": url
            }
            
        except requests.RequestException as e:
            log.warning(f"Error fetching URL {url}: {e}")
            raise
    
    def extract_file_content(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with extracted information
            
        Raises:
            IOError: On file read errors
        """
        path = Path(file_path)
        log.info(f"Extracting content from file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": path.name,
                "path": str(path),
                "extension": path.suffix,
                "content": content[:self.max_url_content_length] if len(content) > self.max_url_content_length else content
            }
            
        except Exception as e:
            log.warning(f"Error reading file {path}: {e}")
            raise
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content."""
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return ""
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to extract useful text."""
        # Remove script and style elements
        html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        # Replace HTML tags with spaces
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Normalize whitespace
        html_content = re.sub(r'\s+', ' ', html_content).strip()
        
        return html_content
