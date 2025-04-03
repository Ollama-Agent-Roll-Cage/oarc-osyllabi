"""
Web resource collection for curriculum generation.

This module provides classes for collecting and extracting content from web URLs,
including handling different content types and formats.
"""
import re
import json
import requests
import concurrent.futures
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from osyllabi.utils.log import log
from osyllabi.utils.decorators.retry import retry
from osyllabi.generator.resource.collector import CollectorABC
from osyllabi.generator.resource.extractor import ContentExtractorABC


class WebResourceCollector(CollectorABC):
    """
    Collects content from web URLs.
    
    This class handles fetching and processing content from web URLs,
    including concurrent fetching and error handling.
    """
    
    def __init__(
        self, 
        max_concurrent_requests: int = 5,
        max_content_length: int = 10000,
        user_agent: str = "Osyllabi Curriculum Generator/0.1"
    ):
        """
        Initialize the web resource collector.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent web requests
            max_content_length: Maximum length of content to extract
            user_agent: User agent string for web requests
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_content_length = max_content_length
        self.user_agent = user_agent
        
        # Initialize extractors for different content types
        self.extractors = {
            "html": HTMLExtractor(),
            "json": JSONExtractor(),
            "text": TextExtractor()
        }
        
        # Statistics
        self.stats = {
            "urls_processed": 0,
            "urls_failed": 0,
            "total_content_size": 0
        }
    
    def collect(self, urls: List[str]) -> Dict[str, Any]:
        """
        Collect resources from URLs concurrently.
        
        Args:
            urls: List of URLs to collect from
            
        Returns:
            Dictionary of collected resources organized by URL
        """
        resources = {
            "urls": {},
            "metadata": {
                "keywords": [],
                "sources": []
            },
            "stats": {}
        }
        
        if not urls:
            resources["stats"] = self.stats
            return resources
        
        log.info(f"Processing {len(urls)} URLs concurrently")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            future_to_url = {executor.submit(self.extract_url_content, url): url for url in urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    url_content = future.result()
                    if url_content:
                        resources["urls"][url] = url_content
                        # Add domain to sources
                        domain = urlparse(url).netloc
                        if domain and domain not in resources["metadata"]["sources"]:
                            resources["metadata"]["sources"].append(domain)
                        
                        # Extract keywords
                        keywords = self._extract_keywords(url_content.get("content", ""))
                        resources["metadata"]["keywords"].extend(keywords)
                except Exception as e:
                    self.stats["urls_failed"] += 1
                    log.error(f"Failed to extract content from URL {url}: {e}")
        
        # Update statistics
        resources["stats"] = self.stats.copy()
        
        return resources
    
    @retry(attempts=2, delay=1, backoff=2, exceptions=(requests.RequestException,))
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
        self.stats["urls_processed"] += 1
        
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Determine content type and select appropriate extractor
            content_type = response.headers.get('Content-Type', '').lower()
            extractor = self._get_extractor_for_content_type(content_type)
            
            # Extract content using the appropriate extractor
            result = extractor.extract(response)
            
            # Add URL and track content size
            result["url"] = url
            self.stats["total_content_size"] += len(result.get("content", ""))
            
            return result
            
        except requests.RequestException as e:
            log.warning(f"Error fetching URL {url}: {e}")
            raise
    
    def _get_extractor_for_content_type(self, content_type: str) -> ContentExtractorABC:
        """
        Get the appropriate extractor for a content type.
        
        Args:
            content_type: Content type (e.g., 'text/html', 'application/json')
            
        Returns:
            ContentExtractorABC: Appropriate extractor
        """
        if 'application/json' in content_type:
            return self.extractors["json"]
        elif 'text/html' in content_type:
            return self.extractors["html"]
        elif 'text/plain' in content_type:
            return self.extractors["text"]
        else:
            # Default to HTML extractor for unknown content types
            return self.extractors["html"]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary of collection statistics
        """
        return self.stats
    
    def _extract_keywords(self, text: str, min_length: int = 4, max_keywords: int = 20) -> List[str]:
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


class HTMLExtractor(ContentExtractorABC):
    """Extracts content from HTML resources."""
    
    def extract(self, resource: requests.Response) -> Dict[str, Any]:
        """Extract content from an HTML response."""
        html_content = resource.text
        
        # Use BeautifulSoup for more robust HTML parsing
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic information
        domain = urlparse(resource.url).netloc
        title = self._extract_title(soup) or domain
        
        # Clean HTML content using BeautifulSoup
        content = self._clean_html_content(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(resource, soup, html_content)
        
        return {
            "title": title,
            "domain": domain,
            "content": content,
            "content_type": "html",
            "metadata": metadata
        }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        if isinstance(resource, requests.Response):
            content_type = resource.headers.get('Content-Type', '').lower()
            return 'text/html' in content_type
        return False
        
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML content using BeautifulSoup."""
        # Try to get title element
        title_tag = soup.title
        if title_tag and title_tag.string:
            return title_tag.string.strip()
            
        # Try to find h1 if title tag not found
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        return ""
    
    def _clean_html_content(self, soup: BeautifulSoup) -> str:
        """Clean HTML content using BeautifulSoup."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'head', 'iframe', 'nav']):
            element.extract()
            
        # Get text with preserved paragraph structure
        content = ""
        for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = paragraph.get_text().strip()
            if text:
                # Add markers for headings
                if paragraph.name.startswith('h'):
                    content += f"\n\n{text}\n"
                elif paragraph.name == 'li':
                    content += f"\n• {text}"
                else:
                    content += f"\n\n{text}"
        
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content.strip())
        
        return content
    
    def _extract_metadata(self, response: requests.Response, soup: BeautifulSoup, html_content: str) -> Dict[str, Any]:
        """Extract metadata from HTML content."""
        metadata = {
            "content_type": response.headers.get('Content-Type', 'unknown'),
            "content_length": int(response.headers.get('Content-Length', 0)),
            "last_modified": response.headers.get('Last-Modified', None)
        }
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            metadata['description'] = meta_desc['content']
            
        # Extract OpenGraph metadata
        og_metadata = {}
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            if 'content' in meta.attrs and 'property' in meta.attrs:
                og_key = meta['property'][3:]  # Remove 'og:' prefix
                og_metadata[og_key] = meta['content']
                
        if og_metadata:
            metadata['og'] = og_metadata
            
        return metadata


class JSONExtractor(ContentExtractorABC):
    """Extracts content from JSON resources."""
    
    def extract(self, resource: requests.Response) -> Dict[str, Any]:
        """Extract content from a JSON response."""
        try:
            data = resource.json()
            content = f"JSON content:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
            
            # Try to determine a title from the JSON data
            title = self._extract_title_from_json(data) or urlparse(resource.url).netloc
            
            return {
                "title": title,
                "domain": urlparse(resource.url).netloc,
                "content": content,
                "content_type": "json",
                "metadata": {
                    "content_type": resource.headers.get('Content-Type', 'unknown'),
                    "json_keys": list(data.keys()) if isinstance(data, dict) else []
                }
            }
        except json.JSONDecodeError:
            # If parsing fails, treat as plain text
            return {
                "title": urlparse(resource.url).netloc,
                "domain": urlparse(resource.url).netloc,
                "content": resource.text,
                "content_type": "text",
                "metadata": {
                    "content_type": resource.headers.get('Content-Type', 'unknown'),
                    "parse_error": "Invalid JSON"
                }
            }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        if isinstance(resource, requests.Response):
            content_type = resource.headers.get('Content-Type', '').lower()
            return 'application/json' in content_type
        return False
    
    def _extract_title_from_json(self, data: Any) -> Optional[str]:
        """Attempt to extract a title from JSON data."""
        if isinstance(data, dict):
            # Check common keys that might contain title information
            for key in ['title', 'name', 'heading', 'subject', 'topic']:
                if key in data and isinstance(data[key], str):
                    return data[key]
        return None


class TextExtractor(ContentExtractorABC):
    """Extracts content from plain text resources."""
    
    def extract(self, resource: requests.Response) -> Dict[str, Any]:
        """Extract content from a text response."""
        return {
            "title": urlparse(resource.url).netloc,
            "domain": urlparse(resource.url).netloc,
            "content": resource.text,
            "content_type": "text",
            "metadata": {
                "content_type": resource.headers.get('Content-Type', 'unknown'),
                "content_length": len(resource.text)
            }
        }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        if isinstance(resource, requests.Response):
            content_type = resource.headers.get('Content-Type', '').lower()
            return 'text/plain' in content_type
        return False
