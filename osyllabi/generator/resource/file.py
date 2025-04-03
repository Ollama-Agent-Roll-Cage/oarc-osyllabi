"""
File resource collection for curriculum generation.

This module provides classes for collecting and extracting content from local files,
including handling different file types and formats.
"""
import re
import json
import mimetypes
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from osyllabi.utils.log import log
from osyllabi.utils.paths import find_source_files
from osyllabi.generator.resource.collector import CollectorABC
from osyllabi.generator.resource.extractor import ContentExtractorABC


class FileResourceCollector(CollectorABC):
    """
    Collects content from local files and directories.
    
    This class handles finding and processing content from local file resources,
    supporting various file types and formats.
    """
    
    def __init__(
        self, 
        max_file_size_mb: float = 10.0,
        max_content_length: int = 10000
    ):
        """
        Initialize the file resource collector.
        
        Args:
            max_file_size_mb: Maximum file size to process in MB
            max_content_length: Maximum length of content to extract
        """
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.max_content_length = max_content_length
        
        # Initialize extractors for different file types
        self.extractors = {
            "default": FileExtractor(),
        }
        
        # Import specialized extractors
        from osyllabi.generator.resource.code import CodeFileExtractor
        from osyllabi.generator.resource.data import DataFileExtractor
        
        # Add specialized extractors
        self.extractors[".py"] = CodeFileExtractor()
        self.extractors[".js"] = CodeFileExtractor()
        self.extractors[".java"] = CodeFileExtractor()
        self.extractors[".cpp"] = CodeFileExtractor()
        self.extractors[".c"] = CodeFileExtractor()
        self.extractors[".csv"] = DataFileExtractor()
        self.extractors[".json"] = DataFileExtractor()
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_content_size": 0
        }
    
    def collect(self, paths: List[str]) -> Dict[str, Any]:
        """
        Collect resources from file paths.
        
        Args:
            paths: List of file/directory paths to collect from
            
        Returns:
            Dictionary of collected resources organized by file path
        """
        resources = {
            "files": {},
            "metadata": {
                "keywords": []
            },
            "stats": {}
        }
        
        if not paths:
            resources["stats"] = self.stats
            return resources
            
        # Process individual files and directories
        for path_str in paths:
            path = Path(path_str)
            
            if path.is_file():
                try:
                    file_content = self.extract_file_content(path)
                    if file_content:
                        resources["files"][str(path)] = file_content
                        
                        # Extract keywords
                        keywords = self._extract_keywords(file_content.get("content", ""))
                        resources["metadata"]["keywords"].extend(keywords)
                except Exception as e:
                    self.stats["files_failed"] += 1
                    log.error(f"Failed to extract content from file {path}: {e}")
            
            elif path.is_dir():
                # Find all source files in the directory
                source_files = find_source_files([path])
                
                for source_file in source_files:
                    try:
                        # Skip files that are too large
                        if source_file.stat().st_size > self.max_file_size:
                            log.warning(f"Skipping large file: {source_file} ({source_file.stat().st_size/1024/1024:.1f} MB)")
                            continue
                            
                        file_content = self.extract_file_content(source_file)
                        if file_content:
                            resources["files"][str(source_file)] = file_content
                            
                            # Extract keywords
                            keywords = self._extract_keywords(file_content.get("content", ""))
                            resources["metadata"]["keywords"].extend(keywords)
                    except Exception as e:
                        self.stats["files_failed"] += 1
                        log.error(f"Failed to extract content from file {source_file}: {e}")
        
        # Update statistics
        resources["stats"] = self.stats.copy()
        
        return resources
    
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
        self.stats["files_processed"] += 1
        
        try:
            # Get file size and check limit
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                log.warning(f"File exceeds size limit: {path} ({file_size/1024/1024:.1f} MB)")
                return {
                    "title": path.name,
                    "path": str(path),
                    "extension": path.suffix,
                    "content": f"[Content too large - {file_size/1024/1024:.1f} MB]",
                    "metadata": {"size_bytes": file_size}
                }
            
            # Use appropriate extractor based on file extension
            extension = path.suffix.lower()
            extractor = self.extractors.get(extension, self.extractors["default"])
            
            # Extract content
            result = extractor.extract(path)
            
            # Add file path
            result["path"] = str(path)
            self.stats["total_content_size"] += len(result.get("content", ""))
            
            return result
            
        except Exception as e:
            log.warning(f"Error reading file {path}: {e}")
            raise
    
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


class FileExtractor(ContentExtractorABC):
    """Extracts content from generic files."""
    
    def __init__(self):
        """Initialize file extractor."""
        # Check if markdown module is available
        self.markdown_available = importlib.util.find_spec("markdown") is not None
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Get basic file info
            file_size = file_path.stat().st_size
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            # Handle text files with different extensions
            if file_path.suffix.lower() == '.md':
                content_type = "markdown"
                
                # Special handling for markdown
                cleaned_content = self._clean_markdown(content)
                
                # Convert markdown to HTML for better representation if available
                if self.markdown_available:
                    try:
                        import markdown
                        html_content = markdown.markdown(cleaned_content)
                        
                        # Extract structure for metadata
                        headers = re.findall(r'^(#+)\s+(.+?)$', cleaned_content, re.MULTILINE)
                        structure = [{"level": len(h[0]), "title": h[1]} for h in headers]
                        
                        return {
                            "title": file_path.name,
                            "content": cleaned_content,
                            "content_type": content_type,
                            "extension": file_path.suffix,
                            "metadata": {
                                "size_bytes": file_size,
                                "modified_at": modified_time.isoformat(),
                                "html": html_content,
                                "structure": structure
                            }
                        }
                    except ImportError:
                        pass  # Fall back to basic handling
                
                return {
                    "title": file_path.name,
                    "content": cleaned_content,
                    "content_type": content_type,
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_size,
                        "modified_at": modified_time.isoformat()
                    }
                }
            elif file_path.suffix.lower() in ['.txt', '.rst']:
                content_type = "text"
            else:
                # Guess content type based on extension
                content_type = mimetypes.guess_type(str(file_path))[0] or "text/plain"
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": content_type,
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_size,
                    "modified_at": modified_time.isoformat()
                }
            }
        except UnicodeDecodeError:
            # Check for PDF file and try to extract text if possible
            if file_path.suffix.lower() == '.pdf':
                pdf_content = self._extract_pdf_content(file_path)
                if pdf_content:
                    return pdf_content
                    
            # If it can't be read as text, report as binary
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return {
                "title": file_path.name,
                "content": f"[Binary file] - MIME type: {mime_type or 'unknown'}",
                "content_type": "binary",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "mime_type": mime_type
                }
            }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        return isinstance(resource, Path) and resource.is_file()
    
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content, removing front matter."""
        # Remove front matter if present (YAML between --- or +++ lines)
        content = re.sub(r'^---\s*.*?---\s*', '', content, flags=re.DOTALL)
        content = re.sub(r'^\+\+\+\s*.*?\+\+\+\s*', '', content, flags=re.DOTALL)
        
        return content
        
    def _extract_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from PDF file using PyMuPDF if available."""
        try:
            # Check if PyMuPDF is available
            if importlib.util.find_spec("fitz") is not None:
                import fitz  # PyMuPDF
                
                # Extract text from PDF
                text = ""
                doc = fitz.open(file_path)
                metadata = {
                    "page_count": len(doc),
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # Get PDF metadata
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    for key, value in pdf_metadata.items():
                        if value:
                            metadata[key.lower()] = value
                
                # Extract text from each page
                for page in doc:
                    text += page.get_text()
                    
                # Close the document
                doc.close()
                
                return {
                    "title": file_path.name,
                    "content": text,
                    "content_type": "pdf",
                    "extension": file_path.suffix,
                    "metadata": metadata
                }
        except ImportError:
            pass  # PyMuPDF not available
        except Exception as e:
            log.warning(f"Error extracting PDF content: {e}")
            
        return None  # Fall back to binary handling
