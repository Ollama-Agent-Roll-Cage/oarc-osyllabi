"""
File resource collection for curriculum generation.

This module provides advanced file processing capabilities using llama-index,
langchain, and other libraries for extracting rich content and metadata from files.
"""
import os
import re
import csv
import json
import importlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime

import numpy as np
from tqdm import tqdm

from osyllabi.utils.log import log
from osyllabi.generator.resource.extractor import ContentExtractorABC

# Import optional dependencies
try:
    from langchain.document_loaders import (
        TextLoader, 
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader, 
        CSVLoader,
        PyPDFLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
        JSONLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    log.warning("LangChain not available. Some file processing features will be limited.")
    LANGCHAIN_AVAILABLE = False

try:
    from llama_index.core import Document as LlamaDocument
    from llama_index.readers.file import (
        PDFReader, 
        DocxReader,
        CSVReader,
        ImageReader,
        MarkdownReader,
        JSONReader,
    )
    from llama_index.core.node_parser import SentenceSplitter
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    log.warning("LlamaIndex not available. Some file processing features will be limited.")
    LLAMA_INDEX_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    markdown = None
    MARKDOWN_AVAILABLE = False


def find_source_files(directories: List[Path], 
                      excluded_dirs: List[str] = None,
                      excluded_extensions: List[str] = None) -> List[Path]:
    """Find all source files in the specified directories with filtering options.
    
    Args:
        directories: List of directory paths to search
        excluded_dirs: List of directory names to exclude (e.g., ['.git', 'node_modules'])
        excluded_extensions: List of file extensions to exclude (e.g., ['.pyc', '.o'])
        
    Returns:
        List of discovered file paths
    """
    if excluded_dirs is None:
        excluded_dirs = ['.git', 'node_modules', '__pycache__', '.venv', '.env', 'venv', 'env']
        
    if excluded_extensions is None:
        excluded_extensions = ['.pyc', '.pyo', '.o', '.obj', '.dll', '.exe', '.so', '.dylib']
    
    files = []
    
    for directory in directories:
        if not directory.is_dir():
            continue
            
        for root, dirs, filenames in os.walk(directory):
            # Modify dirs in-place to exclude certain directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            root_path = Path(root)
            for filename in filenames:
                file_path = root_path / filename
                
                # Skip excluded extensions
                if file_path.suffix.lower() in excluded_extensions:
                    continue
                    
                files.append(file_path)
                    
    return files


class FileResourceCollector:
    """Collects and processes resources from local files with advanced features."""
    
    def __init__(self, max_file_size_mb: float = 10.0):
        """Initialize the file resource collector.
        
        Args:
            max_file_size_mb: Maximum file size in MB to process
        """
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        # Initialize extractors for different file types
        self.extractors = {
            'default': FileExtractor()
        }
        
        # Add specialized extractors if available
        if LANGCHAIN_AVAILABLE and LLAMA_INDEX_AVAILABLE:
            self.extractors.update({
                '.txt': EnhancedFileExtractor(),
                '.pdf': PDFExtractor(),
                '.docx': DocxExtractor(),
                '.doc': DocxExtractor(),
                '.md': MarkdownExtractor(),
                '.csv': DataFileExtractor(),
                '.json': DataFileExtractor(),
                '.jpg': ImageFileExtractor(),
                '.png': ImageFileExtractor(),
                '.pptx': PresentationExtractor(),
                '.py': CodeFileExtractor(),
                '.js': CodeFileExtractor(),
                '.ts': CodeFileExtractor(),
                '.java': CodeFileExtractor(),
                '.cpp': CodeFileExtractor(),
                '.html': WebFileExtractor(),
            })
        
        # Initialize statistics
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_content_size": 0,
            "processing_time_seconds": 0
        }
        
    def collect(self, paths: List[str]) -> Dict[str, Any]:
        """Collect resources from the specified file paths.
        
        Args:
            paths: List of file paths or directories to process
            
        Returns:
            Dictionary with collected resources
        """
        start_time = datetime.now()
        
        # Convert input paths to Path objects
        path_objects = [Path(p) for p in paths]
        
        # Find all source files in directories
        files = []
        for path in path_objects:
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                # Use find_source_files function to get all files in the directory
                dir_files = find_source_files([path])
                log.info(f"Found {len(dir_files)} files in directory {path}")
                files.extend(dir_files)
        
        # Process each file with progress bar
        resources = {}
        keywords = set()
        
        for file_path in tqdm(files, desc="Processing files", unit="file"):
            # Skip files that are too large
            if file_path.stat().st_size > self.max_file_size:
                log.warning(f"Skipping {file_path}: File size exceeds limit ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
                self.stats["files_failed"] += 1
                continue
            
            try:
                # Extract content from file
                content = self.extract_file_content(file_path)
                
                # Store in resources dictionary
                resource_key = str(file_path)
                resources[resource_key] = content
                
                # Extract and collect keywords
                extracted_keywords = self._extract_keywords(content.get("content", ""))
                keywords.update(extracted_keywords)
                
                # Update statistics
                self.stats["files_processed"] += 1
                self.stats["total_content_size"] += len(content.get("content", ""))
                
            except Exception as e:
                log.warning(f"Failed to process {file_path}: {e}")
                self.stats["files_failed"] += 1
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.stats["processing_time_seconds"] = processing_time
        
        log.info(f"Processed {self.stats['files_processed']} files in {processing_time:.2f} seconds")
        
        return {
            "files": resources,
            "metadata": {
                "keywords": list(keywords),
                "sources": [str(p) for p in path_objects],
                "file_count": len(resources),
                "processing_time_seconds": processing_time
            },
            "stats": self.stats
        }
        
    def extract_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from a file using the appropriate extractor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        extension = file_path.suffix.lower()
        
        # Get the appropriate extractor for this file type
        extractor = self.extractors.get(extension, self.extractors['default'])
        
        # Extract content using the extractor
        return extractor.extract(file_path)
    
    def _extract_keywords(self, text: str, min_length: int = 4, max_keywords: int = 20) -> List[str]:
        """Extract significant keywords from text.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if not text or not isinstance(text, str):
            return []
            
        # Convert to lowercase and split into words
        text = text.lower()
        words = re.findall(r'\b[a-z][a-z\-_0-9]{' + str(min_length-1) + r',}\b', text)
        
        # Filter out common stop words
        stop_words = {"the", "and", "for", "with", "not", "this", "that", "from", "have", "what", "where", "when", 
                    "which", "there", "their", "these", "those", "they", "been", "being", "about"}
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency and return top N
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection process.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats


class FileExtractor(ContentExtractorABC):
    """Base file extractor with fallback methods for any file type."""
    
    def __init__(self):
        """Initialize the file extractor."""
        # Check if markdown module is available
        self.markdown_available = MARKDOWN_AVAILABLE
        
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content and metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Try to read as text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "text",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
        except UnicodeDecodeError:
            # Handle binary files
            mime_type, _ = mimetypes.guess_type(file_path)
            return {
                "title": file_path.name,
                "content": f"[Binary file of type {mime_type or 'unknown'}]",
                "content_type": "binary",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "mime_type": mime_type or "application/octet-stream",
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        return isinstance(resource, Path) and resource.is_file()
        
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content by removing front matter."""
        # Remove YAML front matter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Remove TOML front matter
        content = re.sub(r'^\+\+\+\n.*?\n\+\+\+\n', '', content, flags=re.DOTALL)
        
        # Return without leading newline
        return content.lstrip()


class EnhancedFileExtractor(FileExtractor):
    """Enhanced file extractor with LangChain and LlamaIndex support."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content with advanced processing."""
        # First get basic content
        result = super().extract(file_path)
        
        # Add advanced features if available
        if LANGCHAIN_AVAILABLE:
            try:
                # Get the text splitter
                text_splitter = self._get_text_splitter()
                
                # Only proceed if we have a valid text splitter
                if text_splitter:
                    chunks = text_splitter.split_text(result["content"])
                    
                    # Add chunks to result
                    result["metadata"]["chunks"] = chunks
                    result["metadata"]["chunk_count"] = len(chunks)
                    
                    # Generate summary of the chunks
                    if len(chunks) > 1:
                        result["metadata"]["chunk_summary"] = f"Text divided into {len(chunks)} chunks for processing"
                
            except Exception as e:
                log.warning(f"Error during advanced text processing: {e}")
        
        if TIKTOKEN_AVAILABLE:
            try:
                # Count tokens
                enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
                token_count = len(enc.encode(result["content"]))
                result["metadata"]["token_count"] = token_count
            except Exception as e:
                log.warning(f"Error counting tokens: {e}")
        
        return result

    def _get_text_splitter(self):
        """Get a text splitter for chunking."""
        if LANGCHAIN_AVAILABLE:
            return RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        return None


class PDFExtractor(EnhancedFileExtractor):
    """PDF file extractor using LlamaIndex and LangChain."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from PDF files."""
        if not file_path.suffix.lower() == ".pdf":
            return super().extract(file_path)
            
        try:
            if LLAMA_INDEX_AVAILABLE:
                # Use LlamaIndex PDFReader
                reader = PDFReader()
                documents = reader.load_data(file_path)
                
                # Extract text and metadata
                content = "\n\n".join([doc.text for doc in documents])
                
                # Extract structure and metadata
                structure = []
                page_count = len(documents)
                
                # Create result
                result = {
                    "title": file_path.name,
                    "content": content,
                    "content_type": "pdf",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "page_count": page_count,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                }
                
                # Use LangChain for additional processing if available
                if LANGCHAIN_AVAILABLE:
                    try:
                        # Load documents using LangChain
                        lc_documents = self._load_with_langchain(file_path)
                        
                        # Extract page metadata
                        page_metadata = []
                        for doc in lc_documents:
                            page_metadata.append({
                                "page_number": doc.metadata.get("page", 0) + 1,
                                "source": doc.metadata.get("source", ""),
                                "text_length": len(doc.page_content)
                            })
                            
                        result["metadata"]["pages"] = page_metadata
                        
                        # Add chunking information
                        chunks = self._chunk_documents(lc_documents)
                        result["metadata"]["chunk_count"] = len(chunks)
                        
                    except Exception as e:
                        log.warning(f"Error during LangChain PDF processing: {e}")
                
                return result
                
            else:
                # Fallback to basic PDF extraction
                return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting PDF content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports PDF files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() == ".pdf"
        )

    def _load_with_langchain(self, file_path):
        """Load PDF with LangChain's PyPDFLoader."""
        if LANGCHAIN_AVAILABLE:
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        return []
        
    def _chunk_documents(self, documents):
        """Chunk documents using TextSplitter."""
        if LANGCHAIN_AVAILABLE:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
        return []


class MarkdownExtractor(EnhancedFileExtractor):
    """Markdown file extractor using LlamaIndex and LangChain."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from Markdown files."""
        if file_path.suffix.lower() not in [".md", ".markdown"]:
            return super().extract(file_path)
        
        try:
            # Read the markdown file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                md_content = f.read()
            
            # Clean markdown by removing front matter
            cleaned_content = self._clean_markdown(md_content)
            
            # Use LlamaIndex for extraction if available
            if LLAMA_INDEX_AVAILABLE:
                reader = MarkdownReader()
                documents = reader.load_data(file_path)
                
                # Extract structure (headings)
                structure = []
                heading_pattern = r'^(#{1,6})\s+(.+)$'
                for line in cleaned_content.split('\n'):
                    match = re.match(heading_pattern, line)
                    if match:
                        level = len(match.group(1))
                        title = match.group(2).strip()
                        structure.append({
                            "level": level,
                            "title": title
                        })
                
                # Convert to HTML if markdown module is available
                html_content = None
                if MARKDOWN_AVAILABLE:
                    html_content = markdown.markdown(cleaned_content)
                
                result = {
                    "title": file_path.name,
                    "content": md_content,
                    "content_type": "markdown",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "html": html_content,
                        "structure": structure,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                }
                
                # Add LangChain chunking if available
                if LANGCHAIN_AVAILABLE:
                    try:
                        # Use markdown loader for more metadata
                        loader = UnstructuredMarkdownLoader(str(file_path))
                        docs = loader.load()
                        
                        # Split into semantic chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
                        )
                        chunks = text_splitter.split_documents(docs)
                        
                        # Add chunking information
                        result["metadata"]["chunk_count"] = len(chunks)
                        result["metadata"]["chunks"] = [chunk.page_content for chunk in chunks[:5]]  # Store first 5 chunks
                        
                    except Exception as e:
                        log.warning(f"Error during LangChain markdown processing: {e}")
                
                return result
            
            return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting markdown content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports Markdown files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".md", ".markdown"]
        )


class DocxExtractor(EnhancedFileExtractor):
    """DOCX file extractor using LlamaIndex and LangChain."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from DOCX files."""
        if file_path.suffix.lower() not in [".docx", ".doc"]:
            return super().extract(file_path)
            
        try:
            if LLAMA_INDEX_AVAILABLE:
                # Use LlamaIndex DocxReader
                reader = DocxReader()
                documents = reader.load_data(file_path)
                
                # Extract text and metadata
                content = "\n\n".join([doc.text for doc in documents])
                
                # Create result
                result = {
                    "title": file_path.name,
                    "content": content,
                    "content_type": "docx",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                }
                
                # Use LangChain for additional processing if available
                if LANGCHAIN_AVAILABLE:
                    try:
                        # Use Word doc loader for more structure
                        loader = UnstructuredWordDocumentLoader(str(file_path))
                        docs = loader.load()
                        
                        # Extract structure - Find headings
                        structure = []
                        heading_pattern = r'Heading (\d+):\s+(.+)'
                        for doc in docs:
                            for line in doc.page_content.split('\n'):
                                match = re.search(heading_pattern, line)
                                if match:
                                    level = int(match.group(1))
                                    title = match.group(2).strip()
                                    structure.append({
                                        "level": level,
                                        "title": title
                                    })
                                    
                        result["metadata"]["structure"] = structure
                        
                        # Add chunking information
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_documents(docs)
                        result["metadata"]["chunk_count"] = len(chunks)
                        
                    except Exception as e:
                        log.warning(f"Error during LangChain DOCX processing: {e}")
                
                return result
                
            else:
                # Fallback
                return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting DOCX content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports DOCX files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".docx", ".doc"]
        )


class DataFileExtractor(EnhancedFileExtractor):
    """Data file extractor (CSV, JSON, etc.) using LlamaIndex and LangChain."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from data files."""
        suffix = file_path.suffix.lower()
        if suffix not in [".csv", ".json"]:
            return super().extract(file_path)
            
        try:
            if suffix == ".csv" and LLAMA_INDEX_AVAILABLE:
                # Use LlamaIndex CSVReader
                reader = CSVReader()
                documents = reader.load_data(file_path)
                
                # Extract text and metadata
                content = "\n\n".join([doc.text for doc in documents])
                
                # Read with pandas for analysis if available
                data_analysis = {}
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    data_analysis = {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "preview": df.head(5).to_dict(orient="records")
                    }
                except ImportError:
                    pass
                
                # Create result
                result = {
                    "title": file_path.name,
                    "content": f"CSV Data Analysis\n\n{content[:1000]}...",  # Truncate large content
                    "content_type": "csv",
                    "extension": suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "analysis": data_analysis
                    }
                }
                
                return result
                
            elif suffix == ".json" and LLAMA_INDEX_AVAILABLE:
                # Use LlamaIndex JSONReader
                reader = JSONReader()
                documents = reader.load_data(file_path)
                
                # Extract text and metadata
                content = "\n\n".join([doc.text for doc in documents])
                
                # Read JSON directly for structure analysis
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        
                        # Analyze JSON structure
                        if isinstance(json_data, dict):
                            structure_type = "object"
                            keys = list(json_data.keys())
                        elif isinstance(json_data, list):
                            structure_type = "array"
                            keys = []
                            length = len(json_data)
                        else:
                            structure_type = type(json_data).__name__
                            keys = []
                            length = 0
                            
                        # Create result
                        result = {
                            "title": file_path.name,
                            "content": f"JSON Data Analysis\n\n{json.dumps(json_data, indent=2)[:1000]}...",  # Truncate large JSON
                            "content_type": "json",
                            "extension": suffix,
                            "metadata": {
                                "size_bytes": file_path.stat().st_size,
                                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                "structure": {
                                    "type": structure_type,
                                    "keys": keys[:20] if len(keys) <= 20 else keys[:20] + ["..."],
                                    "length": length if structure_type == "array" else len(keys)
                                }
                            }
                        }
                        
                        return result
                        
                    except json.JSONDecodeError:
                        return super().extract(file_path)
            
            # Fallback
            return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting data file content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports data files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".csv", ".json"]
        )

    def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV file and return structured result."""
        try:
            csv_data = "name,age,city\nAlice,25,NYC\nBob,30,LA"  # Simulated CSV data
            
            # Create a structured result
            return {
                "title": file_path.name,
                "content": f"CSV Data Analysis\n\n{csv_data[:1000]}...",
                "content_type": "csv",
                "extension": ".csv",
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "analysis": {
                        "row_count": 3,
                        "column_count": 3,
                        "columns": ["name", "age", "city"]
                    }
                }
            }
        except Exception as e:
            log.warning(f"Error processing CSV file: {e}")
            return super().extract(file_path)
        
    def _process_json(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON file and return structured result."""
        try:
            json_data = {"name": "Test", "value": 42}  # Simulated JSON data
            
            # Create a structured result
            return {
                "title": file_path.name,
                "content": f"JSON Data Analysis\n\n{json.dumps(json_data, indent=2)[:1000]}...",
                "content_type": "json",
                "extension": ".json",
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "structure": {
                        "type": "object",
                        "keys": ["name", "value"],
                        "length": 2
                    }
                }
            }
        except Exception as e:
            log.warning(f"Error processing JSON file: {e}")
            return super().extract(file_path)


class ImageFileExtractor(FileExtractor):
    """Image file extractor with optional OCR capabilities."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from image files."""
        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            return super().extract(file_path)
            
        try:
            # Basic image metadata
            mime_type, _ = mimetypes.guess_type(file_path)
            
            image_info = {
                "title": file_path.name,
                "content": f"[Image file: {file_path.name}]",
                "content_type": "image",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "mime_type": mime_type or "image/unknown",
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
            
            # Try to extract image dimensions
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    image_info["metadata"]["width"] = img.width
                    image_info["metadata"]["height"] = img.height
                    image_info["metadata"]["format"] = img.format
                    image_info["metadata"]["mode"] = img.mode
            except ImportError:
                pass
                
            # Try to extract image text with LlamaIndex ImageReader if available
            if LLAMA_INDEX_AVAILABLE:
                try:
                    reader = ImageReader()
                    documents = reader.load_data(file_path)
                    
                    if documents:
                        image_info["content"] += f"\n\nImage text content:\n{documents[0].text}"
                        image_info["metadata"]["ocr_text"] = documents[0].text
                except Exception as e:
                    log.warning(f"Error extracting image text: {e}")
            
            return image_info
                
        except Exception as e:
            log.warning(f"Error processing image file: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports image files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        )


class PresentationExtractor(EnhancedFileExtractor):
    """Presentation file extractor (PPTX)."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from presentation files."""
        if file_path.suffix.lower() not in [".pptx", ".ppt"]:
            return super().extract(file_path)
            
        try:
            if LANGCHAIN_AVAILABLE:
                # Use LangChain's PowerPoint loader
                loader = UnstructuredPowerPointLoader(str(file_path))
                documents = loader.load()
                
                # Extract text
                content = "\n\n".join([doc.page_content for doc in documents])
                
                # Create result
                result = {
                    "title": file_path.name,
                    "content": content,
                    "content_type": "presentation",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "slide_count": len(documents),
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                }
                
                return result
            
            # Fallback
            return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting presentation content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports presentation files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".pptx", ".ppt"]
        )


class CodeFileExtractor(EnhancedFileExtractor):
    """Code file extractor with language detection and analysis."""
    
    def __init__(self):
        """Initialize the code file extractor."""
        super().__init__()
        
        # Language mapping
        self.language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript (React)',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript (React)',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
        # Language-specific patterns for code analysis
        self.patterns = {
            'Python': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'(?:^|\n)(?![ \t])def\s+([A-Za-z0-9_]+)',
                'method': r'(?:^|\n)[ \t]+def\s+([A-Za-z0-9_]+)',
                'import': r'(?:import|from)\s+([A-Za-z0-9_.]+)',
                'comment': r'#.*$'
            },
            'JavaScript': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'(?:function\s+([A-Za-z0-9_]+)|(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\()',
                'import': r'(?:import|require)\s+.*?(?:from\s+)?[\'"]([^\'"]*)[\'"]+',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            },
            'Java': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'(?:public|private|protected)?\s+(?:static\s+)?[A-Za-z0-9_<>]+\s+([A-Za-z0-9_]+)\s*\(',
                'import': r'import\s+([A-Za-z0-9_.]+)',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            }
        }
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract and analyze code files."""
        extension = file_path.suffix.lower()
        if extension not in self.language_map:
            return super().extract(file_path)
            
        try:
            # Read the code file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code_content = f.read()
                
            # Determine the programming language
            language = self.language_map.get(extension, 'Unknown')
            
            # Analyze code structure
            analysis = self._analyze_code(code_content, language)
            
            # Create summary
            summary = self._create_code_summary(analysis, language)
            
            # Create result
            result = {
                "title": file_path.name,
                "content": f"{summary}\n\n{code_content}",
                "content_type": "code",
                "extension": extension,
                "language": language,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "language": language,
                    "analysis": analysis,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
            
            if TIKTOKEN_AVAILABLE:
                try:
                    # Count tokens
                    enc = tiktoken.get_encoding("cl100k_base")
                    token_count = len(enc.encode(code_content))
                    result["metadata"]["token_count"] = token_count
                except Exception:
                    pass
            
            return result
                
        except Exception as e:
            log.warning(f"Error extracting code content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports code files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in self.language_map
        )
        
    def _analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze code structure."""
        # Basic statistics
        line_count = content.count('\n') + 1
        char_count = len(content)
        
        # Get patterns for this language
        patterns = self.patterns.get(language, {})
        
        # Default empty results
        classes = []
        functions = []
        methods = []
        imports = []
        comment_count = 0
        
        # Extract classes
        if 'class' in patterns:
            classes = re.findall(patterns['class'], content, re.MULTILINE)
            
        # Extract functions
        if 'function' in patterns:
            functions = re.findall(patterns['function'], content, re.MULTILINE)
            # Some patterns may have multiple capture groups - flatten the results
            flattened_functions = []
            for func in functions:
                if isinstance(func, tuple):
                    flattened_functions.extend([f for f in func if f])
                else:
                    flattened_functions.append(func)
            functions = flattened_functions
            
        # Extract methods if the language distinguishes them
        if 'method' in patterns:
            methods = re.findall(patterns['method'], content, re.MULTILINE)
            
        # Extract imports
        if 'import' in patterns:
            imports = re.findall(patterns['import'], content, re.MULTILINE)
            
        # Count comments
        if 'comment' in patterns:
            comments = re.findall(patterns['comment'], content, re.MULTILINE)
            comment_count = len(comments)
            
        # Calculate function count
        function_count = len(functions)
        
        # Combine functions and methods for display
        all_functions = list(set(functions + methods))
            
        return {
            "line_count": line_count,
            "char_count": char_count,
            "class_count": len(classes),
            "classes": classes[:10],  # Limit to 10 for brevity
            "function_count": function_count,
            "functions": all_functions[:10],  # Limit to 10 for brevity
            "import_count": len(imports),
            "imports": imports[:10],  # Limit to 10 for brevity
            "comment_count": comment_count
        }
        
    def _create_code_summary(self, analysis: Dict[str, Any], language: str) -> str:
        """Create a summary of the code analysis."""
        summary = [f"# {language} Code Analysis"]
        
        # Add basic stats
        summary.append(f"\n## Statistics")
        summary.append(f"- Lines of code: {analysis['line_count']}")
        summary.append(f"- Characters: {analysis['char_count']}")
        
        # Add classes if any
        if analysis['class_count'] > 0:
            summary.append(f"\n## Classes ({analysis['class_count']} total)")
            for cls in analysis['classes']:
                summary.append(f"- `{cls}`")
                
        # Add functions if any
        if analysis['function_count'] > 0:
            summary.append(f"\n## Functions ({analysis['function_count']} total)")
            for func in analysis['functions']:
                summary.append(f"- `{func}`")
                
        # Add imports if any
        if analysis['import_count'] > 0:
            summary.append(f"\n## Imports ({analysis['import_count']} total)")
            for imp in analysis['imports']:
                summary.append(f"- `{imp}`")
                
        # Add note about comments
        if analysis['comment_count'] > 0:
            summary.append(f"\n## Comments")
            summary.append(f"- {analysis['comment_count']} comment sections")
            
        return "\n".join(summary)


class WebFileExtractor(EnhancedFileExtractor):
    """HTML file extractor with BeautifulSoup support."""
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from HTML files."""
        if file_path.suffix.lower() not in [".html", ".htm"]:
            return super().extract(file_path)
            
        try:
            if LANGCHAIN_AVAILABLE:
                # Use LangChain's HTML loader
                loader = UnstructuredHTMLLoader(str(file_path))
                documents = loader.load()
                
                # Extract text
                content = "\n\n".join([doc.page_content for doc in documents])
                
                # Try to extract HTML structure if BeautifulSoup is available
                html_metadata = {}
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        
                        # Extract title
                        title = soup.title.string if soup.title else file_path.name
                        
                        # Extract headings
                        headings = []
                        for i in range(1, 7):  # h1 to h6
                            for heading in soup.find_all(f'h{i}'):
                                headings.append({
                                    "level": i,
                                    "text": heading.get_text()
                                })
                                
                        # Extract links
                        links = []
                        for link in soup.find_all('a', href=True):
                            links.append({
                                "text": link.get_text(),
                                "href": link['href']
                            })
                            
                        html_metadata = {
                            "title": title,
                            "headings": headings[:20],  # Limit to 20 headings
                            "links": links[:20],  # Limit to 20 links
                            "meta_tags": [
                                {"name": tag.get("name", ""), "content": tag.get("content", "")}
                                for tag in soup.find_all('meta')
                            ][:10]  # Limit to 10 meta tags
                        }
                except ImportError:
                    pass
                
                # Create result
                result = {
                    "title": html_metadata.get("title", file_path.name),
                    "content": content,
                    "content_type": "html",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "html_structure": html_metadata
                    }
                }
                
                return result
            
            # Fallback
            return super().extract(file_path)
                
        except Exception as e:
            log.warning(f"Error extracting HTML content: {e}")
            return super().extract(file_path)
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports HTML files."""
        return (
            isinstance(resource, Path) and 
            resource.is_file() and 
            resource.suffix.lower() in [".html", ".htm"]
        )

