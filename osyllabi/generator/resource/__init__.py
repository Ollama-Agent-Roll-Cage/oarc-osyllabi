"""
Resource collection and extraction functionality for curriculum generation.

This package provides classes for collecting and extracting content from various
sources such as URLs, files, and directories.
"""

from osyllabi.generator.resource.collector import ResourceCollector, CollectorABC
from osyllabi.generator.resource.extractor import ContentExtractorABC
from osyllabi.generator.resource.manager import ResourceManager, ResourceCollectionManager
from osyllabi.generator.resource.web import WebResourceCollector
from osyllabi.generator.resource.file import FileResourceCollector, FileExtractor
from osyllabi.generator.resource.code import CodeFileExtractor
from osyllabi.generator.resource.data import DataFileExtractor

__all__ = [
    "ResourceCollector",
    "CollectorABC",
    "ContentExtractorABC",
    "ResourceManager",
    "WebResourceCollector",
    "FileResourceCollector",
    "FileExtractor",
    "CodeFileExtractor",
    "DataFileExtractor",
    "ResourceCollectionManager"
]
