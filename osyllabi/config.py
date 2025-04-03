"""
Configuration settings for the Osyllabi project.

This module contains global configuration settings that can be overridden
via environment variables for customizing application behavior.
"""
import os
from pathlib import Path
from osyllabi.utils.const import DEFAULT_OLLAMA_URL, DEFAULT_MODEL

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.getenv('OSYLLABUS_OUTPUT_DIR', BASE_DIR / 'output')

# AI Configuration
AI_CONFIG = {
    'model': os.getenv('OSYLLABUS_AI_MODEL', 'gpt-4'),
    'temperature': float(os.getenv('OSYLLABUS_AI_TEMP', '0.7')),
    'max_tokens': int(os.getenv('OSYLLABUS_AI_MAX_TOKENS', '4000')),
    # Ollama specific settings
    'ollama_api_url': os.getenv('OSYLLABUS_OLLAMA_URL', DEFAULT_OLLAMA_URL),
    'default_model': os.getenv('OSYLLABUS_OLLAMA_MODEL', DEFAULT_MODEL),
}

# Search Configuration
SEARCH_CONFIG = {
    'engines': ['duckduckgo'] + (os.getenv('OSYLLABUS_EXTRA_ENGINES', '').split(',') if os.getenv('OSYLLABUS_EXTRA_ENGINES') else []),
    'max_results': int(os.getenv('OSYLLABUS_MAX_RESULTS', '30')),
    'search_depth': int(os.getenv('OSYLLABUS_SEARCH_DEPTH', '2')),
}

# RAG Configuration
RAG_CONFIG = {
    'embedding_model': os.getenv('OSYLLABUS_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    'chunk_size': int(os.getenv('OSYLLABUS_CHUNK_SIZE', '512')),
    'chunk_overlap': int(os.getenv('OSYLLABUS_CHUNK_OVERLAP', '50')),
    'retrieval_top_k': int(os.getenv('OSYLLABUS_RETRIEVAL_TOP_K', '5')),
    'similarity_threshold': float(os.getenv('OSYLLABUS_SIMILARITY_THRESHOLD', '0.7')),
    'enable_rag': os.getenv('OSYLLABUS_ENABLE_RAG', 'true').lower() in ('true', 'yes', '1')
}

# FAISS Configuration
FAISS_CONFIG = {
    'upgrade_on_import': os.getenv('OSYLLABUS_UPGRADE_FAISS', 'false').lower() in ('true', 'yes', '1'),
    'use_gpu': os.getenv('OSYLLABUS_FAISS_USE_GPU', 'true').lower() in ('true', 'yes', '1')
}

# Supported file types for sources
SUPPORTED_FILE_EXTENSIONS = [
    '.txt', '.md', '.tex', '.rst', '.html',
    '.py', '.js', '.java', '.cpp', '.c',
    '.json', '.yaml', '.yml', '.csv',
]

# Export formats
EXPORT_FORMATS = ['md', 'pdf', 'html', 'docx', 'json']
DEFAULT_EXPORT_FORMAT = os.getenv('OSYLLABUS_DEFAULT_FORMAT', 'md')
