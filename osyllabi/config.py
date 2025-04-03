from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.getenv('OSYLLABUS_OUTPUT_DIR', BASE_DIR / 'output')

# AI Configuration
AI_CONFIG = {
    'model': os.getenv('OSYLLABUS_AI_MODEL', 'gpt-4'),
    'temperature': float(os.getenv('OSYLLABUS_AI_TEMP', '0.7')),
    'max_tokens': int(os.getenv('OSYLLABUS_AI_MAX_TOKENS', '4000')),
}

# Search Configuration
SEARCH_CONFIG = {
    'engines': ['duckduckgo'] + (os.getenv('OSYLLABUS_EXTRA_ENGINES', '').split(',') if os.getenv('OSYLLABUS_EXTRA_ENGINES') else []),
    'max_results': int(os.getenv('OSYLLABUS_MAX_RESULTS', '30')),
    'search_depth': int(os.getenv('OSYLLABUS_SEARCH_DEPTH', '2')),
}

# Supported file types for sources
SUPPORTED_FILE_EXTENSIONS = [
    '.txt', '.md', '.tex', '.rst', '.html',
    '.py', '.js', '.java', '.cpp', '.c',
    '.json', '.yaml', '.yml', '.csv',
]

# Export formats
EXPORT_FORMATS = ['md', 'pdf', 'html', 'docx']
DEFAULT_EXPORT_FORMAT = os.getenv('OSYLLABUS_DEFAULT_FORMAT', 'md')
