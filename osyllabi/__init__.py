"""
Osyllabus: A streamlined Python app for designing personalized curriculums 
using AI, web crawling, and data integration.
"""

__version__ = "0.1.0"
__author__ = "p3nGu1nZz"

from osyllabi.main import main

# Export the curriculum class for direct usage
from osyllabi.curriculum import Curriculum

__all__ = [
    '__version__',
    '__author__',
    'main',
    'Curriculum',
]