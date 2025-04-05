"""
Generator package for Osyllabi.

This package provides functionality for curriculum generation.
"""

# Import from the resource package
from osyllabi.generator.resource import (
    ResourceCollector,
    ContentExtractorABC,
    ResourceManager,
    ResourceCollectionManager
)
from osyllabi.generator.workflow import CurriculumWorkflow

__all__ = [
    'ResourceCollectionManager',
    'CurriculumWorkflow',
    'ResourceCollector',
    'ContentExtractorABC',
    'ResourceManager'
]
