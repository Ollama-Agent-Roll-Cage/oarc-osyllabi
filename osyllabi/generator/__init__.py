"""
Curriculum generation functionality.

This package provides components for generating curriculum content
using AI and resource collection.
"""

# Import main components
from osyllabi.generator.resources import ResourceCollectionManager
from osyllabi.generator.workflow import CurriculumWorkflow
from osyllabi.generator.resource import (
    ResourceCollector,
    ContentExtractorABC,
    ResourceManager
)

__all__ = [
    'ResourceCollectionManager',
    'CurriculumWorkflow',
    'ResourceCollector',
    'ContentExtractorABC',
    'ResourceManager'
]
