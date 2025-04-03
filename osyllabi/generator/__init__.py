"""
Curriculum generation modules for Osyllabi.

This package provides functionality for generating curriculum content
based on topics, skill levels, and available resources.
"""

from osyllabi.generator.workflow import CurriculumWorkflow
from osyllabi.generator.resources import ResourceCollector

__all__ = ["CurriculumWorkflow", "ResourceCollector"]
