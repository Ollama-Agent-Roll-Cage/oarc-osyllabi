"""
AI integration modules for Osyllabi.

This package provides functionality for interacting with AI models
to generate curriculum content and analyze resources.
"""

from osyllabi.ai.client import OllamaClient
from osyllabi.ai.prompts import PromptTemplate

__all__ = ["OllamaClient", "PromptTemplate"]
