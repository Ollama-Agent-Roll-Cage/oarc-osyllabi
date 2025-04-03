"""
Vector utilities for RAG operations.

This package provides utility functions for working with vectors in the RAG system.
"""

from osyllabi.utils.vector.operations import (
    cosine_similarity, normalize_vector, mean_vector, concatenate_vectors
)

__all__ = [
    "cosine_similarity", "normalize_vector", "mean_vector", "concatenate_vectors"
]
