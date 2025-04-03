"""
Vector utilities for RAG operations.

This package provides utility functions for working with vectors in the RAG system.
"""
import os
from osyllabi.utils.log import log
from osyllabi.utils.utils import upgrade_faiss_to_gpu, check_faiss_gpu_capability

# Try to upgrade FAISS to GPU version if auto_upgrade is enabled
AUTO_UPGRADE_FAISS = os.environ.get('OSYLLABI_UPGRADE_FAISS', 'false').lower() in ('true', 'yes', '1')

# Initialize GPU capability flag
FAISS_GPU_ENABLED = False

# Attempt FAISS upgrade if requested
if AUTO_UPGRADE_FAISS:
    success, message = upgrade_faiss_to_gpu()
    if success:
        log.info(f"FAISS GPU upgrade: {message}")
        FAISS_GPU_ENABLED = check_faiss_gpu_capability()
    else:
        log.warning(f"FAISS GPU upgrade not performed: {message}")
else:
    # Still check if we already have GPU support
    FAISS_GPU_ENABLED = check_faiss_gpu_capability()
    if FAISS_GPU_ENABLED:
        log.info("FAISS GPU support detected")

# Import vector operations
from osyllabi.utils.vector.operations import (
    cosine_similarity, normalize_vector, mean_vector, concatenate_vectors,
    create_faiss_index, faiss_search, batch_cosine_similarity, reduce_dimensions
)

__all__ = [
    "cosine_similarity", "normalize_vector", "mean_vector", "concatenate_vectors",
    "create_faiss_index", "faiss_search", "batch_cosine_similarity", "reduce_dimensions",
    "FAISS_GPU_ENABLED"
]
