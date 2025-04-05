"""
Vector operations and utilities.
"""
import numpy as np

from osyllabi.utils.log import log
from osyllabi.utils.deps import DependencyManager

# First import operations (including the FAISS_GPU_ENABLED flag)
from osyllabi.utils.vector.operations import cosine_similarity, FAISS_GPU_ENABLED

# Export dependency manager methods
upgrade_faiss_to_gpu = DependencyManager.upgrade_faiss
check_faiss_gpu_capability = DependencyManager._is_faiss_gpu_installed

__all__ = [
    'cosine_similarity',
    'upgrade_faiss_to_gpu',
    'check_faiss_gpu_capability',
    'FAISS_GPU_ENABLED'
]
