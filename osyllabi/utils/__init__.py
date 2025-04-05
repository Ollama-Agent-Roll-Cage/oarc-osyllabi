"""
Utility functions and helpers.
"""
from osyllabi.utils.log import log, is_debug_mode
from osyllabi.utils.utils import (
    safe_to_int,
    safe_to_float,
    find_files_by_extensions,
    get_app_dirs,
    get_system_info,
    get_timestamp,
    sanitize_filename,
    check_for_ollama
)
from osyllabi.utils.deps import DependencyManager

# Export DependencyManager methods as module-level functions
check_cuda_capability = DependencyManager.check_cuda_capability
install_cuda_toolkit = DependencyManager.install_cuda_toolkit
upgrade_faiss = DependencyManager.upgrade_faiss
check_deps = DependencyManager.check_deps

__all__ = [
    'log',
    'is_debug_mode',
    'safe_to_int',
    'safe_to_float',
    'find_files_by_extensions',
    'get_app_dirs',
    'get_system_info',
    'get_timestamp',
    'sanitize_filename',
    'check_for_ollama',
    'check_cuda_capability',
    'install_cuda_toolkit',
    'upgrade_faiss',
    'check_deps',
    'DependencyManager'
]
