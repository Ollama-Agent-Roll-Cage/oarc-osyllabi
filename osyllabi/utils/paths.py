"""
Path management utilities for Osyllabi.
"""
import tempfile
from pathlib import Path
from typing import List, Union

from osyllabi.utils.utils import sanitize_filename
from osyllabi.config import SUPPORTED_FILE_EXTENSIONS


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path: Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to project root
    """
    # This assumes config.py is in the project root
    from osyllabi.config import BASE_DIR
    return BASE_DIR


def get_temp_directory() -> Path:
    """
    Get a temporary directory for Osyllabi.
    
    Returns:
        Path: Path to temporary directory
    """
    temp_dir = Path(tempfile.gettempdir()) / "osyllabi"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_output_directory() -> Path:
    """
    Get the output directory, creating it if necessary.
    
    Returns:
        Path: Path to output directory
    """
    from osyllabi.config import OUTPUT_DIR
    ensure_directory(OUTPUT_DIR)
    return OUTPUT_DIR


def create_unique_file_path(base_dir: Union[str, Path], name: str, extension: str = "md") -> Path:
    """
    Create a unique file path that doesn't overwrite existing files.
    
    Args:
        base_dir: Base directory for the file
        name: Base name for the file
        extension: File extension (without dot)
        
    Returns:
        Path: Unique file path
    """
    # Sanitize the name and ensure extension doesn't have leading dot
    safe_name = sanitize_filename(name)
    ext = extension.lstrip('.')
    
    dir_path = ensure_directory(base_dir)
    counter = 0
    
    # Start with the base name
    file_path = dir_path / f"{safe_name}.{ext}"
    
    # If file exists, append a counter until we find a unique name
    while file_path.exists():
        counter += 1
        file_path = dir_path / f"{safe_name}_{counter}.{ext}"
    
    return file_path


def is_valid_source_file(path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid source file based on extension.
    
    Args:
        path: Path to file
        
    Returns:
        bool: True if valid, False otherwise
    """
    file_path = Path(path)
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS


def find_source_files(
    source_paths: List[Union[str, Path]], 
    recursive: bool = True
) -> List[Path]:
    """
    Find all source files from a list of directories and files.
    
    Args:
        source_paths: List of source paths (files or directories)
        recursive: Whether to search directories recursively
        
    Returns:
        List[Path]: List of source file paths
    """
    result = []
    
    for source in source_paths:
        path = Path(source)
        
        # If path is a file and has supported extension, add it
        if path.is_file() and is_valid_source_file(path):
            result.append(path)
            
        # If path is a directory, search for files
        elif path.is_dir():
            if recursive:
                # Use recursive glob to find all files
                for ext in SUPPORTED_FILE_EXTENSIONS:
                    result.extend([p for p in path.glob(f"**/*{ext}") if p.is_file()])
            else:
                # Only search immediate directory
                for ext in SUPPORTED_FILE_EXTENSIONS:
                    result.extend([p for p in path.glob(f"*{ext}") if p.is_file()])
                    
    return result
