"""
General utility functions for the Osyllabi project.
"""
import os
import sys
import platform
import tempfile

from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Union, TypeVar, Callable

from osyllabi.utils.log import log

# Type variable for generic functions
T = TypeVar('T')


def safe_to_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        int: Converted integer or default value
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_to_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted float or default value
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def find_files_by_extensions(
    root_dir: Union[str, Path], 
    extensions: List[str],
    skip_hidden: bool = True
) -> List[Path]:
    """
    Find all files with specific extensions under a directory.
    
    Args:
        root_dir: Root directory to search in
        extensions: List of file extensions to include (with dot)
        skip_hidden: Whether to skip hidden files and directories
        
    Returns:
        List[Path]: List of found file paths
    """
    root_path = Path(root_dir)
    result = []
    
    if not root_path.exists():
        return result
    
    try:
        for path in root_path.rglob('*'):
            # Skip hidden items if requested
            if skip_hidden and any(p.startswith('.') for p in path.parts):
                continue
                
            if path.is_file() and path.suffix.lower() in extensions:
                result.append(path)
    except PermissionError:
        # Handle permission errors gracefully
        log.warning(f"Permission denied when accessing {root_path}")
        pass
                
    return result


def get_app_dirs() -> Dict[str, Path]:
    """
    Get application directories for configs, cache, and data.
    
    Returns:
        Dict[str, Path]: Dictionary of directory paths
    """
    # Platform-specific config locations
    app_name = 'osyllabi'
    
    if sys.platform == 'win32':
        app_data = Path(os.environ.get('APPDATA', ''))
        config_dir = app_data / app_name
        cache_dir = Path(os.environ.get('LOCALAPPDATA', '')) / app_name / 'cache'
        data_dir = app_data / app_name / 'data'
    elif sys.platform == 'darwin':
        home = Path.home()
        config_dir = home / 'Library' / 'Application Support' / app_name
        cache_dir = home / 'Library' / 'Caches' / app_name
        data_dir = home / 'Library' / 'Application Support' / app_name / 'data'
    else:
        # Linux and other Unix-like systems
        home = Path.home()
        config_dir = home / '.config' / app_name
        cache_dir = home / '.cache' / app_name
        data_dir = home / '.local' / 'share' / app_name
    
    # Create directories if they don't exist
    for directory in [config_dir, cache_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        'config': config_dir,
        'cache': cache_dir,
        'data': data_dir,
        'temp': Path(tempfile.gettempdir()) / app_name
    }


def get_system_info() -> Dict[str, str]:
    """
    Get system information for diagnostics.
    
    Returns:
        Dict[str, str]: System information
    """
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'node': platform.node()
    }


def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """
    Retry decorator with exponential backoff for functions.
    
    Args:
        attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function decorator
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            nonlocal delay
            last_exception = None
            func_name = getattr(func, '__name__', 'unknown_function')
            
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    log.warning(f"Attempt {attempt}/{attempts} for {func_name} failed: {str(e)}")
                    
                    if attempt < attempts:
                        import time
                        time.sleep(delay)
                        delay *= backoff
            
            # If we get here, all attempts failed
            log.error(f"All {attempts} attempts for {func_name} failed")
            
            # Re-raise the last exception
            raise last_exception
            
        return wrapper
    return decorator


def get_timestamp(format_str: str = '%Y%m%d_%H%M%S') -> str:
    """
    Get a formatted timestamp string.
    
    Args:
        format_str: Format string for datetime.strftime
        
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Trim spaces and limit length
    filename = filename.strip()
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext)]
        filename = name + ext
        
    return filename
