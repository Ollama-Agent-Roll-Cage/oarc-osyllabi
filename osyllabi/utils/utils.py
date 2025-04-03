"""
General utility functions for the Osyllabi project.
"""
import os
import sys
import platform
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Union, TypeVar, Callable, Tuple, Optional

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


def detect_gpu() -> Tuple[bool, Optional[str]]:
    """
    Detect if a CUDA-capable GPU is available on the system without requiring ML frameworks.
    
    Returns:
        Tuple[bool, Optional[str]]: (GPU available, GPU info string)
    """
    # Try using nvidia-smi command (most direct method)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse the output to get GPU info
            gpu_info = f"NVIDIA GPU: {result.stdout.strip()}"
            return True, gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check for CUDA availability using system info (Windows-specific)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ["where", "nvcc"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, "CUDA toolkit detected via nvcc"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check Windows registry for NVIDIA drivers
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\global")
            return True, "NVIDIA drivers detected in Windows registry"
        except (ImportError, OSError, WindowsError):
            pass
    
    # Linux-specific checks
    elif sys.platform.startswith('linux'):
        # Check for /proc entries that indicate NVIDIA GPU
        gpu_devices = Path('/proc/driver/nvidia/gpus')
        if gpu_devices.exists() and any(gpu_devices.iterdir()):
            return True, "NVIDIA GPU detected via /proc/driver/nvidia"
            
        # Check loaded kernel modules
        try:
            result = subprocess.run(
                ["lsmod"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if "nvidia" in result.stdout:
                return True, "NVIDIA kernel modules detected"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # No GPU detected
    return False, None


def upgrade_faiss_to_gpu() -> Tuple[bool, str]:
    """
    Attempt to upgrade faiss-cpu to faiss-gpu if a GPU is available.
    
    Returns:
        Tuple[bool, str]: (Success, Message) - whether upgrade was successful and status message
    """
    # Check if GPU is available
    has_gpu, gpu_info = detect_gpu()
    
    if not has_gpu:
        return False, "No GPU detected, keeping faiss-cpu"
    
    log.info(f"GPU detected: {gpu_info}")
    
    # Check if faiss is installed and which version
    try:
        import faiss
        is_cpu_version = not hasattr(faiss, 'StandardGpuResources')
        
        if not is_cpu_version:
            return True, "faiss-gpu is already installed"
        
        log.info("faiss-cpu detected, upgrading to faiss-gpu")
    except ImportError:
        log.info("faiss not installed, attempting to install faiss-gpu")
        is_cpu_version = False
    
    # Try to install faiss-gpu
    try:
        # Uninstall faiss-cpu if it's installed
        if is_cpu_version:
            log.info("Uninstalling faiss-cpu")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        
        # Install faiss-gpu
        log.info("Installing faiss-gpu")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "faiss-gpu>=1.7.2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return True, "Successfully upgraded to faiss-gpu"
        else:
            error_msg = f"Failed to install faiss-gpu: {result.stderr}"
            log.warning(error_msg)
            
            # If faiss-cpu was uninstalled, attempt to reinstall it
            if is_cpu_version:
                log.info("Reinstalling faiss-cpu")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "faiss-cpu>=1.7.0"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            return False, error_msg
    
    except Exception as e:
        error_msg = f"Error during faiss upgrade: {str(e)}"
        log.error(error_msg)
        return False, error_msg


def check_faiss_gpu_capability() -> bool:
    """
    Check if FAISS has GPU support enabled.
    
    Returns:
        bool: True if faiss-gpu is available and working
    """
    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            # Try to create a GPU resource to verify it actually works
            try:
                res = faiss.StandardGpuResources()
                # Create a small test index
                index = faiss.IndexFlatL2(128)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                log.info("FAISS GPU capability verified")
                return True
            except Exception as e:
                log.warning(f"FAISS has GPU support but initialization failed: {e}")
                return False
        return False
    except ImportError:
        return False

