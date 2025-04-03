"""
Retry decorator for automatic retrying of functions.
"""
import time
from typing import TypeVar, Callable, Tuple
from functools import wraps

from osyllabi.utils.log import log

# Type variable for generic typing
T = TypeVar('T')


def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: Tuple = (Exception,)):
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
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            local_delay = delay
            last_exception = None
            func_name = getattr(func, '__name__', 'unknown_function')
            
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    log.warning(f"Attempt {attempt}/{attempts} for {func_name} failed: {str(e)}")
                    
                    if attempt < attempts:
                        log.debug(f"Retrying in {local_delay:.2f} seconds...")
                        time.sleep(local_delay)
                        local_delay *= backoff
            
            # If we get here, all attempts failed
            log.error(f"All {attempts} attempts for {func_name} failed")
            
            # Re-raise the last exception
            raise last_exception
            
        return wrapper
    return decorator
