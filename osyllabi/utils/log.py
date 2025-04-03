"""
Centralized logging functionality for Osyllabi.

This module provides a ContextAwareLogger class that adds context information
to log messages, making them more useful for debugging and monitoring.
"""
import os
import sys
import logging
import threading
from typing import Optional
from pathlib import Path

from osyllabi.utils.const import DEFAULT_LOG_FORMAT

# Log levels as constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Thread local storage for context
_thread_local = threading.local()


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled via environment variable.
    
    Returns:
        bool: True if debug mode is enabled
    """
    return os.environ.get('OSYLLABI_DEBUG', '').lower() in ('1', 'true', 'yes')


class ContextAwareFilter(logging.Filter):
    """Filter that adds context information to log records."""
    
    def filter(self, record):
        """Add context information to the record."""
        # Get context from thread local storage or use empty dict
        context = getattr(_thread_local, 'context', {})
        
        # Format context as string - add a leading hyphen when context exists
        if context:
            record.context = f" - {', '.join(f'{k}={v}' for k, v in context.items())}"
        else:
            record.context = ""
        return True


class ContextAwareLogger(logging.Logger):
    """Logger that supports context information for each log message."""
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the logger with the given name and level."""
        super().__init__(name, level)
        self.addFilter(ContextAwareFilter())
        
    def set_context(self, **kwargs) -> None:
        """
        Set context values for the current thread.
        
        Args:
            **kwargs: Key-value pairs to add to the context
        """
        if not hasattr(_thread_local, 'context'):
            _thread_local.context = {}
            
        _thread_local.context.update(kwargs)
        
    def clear_context(self) -> None:
        """Clear all context values for the current thread."""
        if hasattr(_thread_local, 'context'):
            _thread_local.context.clear()
            
    def with_context(self, **kwargs):
        """
        Context manager for temporary context values.
        
        Args:
            **kwargs: Context values to set temporarily
            
        Returns:
            Context manager that restores previous context on exit
        """
        class ContextManager:
            def __init__(self, logger):
                self.logger = logger
                self.previous = {}
                
            def __enter__(self):
                # Save the current context
                if hasattr(_thread_local, 'context'):
                    self.previous = _thread_local.context.copy()
                else:
                    self.previous = {}
                
                # Set the new context
                self.logger.set_context(**kwargs)
                return self.logger
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if hasattr(_thread_local, 'context'):
                    _thread_local.context.clear()
                    if self.previous:
                        _thread_local.context.update(self.previous)
                
        return ContextManager(self)


def setup_logger(
    name: str = 'osyllabi',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT
) -> ContextAwareLogger:
    """
    Set up a context-aware logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path to write logs to
        log_format: Format string for log messages
        
    Returns:
        ContextAwareLogger: Configured logger instance
    """
    # Register the ContextAwareLogger class
    logging.setLoggerClass(ContextAwareLogger)
    
    # Get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create the singleton logger instance
log = setup_logger(
    level=DEBUG if is_debug_mode() else INFO
)

# Add module-level functions that delegate to the singleton
def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    log.debug(msg, *args, **kwargs)
    
def info(msg: str, *args, **kwargs):
    """Log an info message."""
    log.info(msg, *args, **kwargs)
    
def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    log.warning(msg, *args, **kwargs)

def warn(msg: str, *args, **kwargs):
    """Log a warning message (alias for warning)."""
    log.warning(msg, *args, **kwargs)
    
def error(msg: str, *args, **kwargs):
    """Log an error message."""
    log.error(msg, *args, **kwargs)
    
def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    log.critical(msg, *args, **kwargs)
    
def set_context(**kwargs):
    """Set context values for the current thread."""
    log.set_context(**kwargs)
    
def clear_context():
    """Clear all context values for the current thread."""
    log.clear_context()
    
def with_context(**kwargs):
    """Context manager for temporary context values."""
    return log.with_context(**kwargs)
