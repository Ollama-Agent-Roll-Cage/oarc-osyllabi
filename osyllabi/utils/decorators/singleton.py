"""
Singleton decorator for ensuring only one instance of a class exists.
"""
import functools
from typing import Any, Dict, Type, TypeVar, cast

from osyllabi.utils.log import log

# Type variable for generic typing
T = TypeVar('T')

# Registry of singleton instances
_instances: Dict[Type, Any] = {}


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator that makes a class follow the singleton pattern.
    
    This decorator ensures only one instance of a class exists. If the class
    is instantiated again, the existing instance is returned instead.
    
    Args:
        cls: The class to make a singleton
        
    Returns:
        Decorated class that follows the singleton pattern
    """
    original_new = cls.__new__
    original_init = cls.__init__
    
    @functools.wraps(original_new)
    def __new__(cls, *args, **kwargs):
        if cls not in _instances:
            instance = original_new(cls)
            _instances[cls] = instance
            return instance
        return _instances[cls]
    
    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        if not hasattr(self, '_initialized'):
            original_init(self, *args, **kwargs)
            self._initialized = True
        elif args or kwargs:
            # Only log when parameters are provided and different from initial call
            name_str = getattr(self, 'name', '')
            signature = f"{self.__class__.__name__}({name_str})"
            if self.__class__.__name__ != "PromptTemplate":
                log.debug(f"Reusing existing singleton instance of {signature}")
    
    cls.__new__ = __new__
    cls.__init__ = __init__
    
    return cls
