"""
Factory pattern implementation as a class decorator.

This module provides a simple decorator to add factory pattern functionality to classes.
"""
from typing import Any, Type, TypeVar, Dict, Optional, Callable

# Type variable for generic typing
T = TypeVar('T')


def factory(cls: Type[T]) -> Type[T]:
    """
    Class decorator that adds a factory method to the decorated class.
    
    This decorator adds a 'create' class method to the decorated class
    that creates and returns new instances of the class.
    
    Example:
        @factory
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
                
        # Now you can create instances using the class method:
        john = Person.create(name="John", age=30)
    """
    # Define the factory method to be added
    @classmethod
    def create(cls_method, *args: Any, **kwargs: Any) -> Any:
        """
        Create and return a new instance of the class.
        
        Args:
            *args: Positional arguments to pass to the class constructor
            **kwargs: Keyword arguments to pass to the class constructor
            
        Returns:
            A new instance of the class or the result of a special processing method
        """
        # Special handling for the 'args' parameter which is expected by _process_args
        if 'args' in kwargs and len(args) == 0:
            instance = cls(args=kwargs['args'])
            # If instance has a _result attribute, return that instead
            if hasattr(instance, '_result') and instance._result is not None:
                return instance._result
            return instance
            
        instance = cls(*args, **kwargs)
        
        # If instance has a result to return (special case for CLI args processing),
        # return that directly
        if hasattr(instance, '_result') and instance._result is not None:
            return instance._result
            
        return instance
    
    # Add the method to the class
    setattr(cls, 'create', create)
    
    return cls
