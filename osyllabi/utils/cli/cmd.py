"""
Base command class definition.

This module provides the abstract base class for command implementations.
"""
import abc
import argparse
from typing import Optional, Any


class Command(abc.ABC):
    """
    Abstract base class for command implementations.
    
    All command handlers should extend this class and implement the execute method.
    """
    
    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        Initialize the command.
        
        Args:
            args: Optional command-line arguments
        """
        self.args = args
    
    @abc.abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code
        """
        pass
