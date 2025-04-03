"""
Base command class for all CLI commands.
"""
import abc
import argparse


class Command(abc.ABC):
    """Abstract base class for all commands."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize with parsed arguments."""
        self.args = args
    
    @classmethod
    @abc.abstractmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        pass
    
    @classmethod
    @abc.abstractmethod
    def help_text(cls) -> str:
        """Return help text for this command."""
        pass
    
    @abc.abstractmethod
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        pass
