"""
Clean command for maintenance tasks.
"""
import argparse

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_clean_arguments
from osyllabi.utils.cli.clean import clean_from_args


class CleanCommand(Command):
    """Command for cleaning up generated files and temporary data."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        setup_clean_arguments(parser)
    
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command and return the exit code.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code
        """
        # Delegate to clean utility with parsed arguments
        return clean_from_args(args)
