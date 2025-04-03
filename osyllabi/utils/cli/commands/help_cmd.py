"""
Help command for displaying usage information.
"""
import argparse
from typing import Optional

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_help_arguments
from osyllabi.utils.cli.help import show_help
from osyllabi.utils.const import SUCCESS


class HelpCommand(Command):
    """Command for displaying help information."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        setup_help_arguments(parser)
    
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command and return the exit code.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code
        """
        # Show help for the requested command or general help
        subcommand = args.subcommand if hasattr(args, 'subcommand') else None
        show_help(subcommand)
        return SUCCESS
