"""
Help command for displaying usage information.
"""
import argparse

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.help import show_help


class HelpCommand(Command):
    """Command for displaying help and usage information."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        parser.add_argument('subcommand', nargs='?', help="Command to get help for")
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        subcommand = getattr(self.args, 'subcommand', None)
        show_help(subcommand)
        return 0
