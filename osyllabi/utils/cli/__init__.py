"""
Command-line interface utilities for Osyllabi.
"""

from osyllabi.utils.cli.router import handle, route_command
from osyllabi.utils.cli.help import show_help
from osyllabi.utils.cli.cmd_types import CommandType
from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.cmd_descriptions import COMMAND_DESCRIPTIONS

__all__ = [
    "handle", 
    "route_command", 
    "show_help", 
    "CommandType", 
    "Command", 
    "COMMAND_DESCRIPTIONS"
]
