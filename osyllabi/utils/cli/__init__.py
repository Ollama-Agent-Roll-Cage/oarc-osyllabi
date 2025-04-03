"""
Command-line interface utilities for Osyllabi.
"""

from osyllabi.utils.cli.router import handle, route_command
from osyllabi.utils.cli.help import show_help
from osyllabi.utils.cli.cmd_types import CommandType
from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.cmd_desc import COMMAND_DESC

__all__ = [
    "handle", 
    "route_command", 
    "show_help", 
    "CommandType", 
    "Command", 
    "COMMAND_DESC"
]
