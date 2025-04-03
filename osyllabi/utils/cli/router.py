"""
Command router for CLI functionality.

This module handles routing of commands to appropriate command handlers.
"""
import sys
from typing import Optional, List

from osyllabi.utils.log import log
from osyllabi.utils.const import SUCCESS, FAILURE
from osyllabi.utils.cli.parser import parse_args
from osyllabi.utils.cli.commands import get_command_handler
from osyllabi.utils.cli.help import show_help


def route_command(args: Optional[List[str]] = None) -> int:
    """
    Route command to appropriate handler based on arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    # Use sys.argv if args not provided
    if args is None:
        args = sys.argv[1:]
    
    # Parse arguments
    parsed_args = parse_args(args)
    
    # Handle help request (special return type from parse_args)
    if isinstance(parsed_args, tuple) and len(parsed_args) == 2:
        help_requested, command = parsed_args
        if help_requested:
            show_help(command)
            return SUCCESS
    
    # If parsing failed completely
    if not parsed_args:
        log.error("Failed to parse command line arguments")
        return FAILURE
        
    # Get command handler for the specified command
    command = parsed_args.command
    handler = get_command_handler(command)
    
    if handler:
        # Execute command
        log.debug(f"Executing command: {command}")
        return handler.execute(parsed_args)
    else:
        log.error(f"Unknown command: {command}")
        return FAILURE


# Legacy name for backwards compatibility
def handle(args: Optional[List[str]] = None) -> int:
    """
    Legacy function name for route_command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    return route_command(args)
