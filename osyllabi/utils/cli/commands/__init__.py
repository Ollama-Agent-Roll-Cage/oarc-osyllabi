"""
Command handlers for CLI commands.

This package provides command handler implementations for the various CLI commands.
"""
from typing import Dict, Optional, Type

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.log import log


# Dictionary to hold registered command handlers
_command_handlers: Dict[str, Type[Command]] = {}


def register_command(name: str, handler_class: Type[Command]) -> None:
    """
    Register a command handler for a command name.
    
    Args:
        name: Command name
        handler_class: Command handler class
    """
    _command_handlers[name] = handler_class
    log.debug(f"Registered command handler for '{name}': {handler_class.__name__}")


def get_command_handler(name: str) -> Optional[Command]:
    """
    Get a command handler instance for a command name.
    
    Args:
        name: Command name
        
    Returns:
        Command: Command handler instance or None if not found
    """
    handler_class = _command_handlers.get(name)
    if handler_class:
        # Create instance without passing args (they'll be passed in execute())
        return handler_class(None)  # Pass None explicitly for clarity
    return None


# Import and register all command handlers
from osyllabi.utils.cli.commands.create_cmd import CreateCommand
from osyllabi.utils.cli.commands.help_cmd import HelpCommand
from osyllabi.utils.cli.commands.clean_cmd import CleanCommand

# Register commands
register_command("create", CreateCommand)
register_command("help", HelpCommand)
register_command("clean", CleanCommand)
