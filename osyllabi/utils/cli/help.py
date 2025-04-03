"""
Help utilities for CLI commands.
"""
import sys
import textwrap
from typing import Dict, Optional, Type

from osyllabi.utils.cli.cmd import Command


def display_command_help(command_name: str, command_map: Dict[str, Type[Command]]) -> None:
    """Display help for a specific command."""
    if command_name in command_map:
        cmd_class = command_map[command_name]
        print(f"\nCommand: {command_name}")
        print(f"  {cmd_class.help_text()}")
    else:
        print(f"\nUnknown command: {command_name}", file=sys.stderr)


def display_general_help(command_map: Dict[str, Type[Command]]) -> None:
    """Display general help for all commands."""
    print("\nAvailable commands:")
    for cmd_name, cmd_class in sorted(command_map.items()):
        print(f"  {cmd_name:<10} - {cmd_class.help_text()}")
    
    print("\nUse 'osyllabi <command> --help' for more information about a specific command.")


def display_epilog() -> None:
    """Display epilog information after help text."""
    epilog = textwrap.dedent("""\
        
        Osyllabi is a tool for creating personalized curriculum materials.
        
        For more information and examples, visit:
        https://github.com/p3nGu1nZz/Osyllabi
        
        Report issues at:
        https://github.com/p3nGu1nZz/Osyllabi/issues
    """)
    print(epilog)


def show_help(command: Optional[str] = None, command_map: Optional[Dict[str, Type[Command]]] = None) -> None:
    """
    Display help information for commands.
    
    Args:
        command: Specific command to show help for, or None for general help
        command_map: Dictionary mapping command names to command classes
    """
    if command_map is None:
        from osyllabi.utils.cli.router import COMMAND_MAP
        command_map = COMMAND_MAP
    
    print("Osyllabi: A streamlined Python app for designing personalized curriculums")
    
    if command:
        display_command_help(command, command_map)
    else:
        display_general_help(command_map)
    
    display_epilog()
