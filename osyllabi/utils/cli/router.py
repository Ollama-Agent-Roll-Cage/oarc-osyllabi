"""
Command router for CLI operations.
"""
import os
import sys
import traceback
from typing import Type, Optional, List

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.cmd_types import CommandType
from osyllabi.utils.cli.parser import parse_args
from osyllabi.utils.log import log
from osyllabi.utils.cli.help import show_help, display_help_for_unknown_command
from osyllabi.utils.const import FAILURE, SUCCESS
from osyllabi.utils.log import DEBUG


# Map of command types to their descriptions
COMMAND_DESCRIPTIONS = {
    CommandType.HELP.value: "Display help for Osyllabi commands",
    CommandType.CREATE.value: "Create a new curriculum",
    CommandType.CLEAN.value: "Clean up generated files and temporary data",
}


def get_command(command_type: str) -> Type[Command]:
    """
    Dynamically import and return the command class.
    
    Args:
        command_type: String identifier for the command
        
    Returns:
        Command class for the specified command type
        
    Raises:
        ValueError: If command type is unknown
    """
    if command_type == CommandType.CREATE.value:
        from osyllabi.utils.cli.commands.create_cmd import CreateCommand
        return CreateCommand
    elif command_type == CommandType.CLEAN.value:
        from osyllabi.utils.cli.commands.clean_cmd import CleanCommand
        return CleanCommand
    elif command_type == CommandType.HELP.value:
        from osyllabi.utils.cli.commands.help_cmd import HelpCommand
        return HelpCommand
    else:
        raise ValueError(f"Unknown command: {command_type}")


def handle(args_list: Optional[List[str]] = None) -> int:
    """
    Parse command line arguments and route to appropriate handler.
    
    Args:
        args_list: Command line arguments, defaults to sys.argv[1:]
        
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Use provided args or default to sys.argv[1:]
        if args_list is None:
            args_list = sys.argv[1:]
            
        # Parse arguments first
        args = parse_args(args_list)
        
        # Handle debug mode
        if hasattr(args, 'debug_mode') and args.debug_mode:
            # Set environment variable for other parts of the application
            os.environ['OSYLLABI_DEBUG'] = '1'
            log.setLevel(DEBUG)
            log.info("Debug mode enabled.")
        
        # Handle help requests consistently
        if args.help_requested or (args.command == 'help' and not getattr(args, 'subcommand', None)):
            show_help()
            return SUCCESS
            
        # If it's "help <command>" or "<command> --help"
        if args.command == 'help' and hasattr(args, 'subcommand') and args.subcommand:
            show_help(args.subcommand)
            return SUCCESS
        elif hasattr(args, 'help_requested') and args.help_requested and args.command:
            show_help(args.command)
            return SUCCESS
            
        # Handle unknown or missing command
        if not args.command:
            show_help()
            return SUCCESS
        
        if args.command not in COMMAND_DESCRIPTIONS:
            display_help_for_unknown_command(args.command)
            return FAILURE
            
        # Execute the requested command
        cmd_class = get_command(args.command)
        command = cmd_class(args)
        exit_code = command.execute()
        
        # If debug mode is enabled, show more information about the result
        if os.environ.get('OSYLLABI_DEBUG', '').lower() in ('1', 'true', 'yes'):
            log.debug(f"Command {args.command} completed with exit code: {exit_code}")
            
        return exit_code
    
    except KeyboardInterrupt:
        log.warning("\nOperation cancelled by user.")
        return FAILURE
        
    except Exception as e:
        # Provide more detailed error information to help with debugging
        log.error(f"Error: {e}")
        
        # Check for debug mode environment variable or command line flag
        if os.environ.get('OSYLLABI_DEBUG', '').lower() in ('1', 'true', 'yes'):
            log.error("Detailed traceback:")
            traceback.print_exc()
        else:
            log.info("Run with --debug flag or set environment variable OSYLLABI_DEBUG=1 for detailed error information.")
            
        return FAILURE
