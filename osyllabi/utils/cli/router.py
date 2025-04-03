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
from osyllabi.utils.cli.help import show_help, is_help_requested, display_help_for_unknown_command
from osyllabi.utils.const import FAILURE, SUCCESS
from osyllabi.utils.log import setup_logger, DEBUG, INFO


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
            # Reconfigure logger with debug level
            from osyllabi.utils.log import log
            log.setLevel(DEBUG)
            print("Debug mode enabled.", file=sys.stderr)
        
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
            print(f"Command {args.command} completed with exit code: {exit_code}", file=sys.stderr)
            
        return exit_code
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return FAILURE
        
    except Exception as e:
        # Provide more detailed error information to help with debugging
        print(f"Error: {e}", file=sys.stderr)
        
        # Check for debug mode environment variable or command line flag
        if os.environ.get('OSYLLABI_DEBUG', '').lower() in ('1', 'true', 'yes'):
            print("\nDetailed traceback:", file=sys.stderr)
            traceback.print_exc()
        else:
            print("\nRun with --debug flag or set environment variable OSYLLABI_DEBUG=1 for detailed error information.", file=sys.stderr)
            
        return FAILURE
