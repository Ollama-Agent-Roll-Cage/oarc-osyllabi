"""
Command router for CLI operations.
"""
import sys
from typing import Dict, Type

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.cmd_types import CommandType
from osyllabi.utils.cli.commands.create_cmd import CreateCommand
from osyllabi.utils.cli.commands.clean_cmd import CleanCommand
from osyllabi.utils.cli.parser import parse_args
from osyllabi.utils.cli.help import show_help
from osyllabi.utils.const import SUCCESS, FAILURE


# Register all available commands
COMMAND_MAP: Dict[str, Type[Command]] = {
    CommandType.CREATE.value: CreateCommand,
    CommandType.CLEAN.value: CleanCommand,
}


def handle() -> int:
    """
    Parse command line arguments and route to appropriate handler.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    args = parse_args()
    
    if not args.command or args.command not in COMMAND_MAP:
        from osyllabi.utils.cli.parser import setup_parser
        setup_parser().print_help()
        show_help(command_map=COMMAND_MAP)
        return FAILURE
    
    cmd_class = COMMAND_MAP[args.command]
    
    # Execute the command
    try:
        command = cmd_class(args)
        return command.execute()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return FAILURE
