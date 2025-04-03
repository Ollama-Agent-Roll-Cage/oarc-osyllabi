"""
Centralized argument parser for CLI commands.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from osyllabi.config import EXPORT_FORMATS, DEFAULT_EXPORT_FORMAT
from osyllabi.utils.cli.help import show_help

def validate_path_arg(path: str, must_exist: bool = False) -> str:
    """
    Validate a file or directory path argument.
    
    Args:
        path: The path to validate
        must_exist: Whether the path must already exist
        
    Returns:
        str: The validated path string
        
    Raises:
        argparse.ArgumentTypeError: If path validation fails
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
        
    return str(path_obj)


def validate_url(url: str) -> str:
    """
    Validate a URL argument.
    
    Args:
        url: The URL to validate
        
    Returns:
        str: The validated URL
        
    Raises:
        argparse.ArgumentTypeError: If URL validation fails
    """
    # Very basic URL validation
    if not url.startswith(('http://', 'https://')):
        raise argparse.ArgumentTypeError(f"Invalid URL format (must start with http:// or https://): {url}")
    return url


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command line argument parser with all available commands.
    """
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            # Don't exit, just show help
            self.print_usage(sys.stderr)
            print(f"{self.prog}: error: {message}\n", file=sys.stderr)
            sys.exit(2)

    parser = CustomArgumentParser(
        description="Osyllabi: Create and manage personalized curriculums",
        add_help=False  # Disable built-in help to use our custom help
    )
    # Add global arguments
    parser.add_argument('--help', '-h', action='store_true', 
                        help="Show this help message", dest='help_requested')
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for detailed logging and error information", 
                        dest='debug_mode')
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add help command
    help_parser = subparsers.add_parser('help', add_help=False, 
                                       help="Display help for Osyllabi commands")
    help_parser.add_argument('subcommand', nargs='?', help="Command to get help for")
    help_parser.add_argument('--help', '-h', action='store_true', 
                          help="Show help for the help command", dest='help_requested')
    help_parser.add_argument('--debug', action='store_true',
                          help="Enable debug mode", dest='debug_mode')
    
    # Register other commands
    from osyllabi.utils.cli.cmd_types import CommandType
    from osyllabi.utils.cli.commands import CleanCommand, CreateCommand
    
    # Create subparsers for other commands
    create_parser = subparsers.add_parser(CommandType.CREATE.value, add_help=False,
                                         help="Create a new curriculum")
    create_parser.add_argument('--help', '-h', action='store_true', 
                            help="Show help for create command", dest='help_requested')
    create_parser.add_argument('--debug', action='store_true',
                            help="Enable debug mode", dest='debug_mode')
    
    clean_parser = subparsers.add_parser(CommandType.CLEAN.value, add_help=False,
                                        help="Clean up generated files and temporary data")
    clean_parser.add_argument('--help', '-h', action='store_true', 
                           help="Show help for clean command", dest='help_requested')
    clean_parser.add_argument('--debug', action='store_true',
                           help="Enable debug mode", dest='debug_mode')
    
    # Register arguments for each command
    CleanCommand.register(clean_parser)
    CreateCommand.register(create_parser)
    
    return parser


def setup_create_arguments(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the create command."""
    parser.add_argument(
        "topic",
        metavar="TOPIC",
        help="Main topic or subject for the curriculum"
    )
    parser.add_argument(
        "--title", "-t",
        help="Custom title for the curriculum (default: '<topic> Curriculum')"
    )
    parser.add_argument(
        "--level", "-s", 
        default="Beginner",
        choices=["Beginner", "Intermediate", "Advanced", "Expert"],
        help="Target skill level"
    )
    parser.add_argument(
        "--links", "-l", 
        nargs="+", 
        default=[],
        type=validate_url,
        help="URLs to include as resources"
    )
    parser.add_argument(
        "--source", 
        nargs="+", 
        default=["."],
        type=lambda x: validate_path_arg(x, must_exist=True),
        help="Source files or directories to include"
    )
    parser.add_argument(
        "--export-path", "-o",
        type=validate_path_arg,
        help="Directory to export the curriculum"
    )
    parser.add_argument(
        "--format", "-f",
        default=DEFAULT_EXPORT_FORMAT,
        choices=EXPORT_FORMATS,
        help="Output format"
    )


def setup_clean_arguments(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the clean command."""
    parser.add_argument(
        "--output-dir",
        type=validate_path_arg,
        help="Clean specific output directory"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Clean all generated files"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Clean only cached files"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force clean without confirmation"
    )


def setup_help_arguments(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the help command."""
    parser.add_argument(
        "subcommand",
        nargs="?",
        help="Command to get help for"
    )


def parse_args(args: Optional[List[str]] = None) -> Union[argparse.Namespace, Tuple[bool, Optional[str]]]:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments to parse, defaults to sys.argv[1:]
        
    Returns:
        Either parsed arguments or a tuple indicating help was requested
    """
    if args is None:
        args = sys.argv[1:]
        
    # Check for global help flags first (as a special case)
    if not args or args[0] in ['-h', '--help']:
        return (True, None)  # Help requested with no specific command
        
    parser = setup_parser()
    
    try:
        parsed_args = parser.parse_args(args)
        
        # If help is requested for a command, we want to handle it
        if hasattr(parsed_args, 'help_requested') and parsed_args.help_requested:
            return (True, parsed_args.command)
            
        return parsed_args
    except SystemExit as e:
        # On parse error, check if it looks like a help request
        cmd = None
        if args and args[0] not in ['-h', '--help'] and not args[0].startswith('-'):
            cmd = args[0]
            
        return (True, cmd)
