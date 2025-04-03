"""
Centralized argument parser for CLI commands.
"""
import argparse
import sys

from osyllabi.config import EXPORT_FORMATS, DEFAULT_EXPORT_FORMAT


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command line argument parser with all available commands.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Osyllabi: Create and manage personalized curriculums",
        add_help=False  # Disable built-in help to use our custom help
    )
    parser.add_argument('--help', '-h', action='store_true', 
                        help="Show this help message", dest='help_requested')
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    help_parser = subparsers.add_parser('help', add_help=False, 
                                       help="Display help for Osyllabi commands")
    help_parser.add_argument('subcommand', nargs='?', help="Command to get help for")
    help_parser.add_argument('--help', '-h', action='store_true', 
                          help="Show help for the help command", dest='help_requested')
    
    # Register other commands
    from osyllabi.utils.cli.cmd_types import CommandType
    from osyllabi.utils.cli.commands import CleanCommand, CreateCommand
    
    # Create subparsers for other commands
    create_parser = subparsers.add_parser(CommandType.CREATE.value, add_help=False,
                                         help="Create a new curriculum")
    create_parser.add_argument('--help', '-h', action='store_true', 
                            help="Show help for create command", dest='help_requested')
    
    clean_parser = subparsers.add_parser(CommandType.CLEAN.value, add_help=False,
                                        help="Clean up generated files and temporary data")
    clean_parser.add_argument('--help', '-h', action='store_true', 
                           help="Show help for clean command", dest='help_requested')
    
    # Register arguments for each command
    CleanCommand.register(clean_parser)
    CreateCommand.register(create_parser)
    
    return parser


def setup_create_arguments(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the create command."""
    parser.add_argument("topic", help="Main topic for the curriculum")
    parser.add_argument("--title", "-t", help="Title of the curriculum")
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
        help="URLs to include as resources"
    )
    parser.add_argument(
        "--source", 
        nargs="+", 
        default=["."],
        help="Source files or directories to include"
    )
    parser.add_argument(
        "--export-path", "-o",
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


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = setup_parser()
    
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace()
        args.help_requested = True
        args.command = None
    
    return args
