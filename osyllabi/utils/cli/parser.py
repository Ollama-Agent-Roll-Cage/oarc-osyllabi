"""
Centralized argument parser for CLI commands.
"""
import argparse

from osyllabi.utils.cli.cmd_types import CommandType
from osyllabi.config import EXPORT_FORMATS, DEFAULT_EXPORT_FORMAT


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command line argument parser with all available commands.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Osyllabi: Create and manage personalized curriculums"
    )
    parser.add_subparsers(dest="command", help="Command to execute")
    return parser


def setup_create_arguments(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the create command."""
    parser.add_argument("topic", help="Main topic for the curriculum")
    parser.add_argument("--title", "-t", help="Title of the curriculum")
    parser.add_argument(
        "--skill-level", "-s", 
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
    return setup_parser().parse_args()
