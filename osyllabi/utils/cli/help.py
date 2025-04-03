"""
Help utilities for CLI commands.
"""
import sys
import textwrap
from typing import Dict, Optional, List


def display_header() -> None:
    """Display the application header."""
    print("\nOsyllabi - A Python-powered curriculum designer")
    print("Version: 0.1.0\n")
    print("Generates personalized curriculums using AI, web crawling, and data integration.\n")


def display_command_help(command_name: str, command_descriptions: Dict[str, str]) -> None:
    """Display help for a specific command."""
    if command_name not in command_descriptions:
        print(f"\nUnknown command: {command_name}", file=sys.stderr)
        print("\nRun 'osyllabi help' to see available commands.")
        return
        
    description = command_descriptions[command_name]
    
    print(f"\nOSYLLABI {command_name.upper()}")
    print(f"\n{description}\n")
    
    # Command-specific details with more netstat-like formatting
    if command_name == "create":
        print("Usage: osyllabi create TOPIC [options]")
        print("\nOptions:")
        print("  --title, -t        Title of the curriculum")
        print("  --level, -s        Target skill level (Beginner, Intermediate, Advanced, Expert)")
        print("  --links, -l        URLs to include as resources")
        print("  --source           Source files or directories to include")
        print("                     Default: current directory")
        print("  --export-path, -o  Directory to export the curriculum")
        print("  --format, -f       Output format (md, pdf, html, docx)")
        print("  --help, -h         Display this help message")
        
    elif command_name == "clean":
        print("Usage: osyllabi clean [options]")
        print("\nOptions:")
        print("  --output-dir       Clean specific output directory")
        print("  --all, -a          Clean all generated files")
        print("  --cache            Clean only cached files")
        print("  --force, -f        Force clean without confirmation")
        print("  --help, -h         Display this help message")
        
    elif command_name == "help":
        print("Usage: osyllabi help [command]")
        print("\nArguments:")
        print("  command            Command to get help for")
        print("\nOptions:")
        print("  --help, -h         Display this help message")
    
    print("\nExamples:")
    if command_name == "create":
        print('  osyllabi create "Machine Learning" --title "ML Fundamentals" --level Beginner')
        print('  osyllabi create "Python Programming" --links "https://docs.python.org" --format pdf')
    elif command_name == "clean":
        print('  osyllabi clean --all')
        print('  osyllabi clean --output-dir "./output" --force')
    elif command_name == "help":
        print('  osyllabi help')
        print('  osyllabi help create')


def display_general_help(command_descriptions: Dict[str, str]) -> None:
    """Display general help for all commands."""
    display_header()
    
    print("Usage: osyllabi COMMAND [options]")
    
    print("\nCommands:")
    max_len = max(len(name) for name in command_descriptions.keys())
    for cmd_name, description in sorted(command_descriptions.items()):
        print(f"  {cmd_name.ljust(max_len)}    {description}")
    
    print("\nOptions:")
    print("  --help, -h         Display help for a command")
    
    print("\nRun 'osyllabi help <command>' for more information on a specific command.")


def display_epilog() -> None:
    """Display epilog information after help text."""
    epilog = textwrap.dedent("""\
        For more information and examples, visit:
        https://github.com/p3nGu1nZz/osyllabi
        
        Report issues at:
        https://github.com/p3nGu1nZz/osyllabi/issues
    """)
    print("\n" + epilog)


def display_help_for_unknown_command(attempted_command: str) -> None:
    """Display help message for an unknown command."""
    print(f"\nUnknown command: '{attempted_command}'")
    print("\nRun 'osyllabi help' to see available commands.")


def show_help(command: Optional[str] = None) -> None:
    """
    Display help information for commands.
    
    Args:
        command: Specific command to show help for, or None for general help
    """
    from osyllabi.utils.cli.router import COMMAND_DESCRIPTIONS
    
    if command:
        display_command_help(command, COMMAND_DESCRIPTIONS)
    else:
        display_general_help(COMMAND_DESCRIPTIONS)
    
    display_epilog()


def is_help_requested(args_list: Optional[List[str]] = None) -> bool:
    """
    Check if help is requested based on command line arguments.
    
    Args:
        args_list: List of command line arguments, uses sys.argv if None
        
    Returns:
        bool: True if help is requested, False otherwise
    """
    if args_list is None:
        args_list = sys.argv[1:]
        
    return '-h' in args_list or '--help' in args_list or 'help' in args_list
