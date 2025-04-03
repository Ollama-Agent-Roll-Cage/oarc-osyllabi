#!/usr/bin/env python
"""
Main entry point for Osyllabi application.
"""
import sys
import os
import argparse
from typing import Optional, List

from osyllabi.utils.utils import setup_logging, is_debug_mode
from osyllabi.utils.cli.router import handle


def configure_environment() -> None:
    """Configure the application environment."""
    # Set up logging based on environment
    log_level = "DEBUG" if is_debug_mode() else "INFO"
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=numeric_level)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point function for the CLI application.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    try:
        # Configure environment
        configure_environment()
        
        # Handle CLI commands
        return handle(args)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        # Last-resort exception handler
        print(f"Unhandled error: {e}", file=sys.stderr)
        
        if is_debug_mode():
            import traceback
            traceback.print_exc()
            
        return 1


if __name__ == "__main__":
    sys.exit(main())
