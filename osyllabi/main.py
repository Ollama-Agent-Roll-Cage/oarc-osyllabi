#!/usr/bin/env python
"""
Main entry point for Osyllabi application.
"""
import sys
from typing import Optional, List

from osyllabi.utils.log import log, is_debug_mode
from osyllabi.utils.cli.router import handle


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point function for the CLI application.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    try:
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
