#!/usr/bin/env python
"""
Main entry point for Osyllabi application.
"""
import sys
from typing import Optional, List

from osyllabi.utils.log import is_debug_mode
from osyllabi.utils.cli.router import route_command


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point function for the CLI application.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    try:
        # Use the route_command function directly
        exit_code = route_command(args)
        return exit_code
    except RuntimeError as e:
        if "Ollama" in str(e):
            print(f"Error: {e}")
            print("Osyllabi requires Ollama to be installed and running.")
            print("Please visit https://ollama.ai/download for installation instructions.")
        else:
            print(f"Runtime error: {e}")
        return 1
    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        print("\nCancelled by user.", file=sys.stderr)
        return 130            # Standard exit code for SIGINT
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)
        
        # Provide detailed error information if in debug mode
        if is_debug_mode():
            import traceback
            traceback.print_exc()
            
        return 1


if __name__ == "__main__":
    sys.exit(main())
