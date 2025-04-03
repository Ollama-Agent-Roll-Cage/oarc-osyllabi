#!/usr/bin/env python
"""
Main entry point for Osyllabi application.
"""
import sys
from osyllabi.utils.cli.router import handle


def main():
    """Main entry point function for the CLI application."""
    return handle()


if __name__ == "__main__":
    sys.exit(main())
