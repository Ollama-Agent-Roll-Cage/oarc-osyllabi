#!/usr/bin/env python
"""
Main entry point for Osyllabi application.
"""
import sys
from osyllabi.utils.cli.router import handle


if __name__ == "__main__":
    sys.exit(handle())
