"""
Command types for the CLI.
"""
from enum import Enum


class CommandType(Enum):
    """Available command types."""
    HELP = "help"
    CREATE = "create"
    CLEAN = "clean"
