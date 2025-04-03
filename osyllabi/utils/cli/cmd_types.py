"""
Command types for the CLI.
"""
from enum import Enum


class CommandType(Enum):
    """Available command types."""
    CREATE = "create"
    CLEAN = "clean"
