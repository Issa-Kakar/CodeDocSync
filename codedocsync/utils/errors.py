"""
Custom exception classes for CodeDocSync.

This module defines the exception hierarchy used throughout the application
for consistent error handling and reporting.
"""

from typing import Optional


class ParsingError(Exception):
    """Base exception for parsing errors."""

    def __init__(self, message: str, recovery_hint: Optional[str] = None):
        """
        Initialize a parsing error.

        Args:
            message: The error message describing what went wrong
            recovery_hint: Optional hint on how to recover from this error
        """
        self.message = message
        self.recovery_hint = recovery_hint
        super().__init__(self.message)


class ValidationError(ParsingError):
    """Exception for validation errors in parsed data."""

    pass


class FileAccessError(ParsingError):
    """Exception for file access related errors."""

    pass


class SyntaxParsingError(ParsingError):
    """Exception for syntax errors during parsing."""

    pass
