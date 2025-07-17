"""
CodeDocSync Parser Module.

This module provides the public API for parsing Python source code files.
"""

from .ast_parser import (
    parse_python_file,
    parse_python_file_lazy,
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
)
from ..utils.errors import ParsingError, ValidationError

__all__ = [
    "parse_python_file",
    "parse_python_file_lazy",
    "ParsedFunction",
    "FunctionSignature",
    "FunctionParameter",
    "ParsingError",
    "ValidationError",
]
