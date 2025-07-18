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
from .docstring_models import (
    DocstringFormat,
    DocstringParameter,
    DocstringReturns,
    DocstringRaises,
    ParsedDocstring,
)
from .docstring_parser import DocstringParser
from ..utils.errors import ParsingError, ValidationError

__all__ = [
    "parse_python_file",
    "parse_python_file_lazy",
    "ParsedFunction",
    "FunctionSignature",
    "FunctionParameter",
    "DocstringFormat",
    "DocstringParameter",
    "DocstringReturns",
    "DocstringRaises",
    "ParsedDocstring",
    "DocstringParser",
    "ParsingError",
    "ValidationError",
]
