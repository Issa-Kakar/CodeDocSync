"""
CodeDocSync Parser Module.

This module provides the public API for parsing Python source code files.
"""

from ..utils.errors import ParsingError, ValidationError
from .ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
    parse_python_file,
    parse_python_file_lazy,
)
from .docstring_models import (
    DocstringFormat,
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
    ParsedDocstring,
)
from .docstring_parser import DocstringParser
from .integrated_parser import IntegratedParser

__all__ = [
    "parse_python_file",
    "parse_python_file_lazy",
    "ParsedFunction",
    "FunctionSignature",
    "FunctionParameter",
    "RawDocstring",
    "DocstringFormat",
    "DocstringParameter",
    "DocstringReturns",
    "DocstringRaises",
    "ParsedDocstring",
    "DocstringParser",
    "IntegratedParser",
    "ParsingError",
    "ValidationError",
]
