"""Data models for parsed docstring components.

This module defines the core data structures used to represent parsed docstrings
across different formats (Google, NumPy, Sphinx, REST).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import re


class DocstringFormat(Enum):
    """Supported docstring formats."""

    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    REST = "rest"
    UNKNOWN = "unknown"


@dataclass
class DocstringParameter:
    """Single parameter documentation."""

    name: str
    type_str: Optional[str] = None
    description: str = ""
    is_optional: bool = False
    default_value: Optional[str] = None

    def __post_init__(self):
        """Validate parameter name."""
        # Validate parameter name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.name):
            # Allow *args and **kwargs
            if not re.match(r"^(\*{1,2})?[a-zA-Z_][a-zA-Z0-9_]*$", self.name):
                raise ValueError(f"Invalid parameter name: {self.name}")


@dataclass
class DocstringReturns:
    """Return value documentation."""

    type_str: Optional[str] = None
    description: str = ""


@dataclass
class DocstringRaises:
    """Exception documentation."""

    exception_type: str
    description: str = ""

    def __post_init__(self):
        """Validate exception type."""
        # Validate exception type is valid identifier
        if not re.match(
            r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", self.exception_type
        ):
            raise ValueError(f"Invalid exception type: {self.exception_type}")


@dataclass
class ParsedDocstring:
    """Complete parsed docstring with all components."""

    format: DocstringFormat
    summary: str
    description: Optional[str] = None
    parameters: List[DocstringParameter] = field(default_factory=list)
    returns: Optional[DocstringReturns] = None
    raises: List[DocstringRaises] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    raw_text: str = ""

    # Additional metadata
    is_valid: bool = True
    parse_errors: List[str] = field(default_factory=list)

    def get_parameter(self, name: str) -> Optional[DocstringParameter]:
        """Get parameter by name."""
        return next((p for p in self.parameters if p.name == name), None)
