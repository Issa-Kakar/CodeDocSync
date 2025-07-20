"""
Output formatters for suggestion system.

This package provides different formatters for presenting suggestions
to users in various contexts (terminal, JSON, HTML, etc.).
"""

from .terminal_formatter import TerminalSuggestionFormatter, OutputStyle
from .json_formatter import JSONSuggestionFormatter

__all__ = [
    "TerminalSuggestionFormatter",
    "JSONSuggestionFormatter",
    "OutputStyle",
]
