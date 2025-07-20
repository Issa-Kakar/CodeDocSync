"""
Output formatters for suggestion system.

This package provides different formatters for presenting suggestions
to users in various contexts (terminal, JSON, HTML, etc.).
"""

from .json_formatter import JSONSuggestionFormatter
from .terminal_formatter import OutputStyle, TerminalSuggestionFormatter

__all__ = [
    "TerminalSuggestionFormatter",
    "JSONSuggestionFormatter",
    "OutputStyle",
]
