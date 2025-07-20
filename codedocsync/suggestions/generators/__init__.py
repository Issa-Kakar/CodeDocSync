"""
Suggestion generators for specific issue types.

This module provides specialized generators for different types of documentation
inconsistencies including parameter issues, return type problems, and more.
"""

from .parameter_generator import ParameterSuggestionGenerator

__all__ = ["ParameterSuggestionGenerator"]
