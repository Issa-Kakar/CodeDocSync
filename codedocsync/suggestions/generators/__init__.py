"""
Suggestion generators for specific issue types.

This module provides specialized generators for different types of documentation
inconsistencies including parameter issues, return type problems, exception
documentation, behavioral descriptions, examples, and edge cases.
"""

from .behavior_generator import BehaviorSuggestionGenerator
from .edge_case_handlers import EdgeCaseSuggestionGenerator
from .example_generator import ExampleSuggestionGenerator
from .parameter_generator import ParameterSuggestionGenerator
from .raises_generator import RaisesSuggestionGenerator
from .return_generator import ReturnSuggestionGenerator

__all__ = [
    "ParameterSuggestionGenerator",
    "ReturnSuggestionGenerator",
    "RaisesSuggestionGenerator",
    "BehaviorSuggestionGenerator",
    "ExampleSuggestionGenerator",
    "EdgeCaseSuggestionGenerator",
]
