"""Integrated parser combining AST and docstring parsing.

This module provides the IntegratedParser class that combines AST parsing
with docstring parsing to provide complete function analysis.
"""

import logging
from collections.abc import Generator

from .ast_parser import (
    ParsedFunction,
    RawDocstring,
    parse_python_file,
    parse_python_file_lazy,
)
from .docstring_parser import DocstringParser, ParsedDocstring

logger = logging.getLogger(__name__)


class IntegratedParser:
    """Combines AST and docstring parsing for complete function analysis."""

    def __init__(self) -> None:
        self.docstring_parser = DocstringParser()
        self._cache: dict[str, ParsedDocstring | None] = (
            {}
        )  # Simple cache for parsed docstrings

    def parse_file(self, file_path: str) -> list[ParsedFunction]:
        """Parse file and enrich with parsed docstrings.

        Steps:
        1. Parse AST to get functions
        2. Parse each function's docstring
        3. Attach parsed docstring to function
        4. Return enriched functions

        Args:
            file_path: Path to Python file to parse

        Returns:
            List of ParsedFunction objects with enriched docstrings

        Raises:
            FileNotFoundError: If file does not exist
            ParsingError: If parsing fails
        """
        # Get functions from AST parser
        functions = parse_python_file(file_path)

        # Enrich with parsed docstrings
        for func in functions:
            if func.docstring and isinstance(func.docstring, RawDocstring):
                # Check cache first
                cache_key = str(hash(func.docstring.raw_text))
                if cache_key in self._cache:
                    func.docstring = self._cache[cache_key]
                else:
                    # Parse docstring
                    parsed = self.docstring_parser.parse(func.docstring.raw_text)
                    if parsed:
                        func.docstring = parsed
                        self._cache[cache_key] = parsed
                    else:
                        # Keep raw docstring if parsing fails
                        logger.warning(
                            f"Failed to parse docstring for {func.signature.name} "
                            f"in {file_path}:{func.line_number}"
                        )

        return functions

    def parse_file_lazy(self, file_path: str) -> Generator[ParsedFunction, None, None]:
        """Generator version for memory efficiency.

        Args:
            file_path: Path to Python file to parse

        Yields:
            ParsedFunction objects with enriched docstrings

        Raises:
            FileNotFoundError: If file does not exist
            ParsingError: If parsing fails
        """
        for func in parse_python_file_lazy(file_path):
            if func.docstring and isinstance(func.docstring, RawDocstring):
                parsed = self.docstring_parser.parse(func.docstring.raw_text)
                if parsed:
                    func.docstring = parsed
            yield func

    def clear_cache(self) -> None:
        """Clear the docstring parsing cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and hit statistics
        """
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())[:10],  # First 10 for debugging
        }
