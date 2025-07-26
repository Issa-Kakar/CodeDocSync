"""
Base suggestion generator interface and validation utilities.

This module provides the foundation for all suggestion generators, including
common functionality like style detection, validation, and error handling.
"""

import ast
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, cast

from .models import (
    Suggestion,
    SuggestionContext,
    SuggestionError,
    SuggestionMetadata,
    SuggestionValidationError,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseSuggestionGenerator(ABC):
    """Base class for all suggestion generators."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the generator with configuration."""
        from .config import SuggestionConfig

        self.config = config or SuggestionConfig()
        self._validation_cache: dict[str, bool] = {}

    @abstractmethod
    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate a suggestion for the given context."""
        pass

    def generate_with_timing(self, context: SuggestionContext) -> Suggestion:
        """Generate suggestion with performance timing."""
        start_time = time.perf_counter()

        try:
            suggestion = self.generate(context)
            generation_time = (time.perf_counter() - start_time) * 1000

            # Update metadata with timing
            suggestion.metadata.generation_time_ms = generation_time

            return suggestion

        except Exception as e:
            generation_time = (time.perf_counter() - start_time) * 1000
            raise SuggestionError(
                f"Failed to generate suggestion after {generation_time:.1f}ms: {e}"
            ) from e

    def validate_suggestion(self, suggestion: Suggestion) -> bool:
        """Validate that suggestion is syntactically correct and actionable."""
        try:
            # 1. Check Python string syntax
            if not self._is_valid_python_string(suggestion.suggested_text):
                suggestion.validation_passed = False
                return False

            # 2. Check proper indentation
            if not self._has_consistent_indentation(suggestion.suggested_text):
                suggestion.validation_passed = False
                return False

            # 3. Check for unescaped quotes
            if not self._has_proper_quote_escaping(suggestion.suggested_text):
                suggestion.validation_passed = False
                return False

            # 4. Check style format compliance
            if not self._matches_expected_style(
                suggestion.suggested_text, suggestion.style
            ):
                suggestion.validation_passed = False
                return False

            # 5. Check actionability (not too vague)
            if not self._is_actionable(suggestion.suggested_text):
                suggestion.is_actionable = False

            suggestion.validation_passed = True
            return True

        except Exception as e:
            raise SuggestionValidationError(
                f"Validation failed: {e}", suggestion.suggested_text
            ) from e

    def _is_valid_python_string(self, text: str) -> bool:
        """Check if text can be parsed as a valid Python string literal."""
        # Use cache for performance
        cache_key = f"python_string:{hash(text)}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        try:
            # Try to parse as triple-quoted string
            test_code = f'"""{text}"""'
            ast.parse(f"x = {test_code}")
            self._validation_cache[cache_key] = True
            return True
        except SyntaxError:
            try:
                # Try with escaped quotes
                escaped = text.replace('"""', '\\"\\"\\"')
                test_code = f'"""{escaped}"""'
                ast.parse(f"x = {test_code}")
                self._validation_cache[cache_key] = True
                return True
            except SyntaxError:
                self._validation_cache[cache_key] = False
                return False

    def _has_consistent_indentation(self, text: str) -> bool:
        """Check for consistent indentation (tabs vs spaces)."""
        lines = text.split("\n")
        has_tabs = any("\t" in line for line in lines)
        has_spaces = any(line.startswith("    ") for line in lines)

        # Mixed indentation is problematic
        if has_tabs and has_spaces:
            return False

        # Check for reasonable indentation levels
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip(" "))
                # Flag excessive indentation (more than 12 spaces = 3 levels)
                if leading_spaces > 12:
                    return False
                # Check for non-multiple-of-4 indentation (when using spaces)
                if leading_spaces > 0 and leading_spaces % 4 != 0:
                    return False

        return True

    def _has_proper_quote_escaping(self, text: str) -> bool:
        """Check for proper quote escaping in docstrings."""
        # Count unescaped triple quotes
        triple_quote_count = 0
        i = 0
        while i < len(text) - 2:
            if text[i : i + 3] == '"""' and (i == 0 or text[i - 1] != "\\"):
                triple_quote_count += 1
            i += 1

        # Should have even number of unescaped triple quotes
        return triple_quote_count % 2 == 0

    def _matches_expected_style(self, text: str, style: str) -> bool:
        """Check if text matches expected docstring style format."""
        if style == "google":
            issues = self._validate_google_style(text)
            return len(issues) == 0
        elif style == "numpy":
            issues = self._validate_numpy_style(text)
            return len(issues) == 0
        elif style == "sphinx":
            issues = self._validate_sphinx_style(text)
            return len(issues) == 0
        elif style == "rest":
            issues = self._validate_rest_style(text)
            return len(issues) == 0
        else:
            # Unknown style, just check basic structure
            return len(text.strip()) > 0

    def _validate_google_style(self, text: str) -> list[str]:
        """Validate Google-style docstring format."""
        issues = []
        google_sections = [
            "Args:",
            "Arguments:",
            "Returns:",
            "Return:",
            "Raises:",
            "Yields:",
            "Note:",
            "Example:",
        ]

        # If it has sections, they should be properly formatted
        for section in google_sections:
            if section in text:
                # Check that section is followed by proper indentation
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    if section in line:
                        # Next non-empty line should be indented
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip():
                                if not lines[j].startswith("    "):
                                    issues.append(
                                        f"Content after '{section}' should be indented 4 spaces"
                                    )
                                break
        return issues

    def _validate_numpy_style(self, text: str) -> list[str]:
        """Validate NumPy-style docstring format."""
        issues = []
        numpy_sections = [
            "Parameters",
            "Returns",
            "Yields",
            "Raises",
            "See Also",
            "Notes",
            "Examples",
        ]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip() in numpy_sections:
                # Next line should be underline with dashes
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not next_line.strip().startswith("-"):
                        issues.append(
                            f"NumPy section '{line.strip()}' should be followed by dashes"
                        )
                    elif len(next_line.strip()) != len(line.strip()):
                        issues.append(
                            "NumPy section underline should match header length"
                        )
        return issues

    def _validate_sphinx_style(self, text: str) -> list[str]:
        """Validate Sphinx-style docstring format."""
        issues = []
        sphinx_patterns = [
            (r":param\s+(\w+)\s*:", "param"),
            (r":type\s+(\w+)\s*:", "type"),
            (r":returns?\s*:", "returns"),
            (r":rtype\s*:", "rtype"),
            (r":raises?\s+(\w+)\s*:", "raises"),
        ]

        for pattern, field_type in sphinx_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Basic validation - field should be at line start or properly indented
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_prefix = text[line_start : match.start()]
                if line_prefix and not line_prefix.isspace():
                    issues.append(
                        f"Sphinx field '{field_type}' should start at beginning of line or be properly indented"
                    )

        return issues

    def _validate_rest_style(self, text: str) -> list[str]:
        """Validate reStructuredText style docstring format."""
        issues = []

        # Check directive formatting
        directive_pattern = r"^\s*\.\.\s+(\w+)::\s*(.*)$"
        lines = text.split("\n")

        for i, line in enumerate(lines):
            match = re.match(directive_pattern, line)
            if match:
                directive_name = match.group(1)
                # Check if content is properly indented on following lines
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        if not lines[j].startswith("   "):  # At least 3 spaces
                            issues.append(
                                f"Content after '{directive_name}::' directive should be indented"
                            )
                        break

        return issues

    def _is_actionable(self, text: str) -> bool:
        """Check if suggestion is specific and actionable."""
        # Flag vague suggestions
        vague_phrases = [
            "fix this",
            "update documentation",
            "improve description",
            "add more details",
            "make it better",
            "TODO",
            "FIXME",
            "[placeholder]",
        ]

        text_lower = text.lower()
        for phrase in vague_phrases:
            if phrase in text_lower:
                return False

        # Check for minimum content length
        if len(text.strip()) < 10:
            return False

        # Should contain actual parameter names, types, or descriptions
        if not re.search(r"[a-zA-Z_][a-zA-Z0-9_]*", text):
            return False

        return True

    def create_metadata(
        self,
        generator_type: str,
        template_used: str | None = None,
        style_detected: str | None = None,
        rule_triggers: list[str] | None = None,
    ) -> SuggestionMetadata:
        """Create metadata for a suggestion."""
        return SuggestionMetadata(
            generator_type=generator_type,
            template_used=template_used,
            style_detected=style_detected,
            rule_triggers=rule_triggers or [],
            llm_used=False,  # Override in subclasses if needed
        )

    def format_docstring_lines(self, content: str, indent: int = 4) -> list[str]:
        """Format docstring content with proper indentation."""
        lines = content.split("\n")
        formatted = []

        for line in lines:
            if line.strip():  # Non-empty line
                formatted.append(" " * indent + line.strip())
            else:  # Empty line
                formatted.append("")

        return formatted

    def wrap_long_lines(self, text: str, max_length: int | None = None) -> str:
        """Wrap long lines to respect max line length."""
        max_length = max_length or self.config.max_line_length

        lines = text.split("\n")
        wrapped_lines = []

        for line in lines:
            if len(line) <= max_length:
                wrapped_lines.append(line)
            else:
                # Simple word wrapping preserving indentation
                indent = len(line) - len(line.lstrip())
                words = line.strip().split()
                current_line = " " * indent

                for word in words:
                    # Check if adding this word would exceed max length
                    test_line = current_line + (
                        " " + word if current_line.strip() else word
                    )

                    if len(test_line) <= max_length:
                        current_line = test_line
                    else:
                        # Start new line
                        if current_line.strip():
                            wrapped_lines.append(current_line)
                        current_line = (
                            " " * (indent + 4) + word
                        )  # Extra indent for continuation

                if current_line.strip():
                    wrapped_lines.append(current_line)

        return "\n".join(wrapped_lines)

    def preserve_existing_content(
        self, original: str, updated_section: str, section_name: str
    ) -> str:
        """Intelligently preserve existing content while updating a section."""
        # This is a simplified implementation
        # In a full implementation, this would parse the docstring structure
        # and only replace the specific section

        if not original.strip():
            return updated_section

        # For now, return the updated section
        # TODO: Implement smart merging in future chunks
        return updated_section

    def _format_rag_examples(self, examples: list[dict[str, Any]]) -> str:
        """Format RAG examples for prompt context."""
        if not examples:
            logger.debug("No RAG examples to format")
            return ""

        logger.debug(f"Formatting {len(examples)} RAG examples for prompt context")

        formatted = ["Similar well-documented functions:"]
        for i, ex in enumerate(examples[:2], 1):
            similarity = ex.get("similarity", 0)
            logger.debug(
                f"Example {i}: {ex.get('signature', '')[:50]} (similarity: {similarity:.2f})"
            )
            formatted.append(f"\nExample {i} (similarity: {similarity:.2f}):")
            formatted.append(f"Signature: {ex.get('signature', '')}")
            formatted.append(f"Documentation:\n{ex.get('docstring', '')}")

        return "\n".join(formatted)


def with_suggestion_fallback(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for graceful degradation in suggestion generation."""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except SuggestionError as e:
            # Return partial result if available
            if hasattr(e, "partial_result") and e.partial_result:
                return cast(T, e.partial_result)

            # Create fallback suggestion
            if args and hasattr(args[0], "_create_fallback_suggestion"):
                return cast(
                    T,
                    args[0]._create_fallback_suggestion(
                        args[1] if len(args) > 1 else None
                    ),
                )

            # Re-raise if no fallback possible
            raise
        except Exception as e:
            # Convert unexpected errors to SuggestionError
            raise SuggestionError(
                f"Unexpected error in suggestion generation: {e}"
            ) from e

    return wrapper
