"""
Comprehensive data models for the suggestion generation system.

This module defines the core data structures for representing documentation
fix suggestions, including different suggestion types, diff information,
and metadata for quality tracking.
"""

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SuggestionType(Enum):
    """Types of suggestions we can generate."""

    FULL_DOCSTRING = "full_docstring"  # Complete docstring replacement
    PARAMETER_UPDATE = "parameter_update"  # Update specific parameter
    RETURN_UPDATE = "return_update"  # Update return documentation
    RAISES_UPDATE = "raises_update"  # Update exception documentation
    DESCRIPTION_UPDATE = "description"  # Update description only
    EXAMPLE_UPDATE = "example"  # Update or add examples


class DocstringStyle(Enum):
    """Supported docstring styles."""

    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    REST = "rest"
    AUTO_DETECT = "auto_detect"


@dataclass
class SuggestionDiff:
    """Represents a diff between original and suggested text."""

    original_lines: list[str]
    suggested_lines: list[str]
    start_line: int
    end_line: int

    def __post_init__(self):
        """Validate diff data."""
        if self.start_line < 1:
            raise ValueError(f"start_line must be positive, got {self.start_line}")

        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )

    def to_unified_diff(self, filename: str = "docstring") -> str:
        """Generate unified diff format."""
        return "\n".join(
            difflib.unified_diff(
                self.original_lines,
                self.suggested_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                fromfiledate="(original)",
                tofiledate="(suggested)",
                lineterm="",
            )
        )

    @property
    def lines_changed(self) -> int:
        """Get number of lines changed."""
        return max(len(self.original_lines), len(self.suggested_lines))

    @property
    def additions(self) -> int:
        """Get number of lines added."""
        return max(0, len(self.suggested_lines) - len(self.original_lines))

    @property
    def deletions(self) -> int:
        """Get number of lines deleted."""
        return max(0, len(self.original_lines) - len(self.suggested_lines))


@dataclass
class SuggestionMetadata:
    """Metadata about how the suggestion was generated."""

    generator_type: str  # Which generator created this
    generator_version: str = "1.0.0"
    template_used: str | None = None  # Template name if applicable
    style_detected: str | None = None  # Detected docstring style
    rule_triggers: list[str] = field(default_factory=list)  # Rules that triggered
    llm_used: bool = False  # Whether LLM was involved
    generation_time_ms: float = 0.0  # Time to generate
    token_usage: int | None = None  # Tokens if LLM used

    def __post_init__(self):
        """Validate metadata."""
        if self.generation_time_ms < 0:
            raise ValueError(
                f"generation_time_ms must be non-negative, got {self.generation_time_ms}"
            )

        if not self.generator_type.strip():
            raise ValueError("generator_type cannot be empty")


@dataclass
class SuggestionContext:
    """Context needed to generate suggestions."""

    # Core data from analyzer
    issue: Any  # InconsistencyIssue from analyzer.models
    function: Any  # ParsedFunction from parser
    docstring: Any | None = None  # ParsedDocstring if available

    # Project context
    project_style: str = "google"  # From configuration
    max_line_length: int = 88  # For line wrapping
    preserve_descriptions: bool = True  # Keep existing descriptions

    # Additional context for advanced suggestions
    surrounding_code: str | None = None  # For context-aware suggestions
    related_functions: list[Any] = field(default_factory=list)  # Similar functions
    file_imports: list[str] = field(default_factory=list)  # Available imports

    def __post_init__(self):
        """Validate context."""
        if self.max_line_length < 40:
            raise ValueError(
                f"max_line_length too small: {self.max_line_length} (min 40)"
            )

        if self.project_style not in ["google", "numpy", "sphinx", "rest"]:
            raise ValueError(
                f"Invalid project_style: {self.project_style}. "
                f"Must be one of: google, numpy, sphinx, rest"
            )


@dataclass
class Suggestion:
    """A single fix suggestion with complete information."""

    # Core content
    suggestion_type: SuggestionType
    original_text: str  # Current docstring/code
    suggested_text: str  # Fixed version
    confidence: float  # 0.0-1.0 confidence in suggestion

    # Diff and formatting
    diff: SuggestionDiff
    style: str  # Docstring style used
    copy_paste_ready: bool = True  # If suggestion can be directly applied

    # Quality indicators
    is_actionable: bool = True  # If suggestion is specific enough
    validation_passed: bool = True  # If suggestion is syntactically valid

    # Metadata
    metadata: SuggestionMetadata = field(
        default_factory=lambda: SuggestionMetadata(generator_type="unknown")
    )

    # Additional context
    affected_sections: list[str] = field(default_factory=list)  # Which parts changed
    line_range: tuple[int, int] = (1, 1)  # Target line range

    def __post_init__(self):
        """Validate suggestion data."""
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate required text fields
        if not self.original_text:
            raise ValueError("original_text cannot be empty")

        if not self.suggested_text:
            raise ValueError("suggested_text cannot be empty")

        # Validate line range
        start, end = self.line_range
        if start < 1 or end < start:
            raise ValueError(
                f"Invalid line_range: ({start}, {end}). "
                f"Start must be >= 1 and end >= start"
            )

        # Validate style
        valid_styles = ["google", "numpy", "sphinx", "rest", "unknown"]
        if self.style not in valid_styles:
            raise ValueError(f"style must be one of {valid_styles}, got '{self.style}'")

    @property
    def diff_preview(self) -> str:
        """Get a formatted diff preview."""
        return self.diff.to_unified_diff()

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence suggestion."""
        return self.confidence >= 0.8

    @property
    def is_ready_to_apply(self) -> bool:
        """Check if suggestion is ready for direct application."""
        return (
            self.copy_paste_ready
            and self.is_actionable
            and self.validation_passed
            and self.confidence >= 0.7
        )

    def get_quality_score(self) -> float:
        """Calculate overall quality score (0.0-1.0)."""
        factors = [
            self.confidence,
            1.0 if self.is_actionable else 0.5,
            1.0 if self.validation_passed else 0.0,
            1.0 if self.copy_paste_ready else 0.7,
        ]
        return sum(factors) / len(factors)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "suggestion_type": self.suggestion_type.value,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "confidence": self.confidence,
            "style": self.style,
            "copy_paste_ready": self.copy_paste_ready,
            "is_actionable": self.is_actionable,
            "validation_passed": self.validation_passed,
            "quality_score": self.get_quality_score(),
            "diff": {
                "unified": self.diff_preview,
                "lines_changed": self.diff.lines_changed,
                "additions": self.diff.additions,
                "deletions": self.diff.deletions,
            },
            "affected_sections": self.affected_sections,
            "line_range": self.line_range,
            "metadata": {
                "generator_type": self.metadata.generator_type,
                "generator_version": self.metadata.generator_version,
                "template_used": self.metadata.template_used,
                "style_detected": self.metadata.style_detected,
                "rule_triggers": self.metadata.rule_triggers,
                "llm_used": self.metadata.llm_used,
                "generation_time_ms": self.metadata.generation_time_ms,
                "token_usage": self.metadata.token_usage,
            },
        }


@dataclass
class SuggestionBatch:
    """A collection of related suggestions."""

    suggestions: list[Suggestion] = field(default_factory=list)
    function_name: str = ""
    file_path: str = ""
    total_generation_time_ms: float = 0.0

    def __post_init__(self):
        """Validate batch data."""
        if self.total_generation_time_ms < 0:
            raise ValueError(
                f"total_generation_time_ms must be non-negative, "
                f"got {self.total_generation_time_ms}"
            )

    @property
    def high_confidence_suggestions(self) -> list[Suggestion]:
        """Get all high-confidence suggestions."""
        return [s for s in self.suggestions if s.is_high_confidence]

    @property
    def ready_to_apply_suggestions(self) -> list[Suggestion]:
        """Get suggestions ready for direct application."""
        return [s for s in self.suggestions if s.is_ready_to_apply]

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all suggestions."""
        if not self.suggestions:
            return 0.0
        return sum(s.confidence for s in self.suggestions) / len(self.suggestions)

    def get_best_suggestion(self) -> Suggestion | None:
        """Get the highest quality suggestion."""
        if not self.suggestions:
            return None
        return max(self.suggestions, key=lambda s: s.get_quality_score())

    def sort_by_quality(self) -> list[Suggestion]:
        """Get suggestions sorted by quality score (best first)."""
        return sorted(
            self.suggestions, key=lambda s: s.get_quality_score(), reverse=True
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for this batch."""
        return {
            "total_suggestions": len(self.suggestions),
            "high_confidence": len(self.high_confidence_suggestions),
            "ready_to_apply": len(self.ready_to_apply_suggestions),
            "average_confidence": self.average_confidence,
            "best_quality_score": (
                self.get_best_suggestion().get_quality_score()
                if self.get_best_suggestion()
                else 0.0
            ),
            "total_generation_time_ms": self.total_generation_time_ms,
            "function_name": self.function_name,
            "file_path": self.file_path,
        }


class SuggestionError(Exception):
    """Base exception for suggestion system."""

    pass


class StyleDetectionError(SuggestionError):
    """Failed to detect docstring style."""

    def __init__(self, message: str, fallback_style: str = "google"):
        super().__init__(message)
        self.fallback_style = fallback_style


class SuggestionGenerationError(SuggestionError):
    """Failed to generate suggestion."""

    def __init__(self, message: str, partial_result: str | None = None):
        super().__init__(message)
        self.partial_result = partial_result


class SuggestionValidationError(SuggestionError):
    """Suggestion failed validation."""

    def __init__(self, message: str, suggestion_text: str):
        super().__init__(message)
        self.suggestion_text = suggestion_text
