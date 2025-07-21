"""Data models for the matching system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codedocsync.parser import ParsedDocstring, ParsedFunction, RawDocstring


class MatchType(Enum):
    """Type of match found between function and documentation."""

    EXACT = "exact"  # Exact name match in same location
    FUZZY = "fuzzy"  # Similar name (get_user vs getUser)
    CONTEXTUAL = "contextual"  # Based on imports/context
    SEMANTIC = "semantic"  # Based on embedding similarity
    NO_MATCH = "no_match"  # No documentation found


@dataclass
class MatchConfidence:
    """Confidence score for a match with detailed breakdown."""

    overall: float  # 0.0 to 1.0
    name_similarity: float
    location_score: float
    signature_similarity: float

    def __post_init__(self) -> None:
        """Validate confidence scores."""
        for attr in [
            "overall",
            "name_similarity",
            "location_score",
            "signature_similarity",
        ]:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {value}")


@dataclass
class MatchedPair:
    """A matched function-documentation pair."""

    function: ParsedFunction
    match_type: MatchType
    confidence: MatchConfidence
    match_reason: str  # Human-readable explanation

    # Optional: The matched documentation (for cross-file matches)
    docstring: RawDocstring | ParsedDocstring | None = None

    # Optional: If documentation is in a different location
    doc_location: str | None = None  # e.g., "class docstring", "module docstring"

    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence match."""
        return self.confidence.overall >= 0.85


@dataclass
class MatchResult:
    """Complete result of matching operation."""

    matched_pairs: list[MatchedPair] = field(default_factory=list)
    unmatched_functions: list[ParsedFunction] = field(default_factory=list)
    total_functions: int = 0
    match_duration_ms: float = 0.0

    @property
    def match_rate(self) -> float:
        """Calculate the percentage of functions matched."""
        if self.total_functions == 0:
            return 0.0
        return len(self.matched_pairs) / self.total_functions

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_functions": self.total_functions,
            "matched": len(self.matched_pairs),
            "unmatched": len(self.unmatched_functions),
            "match_rate": f"{self.match_rate:.1%}",
            "duration_ms": self.match_duration_ms,
            "match_types": self._count_match_types(),
        }

    def _count_match_types(self) -> dict[str, int]:
        """Count matches by type."""
        counts = {match_type.value: 0 for match_type in MatchType}
        for pair in self.matched_pairs:
            counts[pair.match_type.value] += 1
        return counts


class MatchingError(Exception):
    """Base exception for matching errors."""

    def __init__(self, message: str, recovery_hint: str | None = None) -> None:
        super().__init__(message)
        self.recovery_hint = recovery_hint
