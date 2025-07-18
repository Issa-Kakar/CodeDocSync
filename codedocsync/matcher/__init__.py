"""Matcher module for finding function-documentation pairs."""

from .models import MatchedPair, MatchConfidence, MatchType, MatchResult, MatchingError
from .direct_matcher import DirectMatcher

__all__ = [
    "MatchedPair",
    "MatchConfidence",
    "MatchType",
    "MatchResult",
    "DirectMatcher",
    "MatchingError",
]
