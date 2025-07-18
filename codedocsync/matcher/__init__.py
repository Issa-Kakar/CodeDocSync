"""Matcher module for finding function-documentation pairs."""

from .models import MatchedPair, MatchConfidence, MatchType, MatchResult, MatchingError
from .direct_matcher import DirectMatcher
from .facade import MatchingFacade

__all__ = [
    "MatchedPair",
    "MatchConfidence",
    "MatchType",
    "MatchResult",
    "DirectMatcher",
    "MatchingError",
    "MatchingFacade",
]
