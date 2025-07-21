"""Direct matcher for same-file function-documentation matching."""

import logging
import time

from codedocsync.parser import ParsedDocstring, ParsedFunction, RawDocstring

from .models import MatchConfidence, MatchedPair, MatchResult, MatchType

logger = logging.getLogger(__name__)


class DirectMatcher:
    """Matches functions to documentation within the same file."""

    def __init__(self):
        """Initialize the direct matcher."""
        self.stats = {
            "exact_matches": 0,
            "no_matches": 0,
            "total_processed": 0,
        }

    def match_functions(self, functions: list[ParsedFunction]) -> MatchResult:
        """
        Match functions to their documentation.

        Args:
            functions: List of parsed functions from a file/project

        Returns:
            MatchResult containing all matches and unmatched functions

        Example:
            >>> matcher = DirectMatcher()
            >>> result = matcher.match_functions(parsed_functions)
            >>> print(f"Matched {result.match_rate:.1%} of functions")
        """
        start_time = time.time()

        if not functions:
            return MatchResult(total_functions=0, match_duration_ms=0.0)

        # Reset stats for this run
        self._reset_stats()

        matched_pairs = []
        unmatched_functions = []

        # Group functions by file for efficient matching
        functions_by_file = self._group_by_file(functions)

        for _, file_functions in functions_by_file.items():
            file_matches = self._match_functions_in_file(file_functions)

            for _, (func, match) in file_matches.items():
                if match:
                    matched_pairs.append(match)
                else:
                    unmatched_functions.append(func)

        duration_ms = (time.time() - start_time) * 1000

        result = MatchResult(
            matched_pairs=matched_pairs,
            unmatched_functions=unmatched_functions,
            total_functions=len(functions),
            match_duration_ms=duration_ms,
        )

        logger.info(
            f"Direct matching complete: {result.match_rate:.1%} matched "
            f"({len(matched_pairs)}/{len(functions)}) in {duration_ms:.1f}ms"
        )

        return result

    def _match_functions_in_file(
        self, functions: list[ParsedFunction]
    ) -> dict[int, tuple[ParsedFunction, MatchedPair | None]]:
        """Match functions to their own documentation within a single file."""
        matches = {}

        for i, func in enumerate(functions):
            self.stats["total_processed"] += 1

            if not func.docstring:
                matches[i] = (func, None)
                self.stats["no_matches"] += 1
                continue

            # Try exact match (function matches its own docstring)
            match = self._try_exact_match(func)

            matches[i] = (func, match)

            if match:
                self.stats["exact_matches"] += 1
            else:
                self.stats["no_matches"] += 1

        return matches

    def _try_exact_match(self, func: ParsedFunction) -> MatchedPair | None:
        """
        Try to create an exact match for a function.

        A function with a docstring is considered an exact match if:
        1. The docstring exists
        2. The docstring is in the expected location (right after function def)
        3. The function signature is properly documented
        """
        if not func.docstring:
            return None

        # If we have a parsed docstring, check parameter alignment
        confidence = self._calculate_exact_match_confidence(func)

        if confidence.overall >= 0.7:  # Reasonable threshold for exact match
            return MatchedPair(
                function=func,
                match_type=MatchType.EXACT,
                confidence=confidence,
                match_reason="Exact match: function and docstring aligned",
                docstring=func.docstring,  # Include the docstring
            )

        return None

    def _calculate_exact_match_confidence(
        self, func: ParsedFunction
    ) -> MatchConfidence:
        """Calculate confidence for an exact match."""
        # Start with perfect scores
        name_score = 1.0  # Same function, so name matches
        location_score = 1.0  # Docstring is in expected location

        # Calculate signature similarity (handles all docstring types)
        sig_score = self._calculate_signature_similarity(func)

        # For exact matches (function with its own docstring), we want to match
        # even if parameters don't align - that's what the analyzer will detect
        # So we use a higher weight for name and location scores
        overall = name_score * 0.4 + location_score * 0.4 + sig_score * 0.2

        return MatchConfidence(
            overall=overall,
            name_similarity=name_score,
            location_score=location_score,
            signature_similarity=sig_score,
        )

    def _calculate_signature_similarity(self, func: ParsedFunction) -> float:
        """
        Calculate how well the signature matches the documentation.

        Returns a score from 0.0 to 1.0.
        """
        if not func.docstring:
            return 1.0  # No docstring to check

        # Only ParsedDocstring has parameters attribute
        if isinstance(func.docstring, RawDocstring):
            return 1.0  # RawDocstring - can't check parameters

        # Must be ParsedDocstring with parameters
        if not isinstance(func.docstring, ParsedDocstring):
            return 1.0  # Unknown type, can't check

        func_params = {p.name for p in func.signature.parameters}
        doc_params = {p.name for p in func.docstring.parameters}

        if not func_params and not doc_params:
            return 1.0  # Both empty

        if not func_params or not doc_params:
            return 0.5  # Give partial credit if one side is empty

        # Calculate Jaccard similarity
        intersection = func_params & doc_params
        union = func_params | doc_params

        return len(intersection) / len(union)

    def _group_by_file(
        self, functions: list[ParsedFunction]
    ) -> dict[str, list[ParsedFunction]]:
        """Group functions by their file path."""
        grouped = {}
        for func in functions:
            if func.file_path not in grouped:
                grouped[func.file_path] = []
            grouped[func.file_path].append(func)
        return grouped

    def _reset_stats(self):
        """Reset statistics for a new matching run."""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> dict[str, int]:
        """Get matching statistics."""
        return self.stats.copy()
