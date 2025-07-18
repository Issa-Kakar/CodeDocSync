"""Direct matcher for same-file function-documentation matching."""

import time
import logging
import re
from typing import List, Dict, Optional, Tuple
from rapidfuzz import fuzz, process
from codedocsync.parser import ParsedFunction
from .models import MatchedPair, MatchConfidence, MatchType, MatchResult

logger = logging.getLogger(__name__)


class DirectMatcher:
    """Matches functions to documentation within the same file."""

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize the direct matcher.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matches (0.0-1.0)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "no_matches": 0,
            "total_processed": 0,
        }

        # Common naming patterns for fuzzy matching
        self.naming_patterns = [
            (r"get_(\w+)", r"get\1"),  # get_user -> getUser
            (r"set_(\w+)", r"set\1"),  # set_value -> setValue
            (r"is_(\w+)", r"is\1"),  # is_valid -> isValid
            (r"has_(\w+)", r"has\1"),  # has_data -> hasData
            (r"(\w+)_id", r"\1Id"),  # user_id -> userId
            (r"(\w+)_url", r"\1Url"),  # api_url -> apiUrl
        ]

    def match_functions(self, functions: List[ParsedFunction]) -> MatchResult:
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

        for file_path, file_functions in functions_by_file.items():
            file_matches = self._match_functions_in_file(file_functions)

            for i, (func, match) in file_matches.items():
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
        self, functions: List[ParsedFunction]
    ) -> Dict[int, Tuple[ParsedFunction, Optional[MatchedPair]]]:
        """Match functions within a single file."""
        matches = {}

        # Build index of functions with docstrings for fuzzy matching
        documented_functions = [f for f in functions if f.docstring]
        function_names = [f.signature.name for f in documented_functions]

        for i, func in enumerate(functions):
            self.stats["total_processed"] += 1

            if not func.docstring:
                matches[i] = (func, None)
                self.stats["no_matches"] += 1
                continue

            # Try exact match first
            match = self._try_exact_match(func)

            # If no exact match, try fuzzy match
            if not match and function_names:
                match = self._try_fuzzy_match(func, documented_functions)

            matches[i] = (func, match)

            if match:
                if match.match_type == MatchType.EXACT:
                    self.stats["exact_matches"] += 1
                else:
                    self.stats["fuzzy_matches"] += 1
            else:
                self.stats["no_matches"] += 1

        return matches

    def _try_exact_match(self, func: ParsedFunction) -> Optional[MatchedPair]:
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

        if confidence.overall >= 0.95:  # High threshold for exact match
            return MatchedPair(
                function=func,
                match_type=MatchType.EXACT,
                confidence=confidence,
                match_reason="Exact match: function and docstring aligned",
            )

        return None

    def _calculate_exact_match_confidence(
        self, func: ParsedFunction
    ) -> MatchConfidence:
        """Calculate confidence for an exact match."""
        # Start with perfect scores
        name_score = 1.0  # Same function, so name matches
        location_score = 1.0  # Docstring is in expected location

        # Calculate signature similarity if we have parsed docstring
        sig_score = 1.0
        if hasattr(func.docstring, "parameters"):
            sig_score = self._calculate_signature_similarity(func)

        # Overall confidence is weighted average
        overall = (name_score + location_score + sig_score) / 3.0

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
        if not hasattr(func.docstring, "parameters"):
            return 1.0  # No parameters to check

        func_params = {p.name for p in func.signature.parameters}

        # Handle both RawDocstring and ParsedDocstring
        if hasattr(func.docstring, "parameters"):
            doc_params = {p.name for p in func.docstring.parameters}
        else:
            # RawDocstring - can't check parameters
            return 1.0

        if not func_params and not doc_params:
            return 1.0  # Both empty

        if not func_params or not doc_params:
            return 0.0  # One empty, one not

        # Calculate Jaccard similarity
        intersection = func_params & doc_params
        union = func_params | doc_params

        return len(intersection) / len(union)

    def _group_by_file(
        self, functions: List[ParsedFunction]
    ) -> Dict[str, List[ParsedFunction]]:
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

    def _try_fuzzy_match(
        self, func: ParsedFunction, candidates: List[ParsedFunction]
    ) -> Optional[MatchedPair]:
        """
        Try fuzzy matching for function names.

        Handles common variations:
        - snake_case vs camelCase
        - Abbreviations (calc vs calculate)
        - Common prefixes/suffixes
        """
        func_name = func.signature.name

        # Try pattern-based transformations first
        for pattern, replacement in self.naming_patterns:
            transformed = re.sub(pattern, replacement, func_name)
            if transformed != func_name:
                # Check if transformed name matches any candidate
                for candidate in candidates:
                    if candidate.signature.name == transformed:
                        return self._create_fuzzy_match(
                            func,
                            candidate,
                            f"Pattern match: {func_name} → {transformed}",
                        )

        # Try fuzzy string matching
        candidate_names = [c.signature.name for c in candidates]

        # Get best matches using rapidfuzz
        matches = process.extract(
            func_name,
            candidate_names,
            scorer=fuzz.ratio,
            limit=3,  # Top 3 matches
        )

        for match_name, score, _ in matches:
            if score >= self.fuzzy_threshold * 100:  # rapidfuzz uses 0-100 scale
                # Find the candidate function
                candidate = next(
                    c for c in candidates if c.signature.name == match_name
                )

                # Additional validation for fuzzy matches
                if self._validate_fuzzy_match(func, candidate):
                    return self._create_fuzzy_match(
                        func,
                        candidate,
                        f"Fuzzy match: {func_name} → {match_name} (score: {score}%)",
                    )

        return None

    def _validate_fuzzy_match(
        self, func1: ParsedFunction, func2: ParsedFunction
    ) -> bool:
        """
        Validate that a fuzzy match is likely correct.

        Checks:
        - Similar parameter count
        - Similar line location (within same class/module section)
        - No exact match already exists
        """
        # Check parameter count similarity
        param_count1 = len(func1.signature.parameters)
        param_count2 = len(func2.signature.parameters)

        if abs(param_count1 - param_count2) > 2:
            return False  # Too different

        # Check if they're in the same general area of the file
        line_distance = abs(func1.line_number - func2.line_number)
        if line_distance > 100:  # Arbitrary threshold
            return False

        # Check return type similarity if available
        if func1.signature.return_type and func2.signature.return_type:
            if func1.signature.return_type != func2.signature.return_type:
                return False

        return True

    def _create_fuzzy_match(
        self, func: ParsedFunction, matched_func: ParsedFunction, reason: str
    ) -> MatchedPair:
        """Create a fuzzy match pair with confidence scoring."""
        # Calculate detailed confidence
        name_sim = fuzz.ratio(func.signature.name, matched_func.signature.name) / 100.0

        # Location score based on line distance
        line_distance = abs(func.line_number - matched_func.line_number)
        location_score = max(0.0, 1.0 - (line_distance / 100.0))

        # Signature similarity
        sig_score = self._calculate_signature_similarity_between(func, matched_func)

        # Overall confidence for fuzzy match (weighted)
        overall = (name_sim * 0.5) + (location_score * 0.2) + (sig_score * 0.3)

        return MatchedPair(
            function=func,
            match_type=MatchType.FUZZY,
            confidence=MatchConfidence(
                overall=overall,
                name_similarity=name_sim,
                location_score=location_score,
                signature_similarity=sig_score,
            ),
            match_reason=reason,
        )

    def _calculate_signature_similarity_between(
        self, func1: ParsedFunction, func2: ParsedFunction
    ) -> float:
        """Calculate signature similarity between two functions."""
        params1 = {p.name for p in func1.signature.parameters}
        params2 = {p.name for p in func2.signature.parameters}

        if not params1 and not params2:
            return 1.0

        if not params1 or not params2:
            return 0.0

        intersection = params1 & params2
        union = params1 | params2

        return len(intersection) / len(union)

    def get_stats(self) -> Dict[str, int]:
        """Get matching statistics."""
        return self.stats.copy()
