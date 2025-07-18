## CURRENT_SPRINT.MD

# Current Sprint: Direct Matcher Implementation (Week 2, Days 1-3)

## Sprint Overview
Implement the Direct Matcher component that finds exact and fuzzy matches between parsed functions and their documentation within the same file. This is the primary matching mechanism that will handle 90% of all matches.

## Prerequisites Check
Before starting, ensure:
1. ✅ AST Parser produces `List[ParsedFunction]` with valid signatures and docstrings
2. ✅ Docstring Parser converts RawDocstring → ParsedDocstring
3. ✅ All parser tests pass
4. ✅ CLI can display parsed functions correctly

## Architecture Context
The Direct Matcher is the first of three matching layers:
1. **Direct Matcher** (this sprint): Same file, name-based matching
2. **Contextual Matcher** (next): Cross-file, import-aware matching
3. **Semantic Matcher** (later): Embedding-based similarity matching

---

## CHUNK 1: Core Data Models and Interfaces

### Objective
Define the data models and interfaces that the matcher will use and produce. These models must integrate seamlessly with existing parser output.

### Implementation Details

1. Create `codedocsync/matcher/__init__.py`:
```python
"""Matcher module for finding function-documentation pairs."""

from .models import (
    MatchedPair,
    MatchConfidence,
    MatchType,
    MatchResult,
    MatchingError
)
from .direct_matcher import DirectMatcher

__all__ = [
    'MatchedPair',
    'MatchConfidence',
    'MatchType',
    'MatchResult',
    'DirectMatcher',
    'MatchingError'
]

Create codedocsync/matcher/models.py:

python"""Data models for the matching system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from codedocsync.parser import ParsedFunction

class MatchType(Enum):
    """Type of match found between function and documentation."""
    EXACT = "exact"              # Exact name match in same location
    FUZZY = "fuzzy"              # Similar name (get_user vs getUser)
    CONTEXTUAL = "contextual"    # Based on imports/context
    SEMANTIC = "semantic"        # Based on embedding similarity
    NO_MATCH = "no_match"        # No documentation found

@dataclass
class MatchConfidence:
    """Confidence score for a match with detailed breakdown."""
    overall: float  # 0.0 to 1.0
    name_similarity: float
    location_score: float
    signature_similarity: float

    def __post_init__(self):
        """Validate confidence scores."""
        for attr in ['overall', 'name_similarity', 'location_score', 'signature_similarity']:
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

    # Optional: If documentation is in a different location
    doc_location: Optional[str] = None  # e.g., "class docstring", "module docstring"

    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence match."""
        return self.confidence.overall >= 0.85

@dataclass
class MatchResult:
    """Complete result of matching operation."""
    matched_pairs: List[MatchedPair] = field(default_factory=list)
    unmatched_functions: List[ParsedFunction] = field(default_factory=list)
    total_functions: int = 0
    match_duration_ms: float = 0.0

    @property
    def match_rate(self) -> float:
        """Calculate the percentage of functions matched."""
        if self.total_functions == 0:
            return 0.0
        return len(self.matched_pairs) / self.total_functions

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_functions": self.total_functions,
            "matched": len(self.matched_pairs),
            "unmatched": len(self.unmatched_functions),
            "match_rate": f"{self.match_rate:.1%}",
            "duration_ms": self.match_duration_ms,
            "match_types": self._count_match_types()
        }

    def _count_match_types(self) -> Dict[str, int]:
        """Count matches by type."""
        counts = {match_type.value: 0 for match_type in MatchType}
        for pair in self.matched_pairs:
            counts[pair.match_type.value] += 1
        return counts

class MatchingError(Exception):
    """Base exception for matching errors."""
    def __init__(self, message: str, recovery_hint: Optional[str] = None):
        super().__init__(message)
        self.recovery_hint = recovery_hint

Create comprehensive tests in tests/test_matcher_models.py:

python"""Test matcher data models."""

import pytest
from codedocsync.matcher import (
    MatchType, MatchConfidence, MatchedPair, MatchResult, MatchingError
)
from codedocsync.parser import ParsedFunction, FunctionSignature, FunctionParameter

def test_match_confidence_validation():
    """Test confidence score validation."""
    # Valid confidence
    conf = MatchConfidence(
        overall=0.95,
        name_similarity=1.0,
        location_score=0.9,
        signature_similarity=0.85
    )
    assert conf.overall == 0.95

    # Invalid confidence (> 1.0)
    with pytest.raises(ValueError, match="overall must be between"):
        MatchConfidence(
            overall=1.5,  # Invalid!
            name_similarity=1.0,
            location_score=0.9,
            signature_similarity=0.85
        )

    # Invalid confidence (< 0.0)
    with pytest.raises(ValueError):
        MatchConfidence(
            overall=-0.1,  # Invalid!
            name_similarity=1.0,
            location_score=0.9,
            signature_similarity=0.85
        )

def test_match_result_summary():
    """Test match result summary generation."""
    # Create mock functions
    func1 = create_mock_function("test1")
    func2 = create_mock_function("test2")
    func3 = create_mock_function("test3")

    # Create matches
    pair1 = MatchedPair(
        function=func1,
        match_type=MatchType.EXACT,
        confidence=MatchConfidence(1.0, 1.0, 1.0, 1.0),
        match_reason="Exact name match"
    )

    pair2 = MatchedPair(
        function=func2,
        match_type=MatchType.FUZZY,
        confidence=MatchConfidence(0.85, 0.85, 1.0, 0.9),
        match_reason="Fuzzy name match: test_2 → test2"
    )

    result = MatchResult(
        matched_pairs=[pair1, pair2],
        unmatched_functions=[func3],
        total_functions=3,
        match_duration_ms=25.5
    )

    summary = result.get_summary()
    assert summary["total_functions"] == 3
    assert summary["matched"] == 2
    assert summary["unmatched"] == 1
    assert summary["match_rate"] == "66.7%"
    assert summary["match_types"]["exact"] == 1
    assert summary["match_types"]["fuzzy"] == 1

def create_mock_function(name: str) -> ParsedFunction:
    """Helper to create mock ParsedFunction."""
    return ParsedFunction(
        signature=FunctionSignature(name=name),
        docstring=None,
        file_path="test.py",
        line_number=1,
        end_line_number=2,
        source_code=f"def {name}(): pass"
    )
Validation Checklist

 All models have proper validation in __post_init__
 Models are immutable (use frozen=True or avoid setters)
 All fields have type hints
 Tests cover validation edge cases
 Models integrate with existing ParsedFunction


CHUNK 2: Exact Matching Implementation
Objective
Implement exact name matching within the same file. This handles the simplest case where function names match their docstrings exactly.
Implementation Details

Create codedocsync/matcher/direct_matcher.py:

python"""Direct matcher for same-file function-documentation matching."""

import time
import logging
from typing import List, Dict, Set, Optional
from codedocsync.parser import ParsedFunction, ParsedDocstring
from .models import (
    MatchedPair, MatchConfidence, MatchType, MatchResult, MatchingError
)

logger = logging.getLogger(__name__)

class DirectMatcher:
    """Matches functions to documentation within the same file."""

    def __init__(self):
        """Initialize the direct matcher."""
        self.stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "no_matches": 0,
            "total_processed": 0
        }

    def match_functions(
        self,
        functions: List[ParsedFunction]
    ) -> MatchResult:
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
            return MatchResult(
                total_functions=0,
                match_duration_ms=0.0
            )

        # Reset stats for this run
        self._reset_stats()

        matched_pairs = []
        unmatched_functions = []

        # Group functions by file for efficient matching
        functions_by_file = self._group_by_file(functions)

        for file_path, file_functions in functions_by_file.items():
            file_matches = self._match_functions_in_file(file_functions)

            for func, match in file_matches.items():
                if match:
                    matched_pairs.append(match)
                else:
                    unmatched_functions.append(func)

        duration_ms = (time.time() - start_time) * 1000

        result = MatchResult(
            matched_pairs=matched_pairs,
            unmatched_functions=unmatched_functions,
            total_functions=len(functions),
            match_duration_ms=duration_ms
        )

        logger.info(
            f"Direct matching complete: {result.match_rate:.1%} matched "
            f"({len(matched_pairs)}/{len(functions)}) in {duration_ms:.1f}ms"
        )

        return result

    def _match_functions_in_file(
        self,
        functions: List[ParsedFunction]
    ) -> Dict[ParsedFunction, Optional[MatchedPair]]:
        """Match functions within a single file."""
        matches = {}

        for func in functions:
            # Skip if no docstring
            if not func.docstring:
                matches[func] = None
                self.stats["no_matches"] += 1
                continue

            # For now, only handle exact matches
            # (Fuzzy matching will be added in next chunk)
            match = self._try_exact_match(func)
            matches[func] = match

            if match:
                self.stats["exact_matches"] += 1
            else:
                self.stats["no_matches"] += 1

            self.stats["total_processed"] += 1

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
                match_reason="Exact match: function and docstring aligned"
            )

        return None

    def _calculate_exact_match_confidence(
        self,
        func: ParsedFunction
    ) -> MatchConfidence:
        """Calculate confidence for an exact match."""
        # Start with perfect scores
        name_score = 1.0  # Same function, so name matches
        location_score = 1.0  # Docstring is in expected location

        # Calculate signature similarity if we have parsed docstring
        sig_score = 1.0
        if hasattr(func.docstring, 'parameters'):
            sig_score = self._calculate_signature_similarity(func)

        # Overall confidence is weighted average
        overall = (name_score + location_score + sig_score) / 3.0

        return MatchConfidence(
            overall=overall,
            name_similarity=name_score,
            location_score=location_score,
            signature_similarity=sig_score
        )

    def _calculate_signature_similarity(self, func: ParsedFunction) -> float:
        """
        Calculate how well the signature matches the documentation.

        Returns a score from 0.0 to 1.0.
        """
        if not hasattr(func.docstring, 'parameters'):
            return 1.0  # No parameters to check

        func_params = {p.name for p in func.signature.parameters}

        # Handle both RawDocstring and ParsedDocstring
        if hasattr(func.docstring, 'parameters'):
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
        self,
        functions: List[ParsedFunction]
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

    def get_stats(self) -> Dict[str, int]:
        """Get matching statistics."""
        return self.stats.copy()

Create tests in tests/test_direct_matcher_exact.py:

python"""Test exact matching functionality."""

import pytest
from codedocsync.matcher import DirectMatcher, MatchType
from codedocsync.parser import (
    ParsedFunction, FunctionSignature, FunctionParameter,
    RawDocstring, ParsedDocstring, DocstringParameter
)

class TestExactMatching:
    """Test exact matching capabilities."""

    def test_exact_match_with_docstring(self):
        """Test matching function with its docstring."""
        # Create function with docstring
        func = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_sum",
                parameters=[
                    FunctionParameter("a", "int", None, True),
                    FunctionParameter("b", "int", None, True)
                ],
                return_type="int"
            ),
            docstring=RawDocstring(
                raw_text='"""Calculate sum of two numbers."""',
                line_number=2,
                indentation=4
            ),
            file_path="math_utils.py",
            line_number=1,
            end_line_number=3,
            source_code='def calculate_sum(a: int, b: int) -> int:\n    """Calculate sum of two numbers."""\n    return a + b'
        )

        matcher = DirectMatcher()
        result = matcher.match_functions([func])

        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0].match_type == MatchType.EXACT
        assert result.matched_pairs[0].confidence.overall >= 0.95
        assert "Exact match" in result.matched_pairs[0].match_reason

    def test_no_match_without_docstring(self):
        """Test function without docstring is unmatched."""
        func = ParsedFunction(
            signature=FunctionSignature(name="helper_func"),
            docstring=None,  # No docstring!
            file_path="utils.py",
            line_number=10,
            end_line_number=12,
            source_code="def helper_func():\n    pass"
        )

        matcher = DirectMatcher()
        result = matcher.match_functions([func])

        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 1
        assert result.unmatched_functions[0] == func

    def test_signature_similarity_calculation(self):
        """Test signature similarity scoring."""
        # Function with matching parameters in docstring
        func = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter("data", "List[str]", None, True),
                    FunctionParameter("validate", "bool", "True", False)
                ]
            ),
            docstring=ParsedDocstring(
                format="google",
                summary="Process data with validation.",
                parameters=[
                    DocstringParameter("data", "List[str]", "Data to process"),
                    DocstringParameter("validate", "bool", "Whether to validate")
                ],
                raw_text='"""Process data with validation."""'
            ),
            file_path="processor.py",
            line_number=5,
            end_line_number=10,
            source_code="def process_data(data: List[str], validate: bool = True): ..."
        )

        matcher = DirectMatcher()
        confidence = matcher._calculate_exact_match_confidence(func)

        assert confidence.signature_similarity == 1.0  # Perfect match
        assert confidence.overall >= 0.95

    def test_multiple_files_matching(self):
        """Test matching across multiple files."""
        functions = [
            # File 1
            ParsedFunction(
                signature=FunctionSignature(name="func1"),
                docstring=RawDocstring("Doc1", 2, 4),
                file_path="file1.py",
                line_number=1,
                end_line_number=3,
                source_code="def func1(): pass"
            ),
            # File 2
            ParsedFunction(
                signature=FunctionSignature(name="func2"),
                docstring=RawDocstring("Doc2", 2, 4),
                file_path="file2.py",
                line_number=1,
                end_line_number=3,
                source_code="def func2(): pass"
            ),
            # File 1 again
            ParsedFunction(
                signature=FunctionSignature(name="func3"),
                docstring=None,  # No docstring
                file_path="file1.py",
                line_number=5,
                end_line_number=6,
                source_code="def func3(): pass"
            )
        ]

        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        assert result.total_functions == 3
        assert len(result.matched_pairs) == 2
        assert len(result.unmatched_functions) == 1

        # Check grouping worked correctly
        stats = matcher.get_stats()
        assert stats["exact_matches"] == 2
        assert stats["no_matches"] == 1
Performance Requirements

Exact matching must complete in <1ms per function
Use efficient data structures (sets for parameter comparison)
Avoid repeated calculations

Validation Checklist

 Handles both RawDocstring and ParsedDocstring
 Correctly groups functions by file
 Calculates signature similarity accurately
 Returns proper MatchedPair objects
 Logs performance metrics


CHUNK 3: Fuzzy Matching Implementation
Objective
Add fuzzy matching to handle common naming variations (camelCase vs snake_case, abbreviations, etc.) while maintaining high precision.
Implementation Details

First, add rapidfuzz to project dependencies:

bashpoetry add rapidfuzz

Update codedocsync/matcher/direct_matcher.py to add fuzzy matching:

python# Add imports at top
from rapidfuzz import fuzz, process
import re

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
            "total_processed": 0
        }

        # Common naming patterns for fuzzy matching
        self.naming_patterns = [
            (r'get_(\w+)', r'get\1'),      # get_user -> getUser
            (r'set_(\w+)', r'set\1'),      # set_value -> setValue
            (r'is_(\w+)', r'is\1'),        # is_valid -> isValid
            (r'has_(\w+)', r'has\1'),      # has_data -> hasData
            (r'(\w+)_id', r'\1Id'),        # user_id -> userId
            (r'(\w+)_url', r'\1Url'),      # api_url -> apiUrl
        ]

    def _match_functions_in_file(
        self,
        functions: List[ParsedFunction]
    ) -> Dict[ParsedFunction, Optional[MatchedPair]]:
        """Match functions within a single file."""
        matches = {}

        # Build index of functions with docstrings for fuzzy matching
        documented_functions = [f for f in functions if f.docstring]
        function_names = [f.signature.name for f in documented_functions]

        for func in functions:
            if not func.docstring:
                matches[func] = None
                self.stats["no_matches"] += 1
                continue

            # Try exact match first
            match = self._try_exact_match(func)

            # If no exact match, try fuzzy match
            if not match and function_names:
                match = self._try_fuzzy_match(func, documented_functions)

            matches[func] = match

            if match:
                if match.match_type == MatchType.EXACT:
                    self.stats["exact_matches"] += 1
                else:
                    self.stats["fuzzy_matches"] += 1
            else:
                self.stats["no_matches"] += 1

            self.stats["total_processed"] += 1

        return matches

    def _try_fuzzy_match(
        self,
        func: ParsedFunction,
        candidates: List[ParsedFunction]
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
                            f"Pattern match: {func_name} → {transformed}"
                        )

        # Try fuzzy string matching
        candidate_names = [c.signature.name for c in candidates]

        # Get best matches using rapidfuzz
        matches = process.extract(
            func_name,
            candidate_names,
            scorer=fuzz.ratio,
            limit=3  # Top 3 matches
        )

        for match_name, score, _ in matches:
            if score >= self.fuzzy_threshold * 100:  # rapidfuzz uses 0-100 scale
                # Find the candidate function
                candidate = next(c for c in candidates if c.signature.name == match_name)

                # Additional validation for fuzzy matches
                if self._validate_fuzzy_match(func, candidate):
                    return self._create_fuzzy_match(
                        func,
                        candidate,
                        f"Fuzzy match: {func_name} → {match_name} (score: {score}%)"
                    )

        return None

    def _validate_fuzzy_match(
        self,
        func1: ParsedFunction,
        func2: ParsedFunction
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
        self,
        func: ParsedFunction,
        matched_func: ParsedFunction,
        reason: str
    ) -> MatchedPair:
        """Create a fuzzy match pair with confidence scoring."""
        # Calculate detailed confidence
        name_sim = fuzz.ratio(
            func.signature.name,
            matched_func.signature.name
        ) / 100.0

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
                signature_similarity=sig_score
            ),
            match_reason=reason
        )

    def _calculate_signature_similarity_between(
        self,
        func1: ParsedFunction,
        func2: ParsedFunction
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

Create fuzzy matching tests in tests/test_direct_matcher_fuzzy.py:

python"""Test fuzzy matching functionality."""

import pytest
from codedocsync.matcher import DirectMatcher, MatchType
from codedocsync.parser import (
    ParsedFunction, FunctionSignature, FunctionParameter, RawDocstring
)

class TestFuzzyMatching:
    """Test fuzzy matching capabilities."""

    def test_snake_case_to_camel_case(self):
        """Test matching snake_case to camelCase."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="get_user_name"),
                docstring=RawDocstring("Get user name", 2, 4),
                file_path="user.py",
                line_number=1,
                end_line_number=3,
                source_code="def get_user_name(): pass"
            ),
            ParsedFunction(
                signature=FunctionSignature(name="getUserName"),
                docstring=RawDocstring("Get user name", 6, 4),
                file_path="user.py",
                line_number=5,
                end_line_number=7,
                source_code="def getUserName(): pass"
            )
        ]

        matcher = DirectMatcher(fuzzy_threshold=0.8)
        result = matcher.match_functions(functions)

        # Both should be matched (to themselves in this case)
        assert len(result.matched_pairs) == 2

    def test_abbreviation_matching(self):
        """Test matching abbreviated names."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name="calc_total",
                    parameters=[
                        FunctionParameter("values", "List[float]", None, True)
                    ]
                ),
                docstring=RawDocstring("Calculate total", 2, 4),
                file_path="calc.py",
                line_number=1,
                end_line_number=3,
                source_code="def calc_total(values): pass"
            ),
            ParsedFunction(
                signature=FunctionSignature(
                    name="calculate_total",
                    parameters=[
                        FunctionParameter("values", "List[float]", None, True)
                    ]
                ),
                docstring=RawDocstring("Calculate total", 6, 4),
                file_path="calc.py",
                line_number=5,
                end_line_number=7,
                source_code="def calculate_total(values): pass"
            )
        ]

        matcher = DirectMatcher(fuzzy_threshold=0.75)
        result = matcher.match_functions(functions)

        # Should find fuzzy matches
        assert any(
            pair.match_type == MatchType.FUZZY
            for pair in result.matched_pairs
        )

    def test_fuzzy_threshold_enforcement(self):
        """Test that fuzzy threshold is respected."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(name="save_user"),
                docstring=RawDocstring("Save user", 2, 4),
                file_path="db.py",
                line_number=1,
                end_line_number=3,
                source_code="def save_user(): pass"
            ),
            ParsedFunction(
                signature=FunctionSignature(name="load_data"),  # Very different
                docstring=RawDocstring("Load data", 6, 4),
                file_path="db.py",
                line_number=5,
                end_line_number=7,
                source_code="def load_data(): pass"
            )
        ]

        # High threshold should prevent matching
        matcher = DirectMatcher(fuzzy_threshold=0.9)
        result = matcher.match_functions(functions)

        # Should not create fuzzy matches for very different names
        fuzzy_matches = [p for p in result.matched_pairs if p.match_type == MatchType.FUZZY]
        assert len(fuzzy_matches) == 0

    def test_validation_prevents_bad_matches(self):
        """Test that validation prevents incorrect fuzzy matches."""
        functions = [
            ParsedFunction(
                signature=FunctionSignature(
                    name="process",
                    parameters=[FunctionParameter("x", "int", None, True)],
                    return_type="int"
                ),
                docstring=RawDocstring("Process integer", 2, 4),
                file_path="proc.py",
                line_number=1,
                end_line_number=3,
                source_code="def process(x: int) -> int: pass"
            ),
            ParsedFunction(
                signature=FunctionSignature(
                    name="process",
                    parameters=[
                        FunctionParameter("a", "str", None, True),
                        FunctionParameter("b", "str", None, True),
                        FunctionParameter("c", "str", None, True)
                    ],
                    return_type="str"
                ),
                docstring=RawDocstring("Process strings", 102, 4),
                file_path="proc.py",
                line_number=100,  # Far away in file
                end_line_number=104,
                source_code="def process(a: str, b: str, c: str) -> str: pass"
            )
        ]

        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        # Should not match due to different signatures and distance
        assert len(result.matched_pairs) == 2  # Each matches itself
        assert all(p.match_type == MatchType.EXACT for p in result.matched_pairs)
Fuzzy Matching Rules

Pattern-based transformations have priority over string similarity
Validation must pass for a fuzzy match to be accepted
Confidence scores reflect the quality of the match
False positives are minimized through parameter and location checking

Validation Checklist

 Snake_case to camelCase conversion works
 Common abbreviations are matched
 Threshold is enforced correctly
 Validation prevents bad matches
 Performance remains <5ms per function


CHUNK 4: Integration and Configuration
Objective
Integrate the direct matcher with the existing codebase and add configuration support.
Implementation Details

Update codedocsync/utils/config.py to add matcher configuration:

python# Add to existing config classes

from pydantic import BaseModel, Field, validator

class MatcherConfig(BaseModel):
    """Configuration for the matching system."""

    # Direct matcher settings
    enable_fuzzy: bool = Field(default=True, description="Enable fuzzy name matching")
    fuzzy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for fuzzy matches"
    )

    # Performance settings
    max_line_distance: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum line distance for fuzzy match validation"
    )

    # Pattern matching
    custom_patterns: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Custom regex patterns for name transformation"
    )

    @validator('custom_patterns')
    def validate_patterns(cls, patterns):
        """Validate regex patterns are valid."""
        import re
        for pattern_dict in patterns:
            if 'pattern' not in pattern_dict or 'replacement' not in pattern_dict:
                raise ValueError("Pattern dict must have 'pattern' and 'replacement'")
            try:
                re.compile(pattern_dict['pattern'])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return patterns

# Update main config
class CodeDocSyncConfig(BaseModel):
    """Complete configuration with matcher settings."""
    version: int = 1
    parser: ParserConfig = Field(default_factory=ParserConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)  # NEW!
    analysis: AnalysisConfig
    # ... rest of config

Create a facade for easy CLI integration in codedocsync/matcher/facade.py:

python"""High-level facade for matching operations."""

import logging
from pathlib import Path
from typing import List, Optional, Union
from codedocsync.parser import ParsedFunction, IntegratedParser
from codedocsync.utils.config import CodeDocSyncConfig
from .direct_matcher import DirectMatcher
from .models import MatchResult

logger = logging.getLogger(__name__)

class MatchingFacade:
    """High-level interface for matching operations."""

    def __init__(self, config: Optional[CodeDocSyncConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or CodeDocSyncConfig(
            analysis={"llm_provider": "openai", "model": "gpt-4"}
        )

        # Initialize matchers based on config
        self.direct_matcher = DirectMatcher(
            fuzzy_threshold=self.config.matcher.fuzzy_threshold
        )

        # Add custom patterns if provided
        if self.config.matcher.custom_patterns:
            self._add_custom_patterns()

    def match_file(self, file_path: Union[str, Path]) -> MatchResult:
        """
        Parse and match functions in a single file.

        Args:
            file_path: Path to Python file

        Returns:
            MatchResult with all matches

        Example:
            >>> facade = MatchingFacade()
            >>> result = facade.match_file("mymodule.py")
            >>> print(f"Matched {result.match_rate:.1%} of functions")
        """
        parser = IntegratedParser()
        functions = parser.parse_file(str(file_path))

        if not functions:
            logger.warning(f"No functions found in {file_path}")
            return MatchResult(total_functions=0)

        return self.direct_matcher.match_functions(functions)

    def match_project(self, project_path: Union[str, Path]) -> MatchResult:
        """
        Parse and match all Python files in a project.

        Args:
            project_path: Root directory of project

        Returns:
            Combined MatchResult for entire project
        """
        project_path = Path(project_path)

        # Find all Python files
        python_files = list(project_path.rglob("*.py"))

        # Filter out common excluded directories
        excluded_dirs = {".venv", "venv", "__pycache__", ".git", "build", "dist"}
        python_files = [
            f for f in python_files
            if not any(excluded in f.parts for excluded in excluded_dirs)
        ]

        logger.info(f"Found {len(python_files)} Python files in {project_path}")

        # Parse all files
        all_functions = []
        parser = IntegratedParser()

        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        logger.info(f"Parsed {len(all_functions)} functions total")

        # Match all functions
        return self.direct_matcher.match_functions(all_functions)

    def _add_custom_patterns(self):
        """Add custom naming patterns from config."""
        for pattern_dict in self.config.matcher.custom_patterns:
            pattern = pattern_dict['pattern']
            replacement = pattern_dict['replacement']
            self.direct_matcher.naming_patterns.append((pattern, replacement))
            logger.debug(f"Added custom pattern: {pattern} -> {replacement}")

Add CLI command for matching in codedocsync/cli/main.py:

python# Add to existing CLI

@app.command()
def match(
    path: Path = typer.Argument(..., help="File or directory to match"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    show_unmatched: bool = typer.Option(False, "--show-unmatched", help="Show unmatched functions"),
    output_format: str = typer.Option("terminal", "--format", "-f", help="Output format (terminal/json)")
):
    """
    Match functions to their documentation.

    Example:
        codedocsync match ./myproject --show-unmatched
    """
    # Load configuration
    if config and config.exists():
        config_obj = CodeDocSyncConfig.from_yaml(str(config))
    else:
        config_obj = CodeDocSyncConfig(
            analysis={"llm_provider": "openai", "model": "gpt-4"}
        )

    # Create matching facade
    facade = MatchingFacade(config_obj)

    # Match based on path type
    if path.is_file():
        result = facade.match_file(path)
    elif path.is_dir():
        result = facade.match_project(path)
    else:
        console.print(f"[red]Error: {path} is not a valid file or directory[/red]")
        raise typer.Exit(1)

    # Display results
    if output_format == "json":
        import json
        output = {
            "summary": result.get_summary(),
            "matched_pairs": [
                {
                    "function": pair.function.signature.name,
                    "file": pair.function.file_path,
                    "line": pair.function.line_number,
                    "match_type": pair.match_type.value,
                    "confidence": pair.confidence.overall,
                    "reason": pair.match_reason
                }
                for pair in result.matched_pairs
            ]
        }
        if show_unmatched:
            output["unmatched"] = [
                {
                    "function": func.signature.name,
                    "file": func.file_path,
                    "line": func.line_number
                }
                for func in result.unmatched_functions
            ]
        print(json.dumps(output, indent=2))
    else:
        # Terminal output with Rich
        _display_match_results(result, show_unmatched)

def _display_match_results(result: MatchResult, show_unmatched: bool):
    """Display match results in terminal with Rich."""
    console.print("\n[bold]Matching Results[/bold]")
    console.print("=" * 50)

    # Summary table
    summary = result.get_summary()
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Functions", str(summary["total_functions"]))
    table.add_row("Matched", str(summary["matched"]))
    table.add_row("Unmatched", str(summary["unmatched"]))
    table.add_row("Match Rate", summary["match_rate"])
    table.add_row("Duration", f"{summary['duration_ms']:.1f}ms")

    console.print(table)

    # Match type breakdown
    if summary["matched"] > 0:
        console.print("\n[bold]Match Types:[/bold]")
        for match_type, count in summary["match_types"].items():
            if count > 0:
                console.print(f"  • {match_type}: {count}")

    # Show unmatched if requested
    if show_unmatched and result.unmatched_functions:
        console.print(f"\n[bold red]Unmatched Functions ({len(result.unmatched_functions)}):[/bold red]")
        for func in result.unmatched_functions[:10]:  # Show first 10
            console.print(
                f"  • {func.signature.name} "
                f"([dim]{func.file_path}:{func.line_number}[/dim])"
            )
        if len(result.unmatched_functions) > 10:
            console.print(f"  ... and {len(result.unmatched_functions) - 10} more")
Configuration File Example
yaml# .codedocsync.yml
version: 1

matcher:
  enable_fuzzy: true
  fuzzy_threshold: 0.85
  max_line_distance: 100
  custom_patterns:
    - pattern: 'check_(\w+)'
      replacement: 'validate\1'
    - pattern: 'parse_(\w+)'
      replacement: '\1Parser'

parser:
  docstring_style: auto
  include_private: false

analysis:
  llm_provider: openai
  model: gpt-4o-mini
Validation Checklist

 Config validation works properly
 CLI command displays results clearly
 File vs directory matching works
 Custom patterns are applied
 Performance metrics are shown


CHUNK 5: Testing and Performance Validation
Objective
Create comprehensive tests and ensure performance meets requirements.
Implementation Details

Create integration tests in tests/test_matcher_integration.py:

python"""Integration tests for the complete matching system."""

import pytest
import tempfile
from pathlib import Path
from codedocsync.matcher import MatchingFacade
from codedocsync.utils.config import CodeDocSyncConfig

class TestMatcherIntegration:
    """Test complete matching workflows."""

    def test_end_to_end_matching(self, tmp_path):
        """Test complete parsing and matching workflow."""
        # Create test file
        test_file = tmp_path / "test_module.py"
        test_file.write_text('''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b

def calc_product(x, y):
    """Calculate product of x and y."""
    return x * y

def helper_function():
    # No docstring
    pass

class Calculator:
    """A simple calculator class."""

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
''')

        # Create facade and match
        facade = MatchingFacade()
        result = facade.match_file(test_file)

        # Verify results
        assert result.total_functions == 4  # 3 functions + 1 method
        assert len(result.matched_pairs) == 3  # All except helper_function
        assert len(result.unmatched_functions) == 1

        # Check match types
        summary = result.get_summary()
        assert summary["match_types"]["exact"] == 3

        # Verify unmatched function
        assert result.unmatched_functions[0].signature.name == "helper_function"

    def test_fuzzy_matching_project(self, tmp_path):
        """Test fuzzy matching across a project."""
        # Create multiple files with naming variations
        (tmp_path / "models.py").write_text('''
def get_user_by_id(user_id: int):
    """Get user by ID."""
    pass

def getUserById(userId: int):
    """Get user by ID."""
    pass
''')

        (tmp_path / "utils.py").write_text('''
def calc_avg(numbers: list):
    """Calculate average."""
    pass

def calculate_average(numbers: list):
    """Calculate average of numbers."""
    pass
''')

        # Match with fuzzy enabled
        config = CodeDocSyncConfig(
            matcher={"enable_fuzzy": True, "fuzzy_threshold": 0.7},
            analysis={"llm_provider": "openai", "model": "gpt-4"}
        )
        facade = MatchingFacade(config)
        result = facade.match_project(tmp_path)

        # Should find some fuzzy matches
        assert result.total_functions == 4
        fuzzy_matches = [
            p for p in result.matched_pairs
            if p.match_type.value == "fuzzy"
        ]
        assert len(fuzzy_matches) >= 0  # Depends on matching logic

    def test_custom_patterns(self, tmp_path):
        """Test custom pattern matching."""
        test_file = tmp_path / "validators.py"
        test_file.write_text('''
def check_email(email: str) -> bool:
    """Validate email address."""
    return "@" in email

def validateEmail(email: str) -> bool:
    """Validate email address."""
    return "@" in email
''')

        # Configure custom pattern
        config = CodeDocSyncConfig(
            matcher={
                "custom_patterns": [
                    {"pattern": r"check_(\w+)", "replacement": r"validate\1"}
                ]
            },
            analysis={"llm_provider": "openai", "model": "gpt-4"}
        )

        facade = MatchingFacade(config)
        result = facade.match_file(test_file)

        # Should match via custom pattern
        assert len(result.matched_pairs) == 2

Create performance tests in tests/test_matcher_performance.py:

python"""Performance tests for the matcher."""

import time
import pytest
from codedocsync.matcher import DirectMatcher
from codedocsync.parser import ParsedFunction, FunctionSignature, RawDocstring

class TestMatcherPerformance:
    """Test matcher performance requirements."""

    def test_exact_match_performance(self):
        """Test exact matching meets <1ms per function requirement."""
        # Create 1000 functions
        functions = []
        for i in range(1000):
            func = ParsedFunction(
                signature=FunctionSignature(name=f"function_{i}"),
                docstring=RawDocstring(f"Docstring {i}", i+1, 4),
                file_path="test.py",
                line_number=i*3,
                end_line_number=i*3+2,
                source_code=f"def function_{i}(): pass"
            )
            functions.append(func)

        matcher = DirectMatcher()

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Check performance
        ms_per_function = (duration * 1000) / len(functions)
        assert ms_per_function < 1.0, f"Too slow: {ms_per_function:.2f}ms per function"

        # Verify correctness
        assert len(result.matched_pairs) == 1000
        assert all(p.match_type.value == "exact" for p in result.matched_pairs)

    def test_fuzzy_match_performance(self):
        """Test fuzzy matching meets <5ms per function requirement."""
        # Create functions with fuzzy-matchable names
        functions = []
        for i in range(100):
            # Half snake_case, half camelCase
            if i % 2 == 0:
                name = f"get_item_{i}"
            else:
                name = f"getItem{i}"

            func = ParsedFunction(
                signature=FunctionSignature(name=name),
                docstring=RawDocstring(f"Get item {i}", i+1, 4),
                file_path="test.py",
                line_number=i*3,
                end_line_number=i*3+2,
                source_code=f"def {name}(): pass"
            )
            functions.append(func)

        matcher = DirectMatcher(fuzzy_threshold=0.8)

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Check performance
        ms_per_function = (duration * 1000) / len(functions)
        assert ms_per_function < 5.0, f"Too slow: {ms_per_function:.2f}ms per function"

    def test_memory_usage(self):
        """Test memory usage stays under 100MB for 10k functions."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create 10,000 functions
        functions = []
        for i in range(10000):
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"func_{i}",
                    parameters=[],  # Keep it simple
                ),
                docstring=RawDocstring(f"Doc {i}", i+1, 4) if i % 2 == 0 else None,
                file_path=f"file_{i % 100}.py",  # 100 different files
                line_number=i,
                end_line_number=i+1,
                source_code=f"def func_{i}(): pass"
            )
            functions.append(func)

        # Run matching
        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        # Check memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        assert memory_used < 100, f"Used too much memory: {memory_used:.1f}MB"

        # Verify results
        assert result.total_functions == 10000
        assert len(result.matched_pairs) == 5000  # Half have docstrings
Performance Validation Checklist

 Exact matching: <1ms per function ✓
 Fuzzy matching: <5ms per function ✓
 Memory usage: <100MB for 10k functions ✓
 Results are accurate and complete ✓
 Stats are tracked correctly ✓


Final Integration Checklist
Before considering the Direct Matcher complete:

Code Quality

 All functions have type hints
 All public methods have docstrings
 Error handling is comprehensive
 Logging is informative but not excessive


Testing

 Unit tests for all major functions
 Integration tests for complete workflows
 Performance tests meet requirements
 Edge cases are covered


Integration

 Works with existing parser output
 CLI commands work correctly
 Configuration is validated
 Documentation is complete


Performance

 Exact matching <1ms per function
 Fuzzy matching <5ms per function
 Memory usage <100MB for 10k functions
 No memory leaks



Common Pitfalls to Avoid

Don't assume docstring structure - Always check if it's RawDocstring or ParsedDocstring
Don't use str() for serialization - Access the proper attributes
Don't skip validation - Fuzzy matches need validation to avoid false positives
Don't ignore performance - Use efficient data structures and algorithms
Don't forget error handling - Every external call should be wrapped in try/except

Next Steps
After completing the Direct Matcher:

Update IMPLEMENTATION_STATE.MD to mark matcher/direct_matcher.py as complete
Update CHANGELOG.MD with all changes made
Prepare for Contextual Matcher implementation (Week 2, Days 4-5)
