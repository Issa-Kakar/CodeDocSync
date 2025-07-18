"""Test matcher data models."""

import pytest
from codedocsync.matcher import MatchType, MatchConfidence, MatchedPair, MatchResult
from codedocsync.parser import ParsedFunction, FunctionSignature


def test_match_confidence_validation():
    """Test confidence score validation."""
    # Valid confidence
    conf = MatchConfidence(
        overall=0.95, name_similarity=1.0, location_score=0.9, signature_similarity=0.85
    )
    assert conf.overall == 0.95

    # Invalid confidence (> 1.0)
    with pytest.raises(ValueError, match="overall must be between"):
        MatchConfidence(
            overall=1.5,  # Invalid!
            name_similarity=1.0,
            location_score=0.9,
            signature_similarity=0.85,
        )

    # Invalid confidence (< 0.0)
    with pytest.raises(ValueError):
        MatchConfidence(
            overall=-0.1,  # Invalid!
            name_similarity=1.0,
            location_score=0.9,
            signature_similarity=0.85,
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
        match_reason="Exact name match",
    )

    pair2 = MatchedPair(
        function=func2,
        match_type=MatchType.FUZZY,
        confidence=MatchConfidence(0.85, 0.85, 1.0, 0.9),
        match_reason="Fuzzy name match: test_2 → test2",
    )

    result = MatchResult(
        matched_pairs=[pair1, pair2],
        unmatched_functions=[func3],
        total_functions=3,
        match_duration_ms=25.5,
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
        source_code=f"def {name}(): pass",
    )
