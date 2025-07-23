"""
Tests for JSON output formatter.

Tests the JSON formatting functionality for suggestions,
including structured output, metadata inclusion, and batch processing.
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import pytest

from codedocsync.suggestions.formatters.json_formatter import (
    OUTPUT_FORMAT_VERSION,
    JSONSuggestionFormatter,
    analysis_result_to_json,
    batch_results_to_json,
    suggestion_batch_to_json,
    suggestion_to_json,
)
from codedocsync.suggestions.integration import EnhancedAnalysisResult, EnhancedIssue
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionBatch,
    SuggestionDiff,
    SuggestionMetadata,
    SuggestionType,
)


# Test fixtures
@pytest.fixture
def basic_suggestion() -> Any:
    """Create a basic suggestion for testing."""
    diff = SuggestionDiff(
        original_lines=[
            "def func(email: str):",
            '    """Function with email param."""',
        ],
        suggested_lines=[
            "def func(username: str):",
            '    """Function with username param."""',
        ],
        start_line=1,
        end_line=2,
    )

    return Suggestion(
        suggestion_type=SuggestionType.PARAMETER_UPDATE,
        original_text='def func(email: str):\n    """Function with email param."""',
        suggested_text='def func(username: str):\n    """Function with username param."""',
        confidence=0.9,
        diff=diff,
        style="google",
        copy_paste_ready=True,
    )


@pytest.fixture
def suggestion_with_diff() -> Any:
    """Create a suggestion with diff information."""
    diff = SuggestionDiff(
        original_lines=[
            "def func(email: str):",
            '    """Function with email param."""',
        ],
        suggested_lines=[
            "def func(username: str):",
            '    """Function with username param."""',
        ],
        start_line=1,
        end_line=2,
    )

    return Suggestion(
        suggestion_type=SuggestionType.PARAMETER_UPDATE,
        original_text='def func(email: str):\n    """Function with email param."""',
        suggested_text='def func(username: str):\n    """Function with username param."""',
        confidence=0.85,
        style="google",
        copy_paste_ready=True,
        diff=diff,
    )


@pytest.fixture
def suggestion_with_metadata() -> Any:
    """Create a suggestion with metadata."""
    metadata = SuggestionMetadata(
        generator_type="ParameterGenerator",
        generation_time_ms=25.5,
        llm_used=False,
    )

    diff = SuggestionDiff(
        original_lines=['"""Returns something."""'],
        suggested_lines=['"""Returns the processed result."""'],
        start_line=1,
        end_line=1,
    )

    return Suggestion(
        suggestion_type=SuggestionType.RETURN_UPDATE,
        original_text='"""Returns something."""',
        suggested_text='"""Returns the processed result."""',
        confidence=0.8,
        diff=diff,
        style="google",
        copy_paste_ready=True,
        metadata=metadata,
    )


@pytest.fixture
def enhanced_issue(basic_suggestion: Any) -> Any:
    """Create an enhanced issue with suggestion."""
    return EnhancedIssue(
        issue_type="parameter_name_mismatch",
        severity="critical",
        description="Parameter 'email' doesn't match 'username' in code",
        suggestion="Update parameter name",
        line_number=45,
        confidence=0.9,
        details={"old_name": "email", "new_name": "username"},
        rich_suggestion=basic_suggestion,
        ranking_score=8.5,
    )


@pytest.fixture
def mock_function() -> Any:
    """Create a mock ParsedFunction."""
    function: Mock = Mock()
    function.signature = Mock()
    function.signature.name = "authenticate_user"
    function.signature.parameters = []
    function.signature.return_annotation = "bool"
    function.file_path = "auth/user.py"
    function.line_number = 42
    return function


@pytest.fixture
def mock_documentation() -> Any:
    """Create a mock ParsedDocstring."""
    doc: Mock = Mock()
    doc.format = "google"
    doc.summary = "Authenticate user credentials"
    doc.parameters = []
    doc.returns = Mock()
    doc.raises = []
    return doc


@pytest.fixture
def enhanced_result(
    enhanced_issue: Any, mock_function: Any, mock_documentation: Any
) -> Any:
    """Create an enhanced analysis result."""
    mock_pair: Mock = Mock()
    mock_pair.function = mock_function
    mock_pair.documentation = mock_documentation
    mock_pair.confidence = Mock()
    mock_pair.confidence.value = 0.8
    mock_pair.match_type = Mock()
    mock_pair.match_type.value = "direct"
    mock_pair.match_reason = "exact name match"

    return EnhancedAnalysisResult(
        matched_pair=mock_pair,
        issues=[enhanced_issue],
        used_llm=False,
        analysis_time_ms=25.5,
        suggestion_generation_time_ms=15.2,
        suggestions_generated=1,
        suggestions_skipped=0,
        cache_hit=True,
    )


@pytest.fixture
def suggestion_batch(basic_suggestion: Any) -> Any:
    """Create a suggestion batch."""
    return SuggestionBatch(
        suggestions=[basic_suggestion],
        function_name="func",
        file_path="test.py",
        total_generation_time_ms=50.0,
    )


class TestJSONSuggestionFormatterInit:
    """Test JSONSuggestionFormatter initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        formatter = JSONSuggestionFormatter()

        assert formatter.indent == 2
        assert formatter.include_metadata
        assert formatter.include_timestamps

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        formatter = JSONSuggestionFormatter(
            indent=4,
            include_metadata=False,
            include_timestamps=False,
        )

        assert formatter.indent == 4
        assert not formatter.include_metadata
        assert not formatter.include_timestamps


class TestSuggestionFormatting:
    """Test individual suggestion formatting."""

    def test_format_basic_suggestion(self, basic_suggestion: Any) -> None:
        """Test formatting basic suggestion."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion(basic_suggestion)

        assert result["suggestion_type"] == "parameter_update"
        assert result["original_text"] == basic_suggestion.original_text
        assert result["suggested_text"] == basic_suggestion.suggested_text
        assert result["confidence"] == 0.9
        assert result["style"] == "google"
        assert result["copy_paste_ready"]
        assert "generated_at" in result  # Timestamp should be included

    def test_format_suggestion_without_timestamps(self, basic_suggestion: Any) -> None:
        """Test formatting without timestamps."""
        formatter = JSONSuggestionFormatter(include_timestamps=False)
        result = formatter.format_suggestion(basic_suggestion)

        assert "generated_at" not in result

    def test_format_suggestion_with_diff(self, suggestion_with_diff: Any) -> None:
        """Test formatting suggestion with diff information."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion(suggestion_with_diff)

        assert "diff" in result
        diff_data = result["diff"]
        assert "original_lines" in diff_data
        assert "suggested_lines" in diff_data
        assert "start_line" in diff_data
        assert "end_line" in diff_data
        assert "unified_diff" in diff_data
        assert "lines_changed" in diff_data
        assert "additions" in diff_data
        assert "deletions" in diff_data

        assert diff_data["start_line"] == 1
        assert diff_data["end_line"] == 2

    def test_format_suggestion_with_metadata(
        self, suggestion_with_metadata: Any
    ) -> None:
        """Test formatting suggestion with metadata."""
        formatter = JSONSuggestionFormatter(include_metadata=True)
        result = formatter.format_suggestion(suggestion_with_metadata)

        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata["generation_time_ms"] == 25.5
        assert metadata["generator_used"] == "ParameterGenerator"
        assert not metadata["llm_used"]
        assert metadata["cache_hit"]
        assert metadata["validation_passed"]
        assert metadata["quality_score"] == 0.88

    def test_format_suggestion_without_metadata(
        self, suggestion_with_metadata: Any
    ) -> None:
        """Test formatting suggestion without metadata."""
        formatter = JSONSuggestionFormatter(include_metadata=False)
        result = formatter.format_suggestion(suggestion_with_metadata)

        assert "metadata" not in result

    def test_format_suggestion_without_diff(self, basic_suggestion: Any) -> None:
        """Test formatting suggestion without diff."""
        # Ensure no diff
        basic_suggestion.diff = None

        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion(basic_suggestion)

        assert "diff" not in result


class TestEnhancedIssueFormatting:
    """Test enhanced issue formatting."""

    def test_format_enhanced_issue(self, enhanced_issue: Any) -> None:
        """Test formatting enhanced issue."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_enhanced_issue(enhanced_issue)

        assert result["issue_type"] == "parameter_name_mismatch"
        assert result["severity"] == "critical"
        assert result["description"] == enhanced_issue.description
        assert result["basic_suggestion"] == enhanced_issue.suggestion
        assert result["line_number"] == 45
        assert result["confidence"] == 0.9
        assert result["details"] == {"old_name": "email", "new_name": "username"}
        assert result["ranking_score"] == 8.5
        assert "rich_suggestion" in result

    def test_format_issue_without_rich_suggestion(self) -> None:
        """Test formatting issue without rich suggestion."""
        issue = EnhancedIssue(
            issue_type="parameter_missing",
            severity="high",
            description="Missing parameter",
            suggestion="Add parameter",
            line_number=30,
            confidence=0.8,
        )

        formatter = JSONSuggestionFormatter()
        result = formatter.format_enhanced_issue(issue)

        assert "rich_suggestion" not in result
        assert result["issue_type"] == "parameter_missing"

    def test_format_issue_without_ranking_score(self) -> None:
        """Test formatting issue without ranking score."""
        issue = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="Test issue",
            suggestion="Test suggestion",
            line_number=10,
            confidence=0.7,
            # ranking_score is None by default
        )

        formatter = JSONSuggestionFormatter()
        result = formatter.format_enhanced_issue(issue)

        assert "ranking_score" not in result


class TestAnalysisResultFormatting:
    """Test analysis result formatting."""

    def test_format_analysis_result(self, enhanced_result: Any) -> None:
        """Test formatting complete analysis result."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_analysis_result(enhanced_result)

        # Function information
        assert result["function"]["name"] == "authenticate_user"
        assert result["function"]["line_number"] == 42
        assert result["file_path"] == "auth/user.py"

        # Match information
        assert result["match_confidence"] == 0.8
        assert result["match_type"] == "direct"
        assert result["match_reason"] == "exact name match"

        # Analysis information
        analysis = result["analysis"]
        assert not analysis["used_llm"]
        assert analysis["analysis_time_ms"] == 25.5
        assert analysis["suggestion_generation_time_ms"] == 15.2
        assert analysis["total_time_ms"] == 40.7
        assert analysis["cache_hit"]
        assert analysis["suggestions_generated"] == 1
        assert analysis["suggestions_skipped"] == 0

        # Issues
        assert len(result["issues"]) == 1

        # Summary
        summary = result["summary"]
        assert summary["total_issues"] == 1
        assert summary["has_suggestions"]
        assert summary["critical_issues"] == 1
        assert summary["high_issues"] == 0

        # Documentation info
        assert "documentation" in result
        doc = result["documentation"]
        assert doc["format"] == "google"
        assert doc["summary"] == "Authenticate user credentials"

    def test_format_result_without_documentation(self, enhanced_result: Any) -> None:
        """Test formatting result without documentation."""
        enhanced_result.matched_pair.documentation = None

        formatter = JSONSuggestionFormatter()
        result = formatter.format_analysis_result(enhanced_result)

        assert "documentation" not in result

    def test_format_result_with_metadata(self, enhanced_result: Any) -> None:
        """Test formatting result with metadata."""
        formatter = JSONSuggestionFormatter(include_metadata=True)
        result = formatter.format_analysis_result(enhanced_result)

        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata["format_version"] == OUTPUT_FORMAT_VERSION
        assert metadata["processor"] == "CodeDocSync SuggestionFormatter"
        assert "generated_at" in metadata

    def test_format_result_without_metadata(self, enhanced_result: Any) -> None:
        """Test formatting result without metadata."""
        formatter = JSONSuggestionFormatter(include_metadata=False)
        result = formatter.format_analysis_result(enhanced_result)

        assert "metadata" not in result


class TestBatchFormatting:
    """Test batch formatting functionality."""

    def test_format_batch_results(self, enhanced_result: Any) -> None:
        """Test formatting batch of analysis results."""
        results = [enhanced_result]

        formatter = JSONSuggestionFormatter()
        result = formatter.format_batch_results(results)

        assert "results" in result
        assert len(result["results"]) == 1

        # Summary information
        summary = result["summary"]
        assert summary["total_functions"] == 1
        assert summary["functions_with_issues"] == 1
        assert summary["functions_without_issues"] == 0
        assert summary["total_issues"] == 1
        assert summary["total_suggestions"] == 1
        assert summary["cache_hit_rate"] == 1.0
        assert summary["llm_usage_rate"] == 0.0

        # Severity breakdown
        severity_breakdown = summary["severity_breakdown"]
        assert severity_breakdown["critical"] == 1
        assert severity_breakdown["high"] == 0

    def test_format_empty_batch(self) -> None:
        """Test formatting empty batch."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_batch_results([])

        assert result["results"] == []
        summary = result["summary"]
        assert summary["total_functions"] == 0
        assert summary["total_issues"] == 0
        assert summary["average_analysis_time_ms"] == 0

    def test_format_suggestion_batch(self, suggestion_batch: Any) -> None:
        """Test formatting suggestion batch."""
        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion_batch(suggestion_batch)

        assert len(result["suggestions"]) == 1

        summary = result["summary"]
        assert summary["total_suggestions"] == 1
        assert summary["total_issues"] == 2
        assert summary["functions_processed"] == 1
        assert summary["generation_time_ms"] == 50.0

        # Confidence statistics
        confidence_stats = summary["confidence_stats"]
        assert confidence_stats["average"] == 0.9
        assert confidence_stats["min"] == 0.9
        assert confidence_stats["max"] == 0.9
        assert confidence_stats["high_confidence_count"] == 1

        # Suggestion type breakdown
        suggestion_types = summary["suggestion_types"]
        assert suggestion_types["parameter_update"] == 1


class TestJSONSerialization:
    """Test JSON string serialization."""

    def test_to_json_string(self, basic_suggestion: Any) -> None:
        """Test converting to JSON string."""
        formatter = JSONSuggestionFormatter(indent=2)
        data = formatter.format_suggestion(basic_suggestion)
        json_string = formatter.to_json_string(data)

        # Should be valid JSON
        parsed = json.loads(json_string)
        assert parsed["suggestion_type"] == "parameter_update"

        # Should be properly indented
        assert "  " in json_string  # 2-space indentation

    def test_to_json_string_no_indent(self, basic_suggestion: Any) -> None:
        """Test converting to JSON string without indentation."""
        formatter = JSONSuggestionFormatter(indent=None)
        data = formatter.format_suggestion(basic_suggestion)
        json_string = formatter.to_json_string(data)

        # Should be valid JSON
        parsed = json.loads(json_string)
        assert parsed["suggestion_type"] == "parameter_update"

        # Should be compact (no indentation)
        assert "\n" not in json_string


class TestFunctionInformationExtraction:
    """Test function information extraction."""

    def test_format_function_info_complete(self, mock_function: Any) -> None:
        """Test extracting complete function information."""
        # Add parameter to function
        param: Mock = Mock()
        param.name = "username"
        param.is_required = True
        param.type_annotation = "str"
        param.default_value = None
        mock_function.signature.parameters = [param]

        formatter = JSONSuggestionFormatter()
        info = formatter._format_function_info(mock_function)

        assert info["name"] == "authenticate_user"
        assert info["line_number"] == 42
        assert len(info["parameters"]) == 1
        assert info["parameters"][0]["name"] == "username"
        assert info["parameters"][0]["is_required"]
        assert info["parameters"][0]["type_annotation"] == "str"
        assert info["return_annotation"] == "bool"

    def test_format_function_info_minimal(self) -> None:
        """Test extracting minimal function information."""
        minimal_function: Mock = Mock()
        # No signature attribute

        formatter = JSONSuggestionFormatter()
        info = formatter._format_function_info(minimal_function)

        assert info["name"] == "Unknown"
        assert info["line_number"] == 0
        assert info["parameters"] == []


class TestBatchSummaryStatistics:
    """Test batch summary statistics calculation."""

    def test_create_batch_summary_multiple_results(self, enhanced_result: Any) -> None:
        """Test batch summary with multiple results."""
        # Create second result
        mock_function2: Mock = Mock()
        mock_function2.signature = Mock()
        mock_function2.signature.name = "second_function"
        mock_function2.file_path = "other/file.py"

        mock_pair2: Mock = Mock()
        mock_pair2.function = mock_function2

        result2 = EnhancedAnalysisResult(
            matched_pair=mock_pair2,
            issues=[],  # No issues
            used_llm=True,
            analysis_time_ms=50.0,
            cache_hit=False,
        )

        results = [enhanced_result, result2]

        formatter = JSONSuggestionFormatter()
        summary = formatter._create_batch_summary(results)

        assert summary["total_functions"] == 2
        assert summary["functions_with_issues"] == 1
        assert summary["functions_without_issues"] == 1
        assert summary["cache_hit_rate"] == 0.5  # 1 out of 2
        assert summary["llm_usage_rate"] == 0.5  # 1 out of 2
        assert summary["average_analysis_time_ms"] == 37.75  # (25.5 + 50.0) / 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_suggestion_to_json(self, basic_suggestion: Any) -> None:
        """Test suggestion_to_json convenience function."""
        json_string = suggestion_to_json(basic_suggestion, indent=4)

        parsed = json.loads(json_string)
        assert parsed["suggestion_type"] == "parameter_update"

        # Should use custom indent
        assert "    " in json_string  # 4-space indentation

    def test_analysis_result_to_json(self, enhanced_result: Any) -> None:
        """Test analysis_result_to_json convenience function."""
        json_string = analysis_result_to_json(enhanced_result)

        parsed = json.loads(json_string)
        assert parsed["function"]["name"] == "authenticate_user"

    def test_batch_results_to_json(self, enhanced_result: Any) -> None:
        """Test batch_results_to_json convenience function."""
        json_string = batch_results_to_json([enhanced_result])

        parsed = json.loads(json_string)
        assert len(parsed["results"]) == 1
        assert "summary" in parsed

    def test_suggestion_batch_to_json(self, suggestion_batch: Any) -> None:
        """Test suggestion_batch_to_json convenience function."""
        json_string = suggestion_batch_to_json(suggestion_batch)

        parsed = json.loads(json_string)
        assert len(parsed["suggestions"]) == 1
        assert "summary" in parsed


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_format_suggestion_with_none_values(self) -> None:
        """Test formatting suggestion with None values."""
        diff = SuggestionDiff(
            original_lines=["original"],
            suggested_lines=["suggested"],
            start_line=1,
            end_line=1,
        )

        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="original",
            suggested_text="suggested",
            confidence=0.8,
            style="google",
            copy_paste_ready=True,
            diff=diff,
        )

        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion(suggestion)

        # Should format properly
        assert "diff" in result
        assert isinstance(result, dict)

    def test_format_issue_with_empty_details(self) -> None:
        """Test formatting issue with empty details."""
        issue = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="test",
            suggestion="test",
            line_number=1,
            details={},  # Empty details
        )

        formatter = JSONSuggestionFormatter()
        result = formatter.format_enhanced_issue(issue)

        assert result["details"] == {}

    def test_batch_summary_with_no_suggestions(self) -> None:
        """Test batch summary when no suggestions have confidence stats."""
        batch = SuggestionBatch(
            suggestions=[],  # No suggestions
            function_name="test_function",
            file_path="test.py",
            total_generation_time_ms=100.0,
        )

        formatter = JSONSuggestionFormatter()
        result = formatter.format_suggestion_batch(batch)

        # Should handle empty suggestions gracefully
        assert "confidence_stats" not in result["summary"]
        assert "suggestion_types" not in result["summary"]


class TestTimestampHandling:
    """Test timestamp handling."""

    def test_timestamp_format(self, basic_suggestion: Any) -> None:
        """Test timestamp format."""
        formatter = JSONSuggestionFormatter(include_timestamps=True)
        result = formatter.format_suggestion(basic_suggestion)

        # Should include ISO format timestamp
        timestamp = result["generated_at"]
        # Should be parseable as datetime
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed_time, datetime)


if __name__ == "__main__":
    pytest.main([__file__])
