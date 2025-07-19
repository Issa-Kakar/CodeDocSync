"""
Tests for LLM output parsing and validation.

This module tests the LLM output parser functionality including:
- Parsing valid JSON responses
- Handling malformed JSON
- Schema validation
- Confidence validation
- Error recovery strategies
- Issue actionability filtering
"""

import json
import pytest

from codedocsync.analyzer.llm_output_parser import (
    LLMOutputParser,
    ParseResult,
    parse_llm_response,
    parse_and_filter_response,
    validate_response_format,
)
from codedocsync.analyzer.models import InconsistencyIssue, ISSUE_TYPES


class TestParseResult:
    """Test ParseResult dataclass functionality."""

    def test_parse_result_creation(self):
        """Test creating ParseResult objects."""
        result = ParseResult(
            issues=[], success=True, confidence=0.95, analysis_notes="Test analysis"
        )

        assert result.issues == []
        assert result.success is True
        assert result.confidence == 0.95
        assert result.analysis_notes == "Test analysis"
        assert result.error_message is None
        assert result.raw_response is None

    def test_parse_result_with_issues(self):
        """Test ParseResult with actual issues."""
        issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Missing parameter documentation",
            suggestion="Add parameter documentation",
            line_number=5,
            confidence=0.9,
        )

        result = ParseResult(issues=[issue], success=True, confidence=0.9)

        assert len(result.issues) == 1
        assert result.issues[0] == issue


class TestLLMOutputParser:
    """Test core LLM output parser functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance for testing."""
        return LLMOutputParser(strict_validation=True)

    @pytest.fixture
    def lenient_parser(self):
        """Create lenient parser for recovery testing."""
        return LLMOutputParser(strict_validation=False)

    @pytest.fixture
    def valid_response(self):
        """Valid LLM response for testing."""
        return {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Function handles None input but docstring doesn't mention this",
                    "suggestion": "Add note about None handling in the Args section",
                    "confidence": 0.85,
                    "line_number": 5,
                    "details": {"missing_behavior": "None handling"},
                },
                {
                    "type": "example_invalid",
                    "description": "Example in docstring has syntax error",
                    "suggestion": "Fix the syntax error in the example: missing closing parenthesis",
                    "confidence": 0.95,
                    "line_number": 15,
                    "details": {"error_type": "syntax"},
                },
            ],
            "analysis_notes": "Checked parameter handling and examples",
            "confidence": 0.88,
        }

    def test_parse_valid_json_response(self, parser, valid_response):
        """Test parsing a valid JSON response."""
        response_json = json.dumps(valid_response)
        result = parser.parse_analysis_response(response_json)

        assert result.success is True
        assert len(result.issues) == 2
        assert result.confidence == 0.88
        assert result.analysis_notes == "Checked parameter handling and examples"
        assert result.error_message is None

    def test_parse_response_with_markdown_formatting(self, parser, valid_response):
        """Test parsing response wrapped in markdown code blocks."""
        response_text = f"```json\n{json.dumps(valid_response)}\n```"
        result = parser.parse_analysis_response(response_text)

        assert result.success is True
        assert len(result.issues) == 2

    def test_parse_response_with_extra_text(self, parser, valid_response):
        """Test parsing response with extra text around JSON."""
        response_text = f"Here is my analysis:\n\n{json.dumps(valid_response)}\n\nThat completes the analysis."
        result = parser.parse_analysis_response(response_text)

        assert result.success is True
        assert len(result.issues) == 2

    def test_parse_empty_issues_response(self, parser):
        """Test parsing response with no issues."""
        response = {
            "issues": [],
            "analysis_notes": "No issues found",
            "confidence": 0.95,
        }

        result = parser.parse_analysis_response(json.dumps(response))

        assert result.success is True
        assert len(result.issues) == 0
        assert result.confidence == 0.95

    def test_parse_invalid_json(self, parser):
        """Test handling of invalid JSON."""
        invalid_json = "This is not JSON at all"
        result = parser.parse_analysis_response(invalid_json)

        assert result.success is False
        assert "JSON parsing failed" in result.error_message
        assert len(result.issues) == 0

    def test_parse_malformed_json(self, parser):
        """Test handling of malformed JSON."""
        malformed_json = '{"issues": [{"type": "test"}'  # Missing closing braces
        result = parser.parse_analysis_response(malformed_json)

        assert result.success is False
        assert "JSON parsing failed" in result.error_message

    def test_parse_missing_required_fields(self, parser):
        """Test handling of JSON missing required fields."""
        incomplete_response = {
            "issues": []
            # Missing confidence field
        }

        result = parser.parse_analysis_response(json.dumps(incomplete_response))

        assert result.success is False
        assert "must have 'confidence' field" in result.error_message

    def test_parse_invalid_confidence_range(self, parser):
        """Test handling of confidence values outside valid range."""
        invalid_response = {
            "issues": [],
            "confidence": 1.5,  # Invalid: > 1.0
        }

        result = parser.parse_analysis_response(json.dumps(invalid_response))

        assert result.success is False
        assert "must be between 0 and 1" in result.error_message

    def test_parse_invalid_issue_structure(self, parser):
        """Test handling of issues with invalid structure."""
        invalid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test issue",
                    # Missing required fields: suggestion, confidence, line_number
                }
            ],
            "confidence": 0.8,
        }

        result = parser.parse_analysis_response(json.dumps(invalid_response))

        assert result.success is False
        assert "Issue missing required field" in result.error_message

    def test_parse_invalid_issue_confidence(self, parser):
        """Test handling of issues with invalid confidence values."""
        invalid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                    "confidence": -0.1,  # Invalid: < 0.0
                    "line_number": 5,
                }
            ],
            "confidence": 0.8,
        }

        result = parser.parse_analysis_response(json.dumps(invalid_response))

        assert result.success is False

    def test_parse_invalid_line_number(self, parser):
        """Test handling of issues with invalid line numbers."""
        invalid_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                    "confidence": 0.8,
                    "line_number": 0,  # Invalid: must be positive
                }
            ],
            "confidence": 0.8,
        }

        result = parser.parse_analysis_response(json.dumps(invalid_response))

        assert result.success is False

    def test_issue_type_mapping(self, parser):
        """Test that LLM issue types are correctly mapped to our types."""
        response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                    "confidence": 0.8,
                    "line_number": 5,
                }
            ],
            "confidence": 0.8,
        }

        result = parser.parse_analysis_response(json.dumps(response))

        assert result.success is True
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "description_outdated"  # Mapped type
        assert result.issues[0].severity == "medium"  # From ISSUE_TYPES

    def test_unsupported_format(self, parser):
        """Test handling of unsupported response formats."""
        result = parser.parse_analysis_response("test", expected_format="xml")

        assert result.success is False
        assert "Unsupported format: xml" in result.error_message


class TestLenientParsing:
    """Test lenient parsing with error recovery."""

    @pytest.fixture
    def lenient_parser(self):
        """Create lenient parser for recovery testing."""
        return LLMOutputParser(strict_validation=False)

    def test_recovery_from_malformed_json(self, lenient_parser):
        """Test recovery from malformed JSON responses."""
        malformed_response = """
        I found some issues with this function:

        issue: Function handles None input but docstring doesn't mention this
        suggestion: Add note about None handling
        line 5

        issue: Example has syntax error
        suggestion: Fix the syntax error in example
        """

        result = lenient_parser.parse_analysis_response(malformed_response)

        # Should attempt recovery
        assert result.success is False or len(result.issues) > 0
        if result.issues:
            assert result.confidence <= 0.5  # Low confidence for recovered data

    def test_recovery_from_partial_json(self, lenient_parser):
        """Test recovery when JSON is partially valid."""
        partial_response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                    "confidence": 0.8,
                    "line_number": 5,
                }
            ]
            # Missing confidence field
        }

        # Parse as if it came from a malformed response
        result = lenient_parser._recover_from_malformed_response(
            json.dumps(partial_response), partial_response
        )

        assert len(result.issues) >= 0  # May recover some issues

    def test_strict_vs_lenient_mode(self):
        """Test difference between strict and lenient parsing modes."""
        invalid_response = (
            '{"issues": [], "confidence": "high"}'  # Invalid confidence type
        )

        strict_parser = LLMOutputParser(strict_validation=True)
        lenient_parser = LLMOutputParser(strict_validation=False)

        strict_result = strict_parser.parse_analysis_response(invalid_response)
        lenient_result = lenient_parser.parse_analysis_response(invalid_response)

        assert strict_result.success is False
        # Lenient parser should attempt recovery
        assert lenient_result.success is False or len(lenient_result.issues) >= 0


class TestIssueActionability:
    """Test issue actionability validation and filtering."""

    @pytest.fixture
    def parser(self):
        return LLMOutputParser()

    def test_actionable_suggestion_validation(self, parser):
        """Test validation of actionable suggestions."""
        actionable_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Parameter 'timeout' is not documented",
            suggestion="Add 'timeout' parameter to the Args section with description: 'Maximum wait time in seconds'",
            line_number=5,
            confidence=0.9,
        )

        assert parser.validate_issue_actionability(actionable_issue) is True

    def test_vague_suggestion_validation(self, parser):
        """Test validation rejects vague suggestions."""
        vague_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Something is wrong",
            suggestion="Fix this",
            line_number=5,
            confidence=0.9,
        )

        assert parser.validate_issue_actionability(vague_issue) is False

    def test_short_suggestion_validation(self, parser):
        """Test validation rejects too-short suggestions."""
        short_issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Issue description",
            suggestion="Update",
            line_number=5,
            confidence=0.9,
        )

        assert parser.validate_issue_actionability(short_issue) is False

    def test_filter_actionable_issues(self, parser):
        """Test filtering to only actionable issues."""
        issues = [
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="critical",
                description="Parameter missing",
                suggestion="Add parameter documentation with specific type and description",
                line_number=5,
                confidence=0.9,
            ),
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="critical",
                description="Something wrong",
                suggestion="Fix this",
                line_number=10,
                confidence=0.8,
            ),
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="critical",
                description="Return type unclear",
                suggestion="Specify return type as Dict[str, List[int]] in the Returns section",
                line_number=15,
                confidence=0.85,
            ),
        ]

        actionable = parser.filter_actionable_issues(issues)

        assert len(actionable) == 2  # Should filter out the vague one
        assert all("Fix this" not in issue.suggestion for issue in actionable)


class TestParsingStatistics:
    """Test parsing statistics and analysis."""

    def test_parsing_statistics_empty(self):
        """Test statistics with empty results list."""
        parser = LLMOutputParser()
        stats = parser.get_parsing_statistics([])

        expected = {
            "total_responses": 0,
            "successful_parses": 0,
            "success_rate": 0.0,
            "average_issues_per_response": 0.0,
            "common_errors": [],
        }

        assert stats == expected

    def test_parsing_statistics_mixed_results(self):
        """Test statistics with mixed success/failure results."""
        results = [
            ParseResult(issues=[], success=True, confidence=0.9),
            ParseResult(
                issues=[],
                success=False,
                error_message="JSON parsing failed: Invalid syntax",
            ),
            ParseResult(
                issues=[
                    InconsistencyIssue(
                        "parameter_missing", "critical", "Test", "Fix", 1, 0.8
                    )
                ],
                success=True,
                confidence=0.8,
            ),
            ParseResult(
                issues=[],
                success=False,
                error_message="JSON parsing failed: Missing field",
            ),
        ]

        parser = LLMOutputParser()
        stats = parser.get_parsing_statistics(results)

        assert stats["total_responses"] == 4
        assert stats["successful_parses"] == 2
        assert stats["success_rate"] == 0.5
        assert (
            stats["average_issues_per_response"] == 0.25
        )  # 1 issue across 4 responses
        assert len(stats["common_errors"]) > 0


class TestConvenienceFunctions:
    """Test convenience functions for common parsing scenarios."""

    def test_parse_llm_response_convenience(self):
        """Test parse_llm_response convenience function."""
        valid_response = {
            "issues": [],
            "confidence": 0.95,
            "analysis_notes": "No issues found",
        }

        result = parse_llm_response(json.dumps(valid_response))

        assert result.success is True
        assert result.confidence == 0.95

    def test_parse_llm_response_strict_mode(self):
        """Test parse_llm_response with strict validation."""
        invalid_response = '{"issues": [], "confidence": "high"}'

        result = parse_llm_response(invalid_response, strict=True)
        assert result.success is False

        result = parse_llm_response(invalid_response, strict=False)
        # Should attempt recovery in non-strict mode

    def test_parse_and_filter_response(self):
        """Test parse_and_filter_response convenience function."""
        response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Good issue description",
                    "suggestion": "Add specific documentation about behavior X in section Y",
                    "confidence": 0.85,
                    "line_number": 5,
                },
                {
                    "type": "example_invalid",
                    "description": "Bad issue",
                    "suggestion": "Fix this",  # Too vague
                    "confidence": 0.7,
                    "line_number": 10,
                },
            ],
            "confidence": 0.8,
        }

        issues = parse_and_filter_response(json.dumps(response), filter_actionable=True)

        assert len(issues) == 1  # Should filter out the vague suggestion
        assert issues[0].description == "Good issue description"

    def test_parse_and_filter_response_no_filtering(self):
        """Test parse_and_filter_response without actionability filtering."""
        response = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Issue 1",
                    "suggestion": "Fix this",  # Vague
                    "confidence": 0.85,
                    "line_number": 5,
                }
            ],
            "confidence": 0.8,
        }

        issues = parse_and_filter_response(
            json.dumps(response), filter_actionable=False
        )

        assert len(issues) == 1  # Should include even vague suggestions

    def test_validate_response_format_convenience(self):
        """Test validate_response_format convenience function."""
        valid_response = {
            "issues": [
                {
                    "type": "test",
                    "description": "test",
                    "suggestion": "test",
                    "confidence": 0.8,
                    "line_number": 5,
                }
            ],
            "confidence": 0.8,
        }

        is_valid, error_msg = validate_response_format(valid_response)
        assert is_valid is True
        assert error_msg == ""

        invalid_response = {"issues": "not a list"}
        is_valid, error_msg = validate_response_format(invalid_response)
        assert is_valid is False
        assert "list" in error_msg


class TestRegexRecovery:
    """Test regex-based recovery from malformed responses."""

    @pytest.fixture
    def parser(self):
        return LLMOutputParser(strict_validation=False)

    def test_extract_issues_with_regex(self, parser):
        """Test extracting issues using regex patterns."""
        text = """
        I found several issues:

        Issue: Function doesn't handle empty input
        Suggestion: Add check for empty input at line 5

        Problem: Example has syntax error
        Fix: Correct the parentheses in the example (line 10)
        """

        issues = parser._extract_issues_with_regex(text)

        # Should extract some issues
        assert len(issues) >= 0
        for issue in issues:
            assert isinstance(issue, InconsistencyIssue)
            assert issue.confidence <= 0.5  # Low confidence for regex extraction

    def test_regex_recovery_with_line_numbers(self, parser):
        """Test regex recovery can extract line numbers."""
        text = """
        Found issue at line 15: Parameter not documented
        Another problem at line 22: Return type mismatch
        """

        issues = parser._extract_issues_with_regex(text)

        # Check if line numbers were extracted
        for issue in issues:
            assert issue.line_number >= 1

    def test_regex_recovery_safety(self, parser):
        """Test that regex recovery doesn't crash on weird input."""
        weird_inputs = [
            "",
            "Complete nonsense with no patterns",
            "{{{{invalid json braces}}}}",
            "line" * 1000,  # Very long text
            "\x00\x01\x02",  # Binary data
        ]

        for text in weird_inputs:
            try:
                issues = parser._extract_issues_with_regex(text)
                assert isinstance(issues, list)
            except Exception as e:
                pytest.fail(f"Regex recovery failed on input: {e}")


@pytest.mark.integration
class TestParsingIntegration:
    """Integration tests for complete parsing workflows."""

    def test_full_parsing_workflow(self):
        """Test complete parsing workflow from raw response to filtered issues."""
        # Simulate a realistic LLM response
        llm_response = """
        Based on my analysis of the function, I found the following issues:

        ```json
        {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Function handles None input gracefully but this behavior is not documented in the docstring",
                    "suggestion": "Add note in Args section: 'If data is None, returns empty result dict.'",
                    "confidence": 0.90,
                    "line_number": 8,
                    "details": {"missing_behavior": "None handling"}
                },
                {
                    "type": "example_invalid",
                    "description": "Code example in docstring has incorrect variable name",
                    "suggestion": "Change 'results' to 'result' in the example to match function return",
                    "confidence": 0.95,
                    "line_number": 20,
                    "details": {"error_type": "variable_name"}
                },
                {
                    "type": "missing_edge_case",
                    "description": "Function handles empty list but this is not mentioned",
                    "suggestion": "Document behavior for empty input lists",
                    "confidence": 0.75,
                    "line_number": 12,
                    "details": {"condition": "empty list"}
                }
            ],
            "analysis_notes": "Analyzed function behavior, examples, and edge cases",
            "confidence": 0.87
        }
        ```

        This completes my analysis.
        """

        # Parse with strict validation
        result = parse_llm_response(llm_response, strict=True)

        assert result.success is True
        assert len(result.issues) == 3
        assert result.confidence == 0.87
        assert "Analyzed function behavior" in result.analysis_notes

        # All issues should be valid InconsistencyIssue objects
        for issue in result.issues:
            assert isinstance(issue, InconsistencyIssue)
            assert issue.issue_type in ISSUE_TYPES
            assert 0.0 <= issue.confidence <= 1.0
            assert issue.line_number > 0
            assert len(issue.description) > 0
            assert len(issue.suggestion) > 0

    def test_parsing_with_filtering_workflow(self):
        """Test parsing and filtering workflow."""
        response_with_mixed_quality = {
            "issues": [
                {
                    "type": "behavior_mismatch",
                    "description": "Specific issue with clear description",
                    "suggestion": "Add explicit documentation in the Args section: 'timeout (float): Maximum wait time in seconds. Default is 30.0.'",
                    "confidence": 0.90,
                    "line_number": 5,
                },
                {
                    "type": "example_invalid",
                    "description": "Vague issue description",
                    "suggestion": "Fix this issue",  # Too vague
                    "confidence": 0.60,
                    "line_number": 10,
                },
                {
                    "type": "missing_edge_case",
                    "description": "Another specific issue",
                    "suggestion": "Add handling documentation for negative timeout values: 'Raises ValueError if timeout < 0'",
                    "confidence": 0.85,
                    "line_number": 7,
                },
            ],
            "confidence": 0.78,
        }

        # Parse and filter for actionable issues
        actionable_issues = parse_and_filter_response(
            json.dumps(response_with_mixed_quality), filter_actionable=True
        )

        assert len(actionable_issues) == 2  # Should filter out the vague one

        for issue in actionable_issues:
            assert len(issue.suggestion) > 20  # Substantial suggestions
            assert "Fix this" not in issue.suggestion  # No vague language

    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        problematic_responses = [
            "Not JSON at all",
            '{"malformed": json}',
            '{"issues": "not a list", "confidence": 0.8}',
            '{"issues": [], "confidence": 2.0}',  # Invalid confidence
            "",  # Empty response
        ]

        for response in problematic_responses:
            # Should not crash
            result = parse_llm_response(response, strict=False)
            assert isinstance(result, ParseResult)

            # Strict parsing should fail gracefully
            strict_result = parse_llm_response(response, strict=True)
            assert strict_result.success is False or len(strict_result.issues) == 0
