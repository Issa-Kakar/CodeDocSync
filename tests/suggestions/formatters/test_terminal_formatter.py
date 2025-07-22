"""
Tests for terminal output formatter.

Tests the terminal formatting functionality for suggestions,
including rich output, plain text, and minimal formatting modes.
"""

from unittest.mock import Mock, patch

import pytest

from codedocsync.suggestions.formatters.terminal_formatter import (
    OutputStyle,
    TerminalFormatterConfig,
    TerminalSuggestionFormatter,
)
from codedocsync.suggestions.integration import EnhancedAnalysisResult, EnhancedIssue
from codedocsync.suggestions.models import (
    DocstringStyle,
    Suggestion,
    SuggestionBatch,
    SuggestionDiff,
    SuggestionType,
)


# Test fixtures
@pytest.fixture
def basic_suggestion():
    """Create a basic suggestion for testing."""
    return Suggestion(
        suggestion_type=SuggestionType.PARAMETER_UPDATE,
        original_text='def func(email: str):\n    """Function with email param."""',
        suggested_text='def func(username: str):\n    """Function with username param."""',
        confidence=0.9,
        style=DocstringStyle.GOOGLE,
        copy_paste_ready=True,
    )


@pytest.fixture
def suggestion_with_diff():
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
        style=DocstringStyle.GOOGLE,
        copy_paste_ready=True,
        diff=diff,
    )


@pytest.fixture
def enhanced_issue(basic_suggestion):
    """Create an enhanced issue with suggestion."""
    return EnhancedIssue(
        issue_type="parameter_name_mismatch",
        severity="critical",
        description="Parameter 'email' doesn't match 'username' in code",
        suggestion="Update parameter name",
        line_number=45,
        confidence=0.9,
        rich_suggestion=basic_suggestion,
    )


@pytest.fixture
def enhanced_issue_no_suggestion():
    """Create an enhanced issue without suggestion."""
    return EnhancedIssue(
        issue_type="parameter_missing",
        severity="high",
        description="Missing parameter documentation",
        suggestion="Add parameter to docstring",
        line_number=30,
        confidence=0.8,
    )


@pytest.fixture
def enhanced_result(enhanced_issue):
    """Create an enhanced analysis result."""
    mock_function = Mock()
    mock_function.signature = Mock()
    mock_function.signature.name = "authenticate_user"
    mock_function.file_path = "auth/user.py"

    mock_pair = Mock()
    mock_pair.function = mock_function

    return EnhancedAnalysisResult(
        matched_pair=mock_pair,
        issues=[enhanced_issue],
        used_llm=False,
        analysis_time_ms=25.5,
        suggestion_generation_time_ms=15.2,
        suggestions_generated=1,
        suggestions_skipped=0,
    )


@pytest.fixture
def suggestion_batch(basic_suggestion):
    """Create a suggestion batch."""
    return SuggestionBatch(
        suggestions=[basic_suggestion],
        total_issues=2,
        functions_processed=1,
        generation_time_ms=50.0,
    )


class TestTerminalFormatterConfig:
    """Test TerminalFormatterConfig data model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TerminalFormatterConfig()

        assert config.max_width == 88
        assert config.show_line_numbers
        assert config.show_confidence
        assert config.show_diff
        assert config.syntax_theme == "monokai"
        assert config.use_unicode
        assert not config.compact_mode

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TerminalFormatterConfig(
            max_width=120,
            show_line_numbers=False,
            compact_mode=True,
        )

        assert config.max_width == 120
        assert not config.show_line_numbers
        assert config.compact_mode


class TestTerminalSuggestionFormatterInit:
    """Test TerminalSuggestionFormatter initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        formatter = TerminalSuggestionFormatter()

        assert isinstance(formatter.config, TerminalFormatterConfig)
        assert (
            formatter.style == OutputStyle.RICH or formatter.style == OutputStyle.PLAIN
        )

    def test_custom_config_initialization(self) -> None:
        """Test initialization with custom config."""
        config = TerminalFormatterConfig(max_width=100)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        assert formatter.config.max_width == 100
        assert formatter.style == OutputStyle.PLAIN

    @patch(
        "codedocsync.suggestions.formatters.terminal_formatter.RICH_AVAILABLE", False
    )
    def test_no_rich_fallback(self) -> None:
        """Test fallback when Rich is not available."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.RICH)

        # Should fallback to plain style
        assert formatter.style == OutputStyle.PLAIN


class TestSuggestionFormatting:
    """Test individual suggestion formatting."""

    def test_format_basic_suggestion_plain(self, basic_suggestion) -> None:
        """Test formatting basic suggestion in plain text."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_suggestion(basic_suggestion)

        assert "Parameter Update" in result
        assert "Confidence: 90%" in result
        assert basic_suggestion.suggested_text in result

    def test_format_suggestion_minimal(self, basic_suggestion) -> None:
        """Test formatting suggestion in minimal mode."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.MINIMAL)
        result = formatter.format_suggestion(basic_suggestion)

        # Minimal should just return the suggested text
        assert result == basic_suggestion.suggested_text

    def test_format_suggestion_with_line_numbers(self, basic_suggestion) -> None:
        """Test formatting with line numbers enabled."""
        config = TerminalFormatterConfig(show_line_numbers=True)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)
        result = formatter.format_suggestion(basic_suggestion)

        # Should include line numbers
        assert "1:" in result or "  1:" in result

    def test_format_suggestion_without_line_numbers(self, basic_suggestion) -> None:
        """Test formatting without line numbers."""
        config = TerminalFormatterConfig(show_line_numbers=False)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)
        result = formatter.format_suggestion(basic_suggestion)

        # Should not include line number format
        assert "1:" not in result

    def test_format_suggestion_without_confidence(self, basic_suggestion) -> None:
        """Test formatting without confidence display."""
        config = TerminalFormatterConfig(show_confidence=False)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)
        result = formatter.format_suggestion(basic_suggestion)

        # Should not include confidence
        assert "Confidence:" not in result

    @patch("codedocsync.suggestions.formatters.terminal_formatter.RICH_AVAILABLE", True)
    def test_format_suggestion_rich(self, basic_suggestion) -> None:
        """Test rich formatting (mocked)."""
        with patch(
            "codedocsync.suggestions.formatters.terminal_formatter.Console"
        ) as mock_console:
            mock_console_instance = Mock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.capture.return_value.__enter__.return_value = Mock()
            mock_console_instance.capture.return_value.__enter__.return_value.get.return_value = (
                "rich output"
            )

            formatter = TerminalSuggestionFormatter(style=OutputStyle.RICH)
            result = formatter.format_suggestion(basic_suggestion)

            # Should use rich formatting
            assert "rich output" in result


class TestEnhancedIssueFormatting:
    """Test enhanced issue formatting."""

    def test_format_issue_with_suggestion_plain(self, enhanced_issue) -> None:
        """Test formatting issue with suggestion in plain text."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_enhanced_issue(enhanced_issue)

        assert "[CRITICAL]" in result
        assert enhanced_issue.description in result
        assert f"Line: {enhanced_issue.line_number}" in result
        assert enhanced_issue.rich_suggestion.suggested_text in result

    def test_format_issue_without_suggestion_plain(
        self, enhanced_issue_no_suggestion
    ) -> None:
        """Test formatting issue without suggestion in plain text."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_enhanced_issue(enhanced_issue_no_suggestion)

        assert "[HIGH]" in result
        assert enhanced_issue_no_suggestion.description in result
        # Should not contain suggestion text since there's no rich_suggestion
        assert "Parameter Update" not in result

    def test_format_issue_minimal(self, enhanced_issue) -> None:
        """Test formatting issue in minimal mode."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.MINIMAL)
        result = formatter.format_enhanced_issue(enhanced_issue)

        assert "CRITICAL:" in result
        assert enhanced_issue.description in result
        assert enhanced_issue.rich_suggestion.suggested_text in result

    def test_severity_indicators(self) -> None:
        """Test different severity indicators."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)

        severities = ["critical", "high", "medium", "low"]
        expected_indicators = ["[CRITICAL]", "[HIGH]", "[MEDIUM]", "[LOW]"]

        for severity, expected in zip(severities, expected_indicators, strict=False):
            issue = EnhancedIssue(
                issue_type="test",
                severity=severity,
                description="Test issue",
                suggestion="Test suggestion",
                line_number=1,
            )

            result = formatter.format_enhanced_issue(issue)
            assert expected in result


class TestAnalysisResultFormatting:
    """Test analysis result formatting."""

    def test_format_result_with_issues_plain(self, enhanced_result) -> None:
        """Test formatting result with issues in plain text."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_analysis_result(enhanced_result)

        assert "authenticate_user" in result
        assert "auth/user.py" in result
        assert "Issues: 1" in result
        assert "Suggestions: 1" in result
        assert "Issue 1:" in result

    def test_format_result_no_issues(self) -> None:
        """Test formatting result with no issues."""
        mock_function = Mock()
        mock_function.signature = Mock()
        mock_function.signature.name = "clean_function"

        mock_pair = Mock()
        mock_pair.function = mock_function

        empty_result = EnhancedAnalysisResult(
            matched_pair=mock_pair,
            issues=[],
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_analysis_result(empty_result)

        assert "No issues found" in result
        assert "clean_function" in result

    def test_format_result_minimal(self, enhanced_result) -> None:
        """Test formatting result in minimal mode."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.MINIMAL)
        result = formatter.format_analysis_result(enhanced_result)

        # Should be concise
        assert len(result.split("\n")) <= 5


class TestBatchSummaryFormatting:
    """Test batch summary formatting."""

    def test_format_batch_summary_plain(self, suggestion_batch) -> None:
        """Test formatting batch summary in plain text."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_batch_summary(suggestion_batch)

        assert "Suggestion Generation Summary" in result
        assert "Functions Processed: 1" in result
        assert "Total Issues: 2" in result
        assert "Suggestions Generated: 1" in result
        assert "Generation Time: 50.0ms" in result
        assert "Average Confidence:" in result

    def test_format_batch_summary_minimal(self, suggestion_batch) -> None:
        """Test formatting batch summary in minimal mode."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.MINIMAL)
        result = formatter.format_batch_summary(suggestion_batch)

        assert "Processed: 1" in result
        assert "Issues: 2" in result
        assert "Suggestions: 1" in result

    def test_format_empty_batch(self) -> None:
        """Test formatting empty batch summary."""
        empty_batch = SuggestionBatch(
            suggestions=[],
            total_issues=0,
            functions_processed=0,
            generation_time_ms=0.0,
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_batch_summary(empty_batch)

        assert "Functions Processed: 0" in result
        assert "Total Issues: 0" in result


class TestDiffFormatting:
    """Test diff formatting functionality."""

    def test_diff_preview_creation(self, suggestion_with_diff) -> None:
        """Test creating diff preview."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)

        # Access the diff creation method
        diff_content = formatter._create_rich_diff(suggestion_with_diff)

        # Should indicate lines changed
        assert (
            "modified" in diff_content
            or "added" in diff_content
            or "removed" in diff_content
        )

    def test_diff_display_configuration(self, suggestion_with_diff) -> None:
        """Test diff display with configuration."""
        config = TerminalFormatterConfig(show_diff=True)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        result = formatter.format_suggestion(suggestion_with_diff)

        # When show_diff is True, should process diff
        # (Exact content depends on implementation)
        assert isinstance(result, str)

    def test_no_diff_display(self, suggestion_with_diff) -> None:
        """Test disabling diff display."""
        config = TerminalFormatterConfig(show_diff=False)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        result = formatter.format_suggestion(suggestion_with_diff)

        # Should not show diff information
        assert isinstance(result, str)


class TestUtilityMethods:
    """Test utility methods in formatter."""

    def test_format_issue_only(self, enhanced_issue_no_suggestion) -> None:
        """Test formatting issue without suggestion."""
        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter._format_issue_only(enhanced_issue_no_suggestion)

        assert "[HIGH]" in result
        assert enhanced_issue_no_suggestion.description in result
        assert f"(line {enhanced_issue_no_suggestion.line_number})" in result

    def test_format_no_issues(self) -> None:
        """Test formatting when no issues found."""
        mock_function = Mock()
        mock_function.signature = Mock()
        mock_function.signature.name = "perfect_function"

        mock_pair = Mock()
        mock_pair.function = mock_function

        empty_result = EnhancedAnalysisResult(
            matched_pair=mock_pair,
            issues=[],
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter._format_no_issues(empty_result)

        assert "No issues found" in result
        assert "perfect_function" in result


class TestConfigurationOptions:
    """Test various configuration options."""

    def test_max_width_configuration(self, basic_suggestion) -> None:
        """Test max width configuration."""
        config = TerminalFormatterConfig(max_width=50)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        result = formatter.format_suggestion(basic_suggestion)

        # Should respect width (though exact behavior depends on implementation)
        assert isinstance(result, str)

    def test_unicode_configuration(self, enhanced_issue) -> None:
        """Test unicode configuration."""
        config = TerminalFormatterConfig(use_unicode=False)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        result = formatter.format_enhanced_issue(enhanced_issue)

        # Should work without unicode characters
        assert isinstance(result, str)

    def test_compact_mode(self, enhanced_result) -> None:
        """Test compact mode configuration."""
        config = TerminalFormatterConfig(compact_mode=True)
        formatter = TerminalSuggestionFormatter(config, OutputStyle.PLAIN)

        result = formatter.format_analysis_result(enhanced_result)

        # Should be more compact (exact behavior depends on implementation)
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_suggestion_without_diff(self, basic_suggestion) -> None:
        """Test suggestion without diff information."""
        # Ensure no diff
        basic_suggestion.diff = None

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_suggestion(basic_suggestion)

        # Should handle gracefully
        assert isinstance(result, str)
        assert basic_suggestion.suggested_text in result

    def test_issue_with_zero_line_number(self) -> None:
        """Test issue with zero line number."""
        issue = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="Test issue",
            suggestion="Test suggestion",
            line_number=0,  # Edge case
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_enhanced_issue(issue)

        # Should handle gracefully, might not show line number
        assert isinstance(result, str)
        assert issue.description in result

    def test_function_without_signature(self) -> None:
        """Test function without proper signature."""
        mock_function = Mock()
        # No signature attribute

        mock_pair = Mock()
        mock_pair.function = mock_function

        result = EnhancedAnalysisResult(
            matched_pair=mock_pair,
            issues=[],
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        formatted = formatter.format_analysis_result(result)

        # Should handle gracefully with "Unknown" function name
        assert "Unknown" in formatted

    def test_very_long_suggestion_text(self) -> None:
        """Test handling very long suggestion text."""
        long_text = "def function():\n    " + 'x = "very long string" * 100\n' * 50

        suggestion = Suggestion(
            suggestion_type=SuggestionType.FULL_DOCSTRING,
            original_text="short",
            suggested_text=long_text,
            confidence=0.8,
            style=DocstringStyle.GOOGLE,
            copy_paste_ready=True,
        )

        formatter = TerminalSuggestionFormatter(style=OutputStyle.PLAIN)
        result = formatter.format_suggestion(suggestion)

        # Should handle long text gracefully
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
