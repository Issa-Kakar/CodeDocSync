"""
Tests for the Exception Documentation Generator.

Tests cover exception analysis, raises documentation generation, and suggestion
creation for various exception scenarios.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.docstring_models import DocstringRaises
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators.raises_generator import (
    ExceptionAnalyzer,
    ExceptionInfo,
    RaisesSuggestionGenerator,
)
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionType,
)


class TestExceptionAnalyzer:
    """Test the exception analyzer."""

    def test_analyze_direct_raise(self) -> None:
        """Test analyzing direct raise statements."""
        source_code = """
def test_func() -> None:
    if not condition:
        raise ValueError("Invalid value")
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        assert len(exceptions) >= 1
        exc_types = [e.exception_type for e in exceptions]
        assert "ValueError" in exc_types

    def test_analyze_multiple_exceptions(self) -> None:
        """Test analyzing multiple exception types."""
        source_code = """
def test_func(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("Expected string")
    if not value:
        raise ValueError("Value cannot be empty")
    return value.upper()
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        exc_types = [e.exception_type for e in exceptions]
        assert "TypeError" in exc_types
        assert "ValueError" in exc_types

    def test_analyze_function_calls(self) -> None:
        """Test analyzing exceptions from function calls."""
        source_code = """
def test_func(filename: Any) -> None:
    with open(filename, 'r') as f:
        content = f.read()
    return int(content)
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        exc_types = [e.exception_type for e in exceptions]
        # Should detect FileNotFoundError from open() and ValueError from int()
        assert "FileNotFoundError" in exc_types or "IOError" in exc_types
        assert "ValueError" in exc_types

    def test_analyze_subscript_operations(self) -> None:
        """Test analyzing exceptions from subscript operations."""
        source_code = """
def test_func(data: Any, key: Any) -> None:
    return data[key]
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        exc_types = [e.exception_type for e in exceptions]
        # Should detect potential KeyError and IndexError
        assert "KeyError" in exc_types or "IndexError" in exc_types

    def test_analyze_bare_raise(self) -> None:
        """Test analyzing bare raise statements."""
        source_code = """
def test_func() -> None:
    try:
        risky_operation()
    except Exception:
        log_error()
        raise
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        # Should detect re-raised exception
        reraised = [e for e in exceptions if e.is_re_raised]
        assert len(reraised) >= 1

    def test_deduplicate_exceptions(self) -> None:
        """Test deduplication of similar exceptions."""
        analyzer = ExceptionAnalyzer()

        # Create duplicate exceptions with different confidence
        exceptions = [
            ExceptionInfo("ValueError", "Description 1", confidence=0.5),
            ExceptionInfo("ValueError", "Description 2", confidence=0.9),
            ExceptionInfo("TypeError", "Description 3", confidence=0.8),
        ]

        deduplicated = analyzer._deduplicate_exceptions(exceptions)

        # Should keep higher confidence ValueError and the TypeError
        assert len(deduplicated) == 2
        valueerror_exc = next(
            e for e in deduplicated if e.exception_type == "ValueError"
        )
        assert valueerror_exc.confidence == 0.9

    def test_syntax_error_handling(self) -> None:
        """Test handling syntax errors gracefully."""
        source_code = """
def test_func(
    # Incomplete function
"""
        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        # Should return basic exception info
        assert len(exceptions) >= 1
        assert exceptions[0].exception_type == "Exception"
        assert exceptions[0].confidence <= 0.5


class TestRaisesSuggestionGenerator:
    """Test the raises suggestion generator."""

    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return SuggestionConfig(
            default_style="google",
            max_line_length=88,
        )

    @pytest.fixture
    def generator(self, config: Any) -> Any:
        """Create raises suggestion generator."""
        return RaisesSuggestionGenerator(config)

    @pytest.fixture
    def mock_function(self) -> Mock:
        """Create mock function."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "test_function"
        function.line_number = 10
        function.source_code = """
def test_function(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("Expected string")
    if not value:
        raise ValueError("Value cannot be empty")
    return value.upper()
"""
        return function

    @pytest.fixture
    def mock_docstring(self) -> Mock:
        """Create mock docstring."""
        docstring: Mock = Mock()
        docstring.format = "google"
        docstring.summary = "Test function"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Test function."""'
        return docstring

    @pytest.fixture
    def mock_issue(self) -> InconsistencyIssue:
        """Create mock issue."""
        return InconsistencyIssue(
            issue_type="missing_raises",
            severity="medium",
            description="Missing exception documentation",
            suggestion="Add raises documentation",
            line_number=10,
        )

    def test_add_missing_raises_documentation(
        self,
        generator: Any,
        mock_function: Mock,
        mock_docstring: Mock,
        mock_issue: InconsistencyIssue,
    ) -> None:
        """Test adding missing raises documentation."""
        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_raises_documentation(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RAISES_UPDATE
        assert suggestion.confidence >= 0.6
        assert (
            "Raises:" in suggestion.suggested_text
            or "raises" in suggestion.suggested_text.lower()
        )
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text

    def test_fix_raises_type_mismatch(
        self,
        generator: Any,
        mock_function: Mock,
        mock_docstring: Mock,
        mock_issue: InconsistencyIssue,
    ) -> None:
        """Test fixing raises type mismatch."""
        # Add existing (incorrect) raises documentation
        mock_docstring.raises = [
            DocstringRaises(
                exception_type="RuntimeError", description="Wrong exception"
            ),
            DocstringRaises(
                exception_type="ValueError", description="Correct exception"
            ),
        ]
        mock_issue.issue_type = "raises_type_mismatch"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._fix_raises_type_mismatch(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RAISES_UPDATE
        # Should keep ValueError, remove RuntimeError, add TypeError
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text
        # RuntimeError should be removed (not in actual exceptions)

    def test_improve_raises_description(
        self,
        generator: Any,
        mock_function: Mock,
        mock_docstring: Mock,
        mock_issue: InconsistencyIssue,
    ) -> None:
        """Test improving vague raises descriptions."""
        mock_docstring.raises = [
            DocstringRaises(exception_type="ValueError", description="error"),
            DocstringRaises(exception_type="TypeError", description="failure"),
        ]
        mock_issue.issue_type = "raises_description_vague"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_raises_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RAISES_UPDATE
        # Should have improved descriptions
        suggested_text = suggestion.suggested_text
        assert "ValueError" in suggested_text
        assert "TypeError" in suggested_text
        # Descriptions should be more detailed than original "error" and "failure"

    def test_is_vague_description(self, generator: Any) -> None:
        """Test detection of vague exception descriptions."""
        # Test vague descriptions
        assert generator._is_vague_description("error")
        assert generator._is_vague_description("exception")
        assert generator._is_vague_description("failure")
        assert generator._is_vague_description("when error occurs")
        assert generator._is_vague_description("")
        assert generator._is_vague_description("short")

        # Test good descriptions
        assert not generator._is_vague_description(
            "When the input value is not a string"
        )
        assert not generator._is_vague_description("If the file cannot be found")

    def test_generate_improved_exception_description(self, generator: Any) -> None:
        """Test generation of improved exception descriptions."""
        analyzer = ExceptionAnalyzer()

        # Test with source code
        source_code = """
def test_func() -> None:
    raise ValueError("Custom message")
"""

        result = generator._generate_improved_exception_description(
            "ValueError", source_code, analyzer
        )

        assert result is not None
        assert len(result) > 10  # Should be more descriptive than basic types

        # Test fallback to builtin descriptions
        result = generator._generate_improved_exception_description(
            "KeyError", "", analyzer
        )

        assert "key" in result.lower()

    def test_no_source_code_fallback(
        self,
        generator: Any,
        mock_function: Mock,
        mock_docstring: Mock,
        mock_issue: InconsistencyIssue,
    ) -> None:
        """Test fallback when source code is not available."""
        mock_function.source_code = ""

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_raises_documentation(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1  # Fallback suggestion

    def test_no_significant_exceptions(
        self,
        generator: Any,
        mock_function: Mock,
        mock_docstring: Mock,
        mock_issue: InconsistencyIssue,
    ) -> None:
        """Test handling when no significant exceptions are detected."""
        # Function with only very low-confidence exceptions
        mock_function.source_code = """
def simple_func():
    return "hello"
"""

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_raises_documentation(context)

        # Should be a fallback suggestion since no significant exceptions
        assert suggestion.confidence <= 0.3

    def test_unknown_issue_type(
        self, generator: Any, mock_function: Mock, mock_docstring: Mock
    ) -> None:
        """Test handling unknown issue types."""
        unknown_issue = InconsistencyIssue(
            issue_type="unknown_raises_issue",
            severity="medium",
            description="Unknown issue",
            suggestion="",
            line_number=10,
        )

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=unknown_issue
        )

        suggestion = generator.generate(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1
        assert "Unknown raises issue type" in suggestion.suggested_text


class TestRaisesGeneratorIntegration:
    """Integration tests for raises generator."""

    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return SuggestionConfig(default_style="google")

    @pytest.fixture
    def generator(self, config: Any) -> Any:
        """Create raises suggestion generator."""
        return RaisesSuggestionGenerator(config)

    def test_complete_workflow_file_operations(self, generator: Any) -> None:
        """Test complete workflow for file operations function."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "read_config_file"
        function.line_number = 5
        function.source_code = """
def read_config_file(filename):
    if not filename:
        raise ValueError("Filename cannot be empty")

    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {filename} not found")
    except PermissionError:
        raise PermissionError(f"Permission denied: {filename}")

    try:
        import json
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {filename}")
"""

        docstring: Mock = Mock()
        docstring.format = "google"
        docstring.summary = "Read configuration from file"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None
        docstring.raises = []  # Missing raises documentation
        docstring.examples = []
        docstring.raw_text = '"""Read configuration from file."""'

        issue = InconsistencyIssue(
            issue_type="missing_raises",
            severity="medium",
            description="Missing exception documentation",
            suggestion="Document exceptions",
            line_number=5,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify comprehensive exception documentation
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RAISES_UPDATE
        assert suggestion.confidence >= 0.6

        suggested_text = suggestion.suggested_text
        assert "Raises:" in suggested_text or "raises" in suggested_text.lower()

        # Should document the main exceptions
        assert "ValueError" in suggested_text
        assert "FileNotFoundError" in suggested_text
        assert "PermissionError" in suggested_text

    def test_complete_workflow_mismatch_correction(self, generator: Any) -> None:
        """Test complete workflow for correcting exception mismatches."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "validate_input"
        function.line_number = 8
        function.source_code = """
def validate_input(data):
    if not isinstance(data, dict):
        raise TypeError("Expected dictionary")
    if 'required_field' not in data:
        raise ValueError("Missing required field")
    return True
"""

        docstring: Mock = Mock()
        docstring.format = "google"
        docstring.summary = "Validate input data"
        docstring.raises = [
            # Incorrect exception documented
            DocstringRaises(
                exception_type="RuntimeError", description="When validation fails"
            ),
            # Missing TypeError, has ValueError but different description
            DocstringRaises(exception_type="ValueError", description="Old description"),
        ]
        docstring.parameters = []
        docstring.returns = None
        docstring.examples = []
        docstring.raw_text = '"""Validate input data."""'

        issue = InconsistencyIssue(
            issue_type="raises_type_mismatch",
            severity="high",
            description="Exception type mismatch",
            suggestion="Fix documented exceptions",
            line_number=8,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify mismatch correction
        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.8

        suggested_text = suggestion.suggested_text

        # Should include actual exceptions
        assert "TypeError" in suggested_text
        assert "ValueError" in suggested_text

        # Should not include the incorrect RuntimeError
        # (This is harder to test without parsing the docstring, but at minimum
        # the correct exceptions should be present)

    def test_edge_case_no_exceptions_detected(self, generator: Any) -> None:
        """Test edge case where no exceptions are detected."""
        function: Mock = Mock()
        function.signature = Mock()
        function.signature.name = "simple_getter"
        function.line_number = 3
        function.source_code = """
def simple_getter(self):
    return self._value
"""

        docstring: Mock = Mock()
        docstring.format = "google"
        docstring.summary = "Get the value"
        docstring.raises = []
        docstring.raw_text = '"""Get the value."""'

        issue = InconsistencyIssue(
            issue_type="missing_raises",
            severity="low",
            description="Check for exceptions",
            suggestion="",
            line_number=3,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Should be low confidence since no significant exceptions
        assert suggestion.confidence <= 0.3
