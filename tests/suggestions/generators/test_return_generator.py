"""
Tests for the Return Type Suggestion Generator.

Tests cover return type analysis, mismatch detection, and suggestion generation
for various return scenarios including generators, async functions, and complex types.
"""

from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.docstring_models import DocstringReturns
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators.return_generator import (
    ReturnAnalysisResult,
    ReturnStatementAnalyzer,
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionType,
)


class TestReturnStatementAnalyzer:
    """Test the return statement analyzer."""

    def test_analyze_simple_return(self) -> None:
        """Test analyzing function with simple return."""
        source_code = """
def test_func() -> None:
    return "hello"
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert result.has_explicit_return
        assert not result.has_implicit_none
        assert not result.is_generator
        assert not result.is_async
        assert "str" in result.return_types

    def test_analyze_multiple_returns(self) -> None:
        """Test analyzing function with multiple return types."""
        source_code = """
def test_func(flag) -> None:
    if flag:
        return 42
    else:
        return "hello"
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert result.has_explicit_return
        assert len(result.return_types) == 2
        assert "int" in result.return_types
        assert "str" in result.return_types

    def test_analyze_generator_function(self) -> None:
        """Test analyzing generator function."""
        source_code = """
def test_func() -> None:
    for i in range(10):
        yield i
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert result.is_generator
        assert "Generator[int]" in result.return_types

    def test_analyze_async_function(self) -> None:
        """Test analyzing async function."""
        source_code = """
async def test_func() -> None:
    return "hello"
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert result.is_async
        assert result.has_explicit_return
        assert "str" in result.return_types

    def test_analyze_no_return(self) -> None:
        """Test analyzing function with no explicit return."""
        source_code = """
def test_func() -> None:
    print("hello")
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert not result.has_explicit_return
        assert result.has_implicit_none
        assert "None" in result.return_types

    def test_analyze_syntax_error(self) -> None:
        """Test handling syntax errors gracefully."""
        source_code = """
def test_func(
    # Incomplete function
"""
        analyzer = ReturnStatementAnalyzer()
        result = analyzer.analyze(source_code)

        assert result.complexity_score == 0.1
        assert result.has_implicit_none


class TestReturnSuggestionGenerator:
    """Test the return suggestion generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(
            default_style="google",
            max_line_length=88,
        )

    @pytest.fixture
    def generator(self, config):
        """Create return suggestion generator."""
        return ReturnSuggestionGenerator(config)

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "test_function"
        function.signature.return_annotation = "str"
        function.line_number = 10
        function.source_code = """
def test_function() -> None:
    return "hello"
"""
        return function

    @pytest.fixture
    def mock_docstring(self):
        """Create mock docstring."""
        docstring = Mock()
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
    def mock_issue(self):
        """Create mock issue."""
        return InconsistencyIssue(
            issue_type="return_type_mismatch",
            severity="high",
            description="Return type mismatch",
            suggestion="Fix return type",
            line_number=10,
        )

    def test_fix_return_type_mismatch(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test fixing return type mismatch."""
        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._fix_return_type(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert suggestion.confidence >= 0.7
        assert "str" in suggestion.suggested_text

    def test_add_missing_return_documentation(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test adding missing return documentation."""
        mock_issue.issue_type = "missing_returns"
        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._add_missing_return_documentation(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert (
            "Returns:" in suggestion.suggested_text
            or "returns" in suggestion.suggested_text.lower()
        )

    def test_improve_return_description(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test improving vague return description."""
        mock_docstring.returns = DocstringReturns(type_str="str", description="result")
        mock_issue.issue_type = "return_description_vague"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._improve_return_description(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE

    def test_fix_generator_return(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test fixing generator return documentation."""
        mock_function.source_code = """
def test_function() -> None:
    for i in range(10):
        yield i
"""
        mock_issue.issue_type = "generator_return_incorrect"

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._fix_generator_return(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert "Generator" in suggestion.suggested_text

    def test_determine_best_return_type_single(self, generator) -> None:
        """Test determining best return type with single type."""
        analysis = ReturnAnalysisResult()
        analysis.return_types = {"str"}
        analysis.is_generator = False
        analysis.has_implicit_none = False

        mock_function = Mock()
        mock_function.signature.return_annotation = None

        result = generator._determine_best_return_type(analysis, mock_function)
        assert result == "str"

    def test_determine_best_return_type_multiple(self, generator) -> None:
        """Test determining best return type with multiple types."""
        analysis = ReturnAnalysisResult()
        analysis.return_types = {"str", "int"}
        analysis.is_generator = False
        analysis.has_implicit_none = False

        mock_function = Mock()
        mock_function.signature.return_annotation = None

        result = generator._determine_best_return_type(analysis, mock_function)
        assert "Union" in result

    def test_determine_best_return_type_generator(self, generator) -> None:
        """Test determining best return type for generator."""
        analysis = ReturnAnalysisResult()
        analysis.is_generator = True

        mock_function = Mock()

        result = generator._determine_best_return_type(analysis, mock_function)
        assert result == "Generator"

    def test_generate_return_description_basic_types(self, generator) -> None:
        """Test generating return descriptions for basic types."""
        # Test various basic types
        test_cases = [
            ("bool", "True if successful, False otherwise"),
            ("str", "The string result"),
            ("int", "The integer result"),
            ("List", "List of results"),
            ("Dict", "Dictionary containing results"),
        ]

        for return_type, expected_desc in test_cases:
            result = generator._generate_return_description(return_type, None)
            assert expected_desc in result

    def test_generate_return_description_special_cases(self, generator) -> None:
        """Test generating return descriptions for special cases."""
        # Test None
        result = generator._generate_return_description("None", None)
        assert result == "None"

        # Test Generator
        result = generator._generate_return_description("Generator[int]", None)
        assert "Generator" in result and "yielding" in result

        # Test Union
        result = generator._generate_return_description("Union[str, int]", None)
        assert "result" in result.lower()

    def test_no_source_code_fallback(
        self, generator, mock_function, mock_docstring, mock_issue
    ) -> None:
        """Test fallback when source code is not available."""
        mock_function.source_code = ""

        context = SuggestionContext(
            function=mock_function, docstring=mock_docstring, issue=mock_issue
        )

        suggestion = generator._fix_return_type(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence == 0.1  # Fallback suggestion

    def test_unknown_issue_type(self, generator, mock_function, mock_docstring) -> None:
        """Test handling unknown issue types."""
        unknown_issue = InconsistencyIssue(
            issue_type="unknown_return_issue",
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
        assert "Unknown return issue type" in suggestion.description


class TestReturnGeneratorIntegration:
    """Integration tests for return generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SuggestionConfig(default_style="google")

    @pytest.fixture
    def generator(self, config):
        """Create return suggestion generator."""
        return ReturnSuggestionGenerator(config)

    def test_complete_workflow_missing_returns(self, generator) -> None:
        """Test complete workflow for missing returns."""
        # Create realistic function and docstring
        function = Mock()
        function.signature = Mock()
        function.signature.name = "calculate_sum"
        function.signature.return_annotation = "int"
        function.line_number = 5
        function.source_code = """
def calculate_sum(a, b):
    return a + b
"""

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Calculate sum of two numbers"
        docstring.description = None
        docstring.parameters = []
        docstring.returns = None  # Missing returns
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Calculate sum of two numbers."""'

        issue = InconsistencyIssue(
            issue_type="missing_returns",
            severity="high",
            description="Missing return documentation",
            suggestion="Add return documentation",
            line_number=5,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify suggestion quality
        assert isinstance(suggestion, Suggestion)
        assert suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert suggestion.confidence >= 0.8
        assert suggestion.copy_paste_ready

        # Verify content
        suggested_text = suggestion.suggested_text
        assert "Calculate sum of two numbers" in suggested_text
        assert "Returns:" in suggested_text or "returns" in suggested_text.lower()
        assert "int" in suggested_text or "integer" in suggested_text

    def test_complete_workflow_generator_function(self, generator) -> None:
        """Test complete workflow for generator function."""
        function = Mock()
        function.signature = Mock()
        function.signature.name = "number_generator"
        function.signature.return_annotation = None
        function.line_number = 8
        function.source_code = """
def number_generator(n):
    for i in range(n):
        yield i * 2
"""

        docstring = Mock()
        docstring.format = "google"
        docstring.summary = "Generate numbers"
        docstring.returns = DocstringReturns(
            type_str="list", description="List of numbers"
        )  # Wrong for generator
        docstring.parameters = []
        docstring.raises = []
        docstring.examples = []
        docstring.raw_text = '"""Generate numbers."""'

        issue = InconsistencyIssue(
            issue_type="generator_return_incorrect",
            severity="high",
            description="Generator return type incorrect",
            suggestion="Fix generator return",
            line_number=8,
        )

        context = SuggestionContext(function=function, docstring=docstring, issue=issue)

        suggestion = generator.generate(context)

        # Verify generator-specific handling
        assert isinstance(suggestion, Suggestion)
        assert suggestion.confidence >= 0.9
        assert "Generator" in suggestion.suggested_text
