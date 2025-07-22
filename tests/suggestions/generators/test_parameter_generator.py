"""Tests for parameter suggestion generator."""

from unittest.mock import Mock, patch
from typing import Any, List, Optional, Union

import pytest

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
)
from codedocsync.parser.docstring_models import (
    DocstringFormat,
    DocstringParameter,
    ParsedDocstring,
)
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators import ParameterSuggestionGenerator
from codedocsync.suggestions.models import (
    DocstringStyle,
    SuggestionContext,
    SuggestionType,
)


class TestParameterSuggestionGenerator:
    """Test parameter suggestion generator functionality."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create parameter generator instance."""
        config = SuggestionConfig(default_style="google")
        return ParameterSuggestionGenerator(config)

    @pytest.fixture
    def sample_function(self) -> Any:
        """Create sample function for testing."""
        signature = FunctionSignature(
            name="test_function",
            parameters=[
                FunctionParameter(
                    name="param1", type_annotation="str", is_required=True
                ),
                FunctionParameter(
                    name="param2",
                    type_annotation="int",
                    is_required=False,
                    default_value="0",
                ),
            ],
        )

        return ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

    @pytest.fixture
    def sample_docstring(self) -> Any:
        """Create sample docstring for testing."""
        return ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function",
            parameters=[
                DocstringParameter(
                    name="param1", type_str="str", description="First parameter"
                ),
                DocstringParameter(
                    name="wrong_name", type_str="int", description="Second parameter"
                ),
            ],
            raw_text='"""Test function.\n\nArgs:\n    param1: First parameter\n    wrong_name: Second parameter\n"""',
        )

    def test_fix_parameter_name_mismatch(
        self, generator, sample_function, sample_docstring
    ) -> None:
        """Test fixing parameter name mismatch."""
        issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Parameter name mismatch",
            suggestion="Fix parameter name",
            line_number=10,
        )

        context = SuggestionContext(
            issue=issue, function=sample_function, docstring=sample_docstring
        )

        with patch("rapidfuzz.fuzz.ratio", return_value=85):
            suggestion = generator.generate(context)

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence >= 0.9
        assert "param2" in suggestion.suggested_text
        assert "wrong_name" not in suggestion.suggested_text

    def test_add_missing_parameter(self, generator: Any, sample_function: Any) -> None:
        """Test adding missing parameter documentation."""
        # Docstring missing param2
        docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function",
            parameters=[
                DocstringParameter(
                    name="param1", type_str="str", description="First parameter"
                )
            ],
            raw_text='"""Test function.\n\nArgs:\n    param1: First parameter\n"""',
        )

        issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="critical",
            description="Missing parameter documentation",
            suggestion="Add missing parameter",
            line_number=10,
        )

        context = SuggestionContext(
            issue=issue, function=sample_function, docstring=docstring
        )

        suggestion = generator.generate(context)

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence >= 0.9
        assert "param2" in suggestion.suggested_text
        assert "Args:" in suggestion.suggested_text

    def test_fix_parameter_type_mismatch(self, generator: Any, sample_function: Any) -> None:
        """Test fixing parameter type mismatch."""
        # Docstring has wrong type for param1
        docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function",
            parameters=[
                DocstringParameter(
                    name="param1", type_str="int", description="First parameter"
                ),  # Wrong type
                DocstringParameter(
                    name="param2", type_str="int", description="Second parameter"
                ),
            ],
            raw_text='"""Test function.\n\nArgs:\n    param1 (int): First parameter\n    param2 (int): Second parameter\n"""',
        )

        issue = InconsistencyIssue(
            issue_type="parameter_type_mismatch",
            severity="high",
            description="Parameter type mismatch",
            suggestion="Fix parameter type",
            line_number=10,
        )

        context = SuggestionContext(
            issue=issue, function=sample_function, docstring=docstring
        )

        suggestion = generator.generate(context)

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence >= 0.8
        assert "param1 (str)" in suggestion.suggested_text  # Should be corrected to str

    def test_fix_parameter_order(self, generator: Any) -> None:
        """Test fixing parameter order mismatch."""
        # Function with param1, param2 but docstring has param2, param1
        signature = FunctionSignature(
            name="test_function",
            parameters=[
                FunctionParameter(name="param1", type_annotation="str"),
                FunctionParameter(name="param2", type_annotation="int"),
            ],
        )

        function = ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

        docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function",
            parameters=[
                DocstringParameter(
                    name="param2", type_str="int", description="Second parameter"
                ),
                DocstringParameter(
                    name="param1", type_str="str", description="First parameter"
                ),
            ],
        )

        issue = InconsistencyIssue(
            issue_type="parameter_order_different",
            severity="medium",
            description="Parameter order mismatch",
            suggestion="Reorder parameters",
            line_number=10,
        )

        context = SuggestionContext(issue=issue, function=function, docstring=docstring)

        suggestion = generator.generate(context)

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        # Check that param1 comes before param2 in the suggested text
        param1_pos = suggestion.suggested_text.find("param1")
        param2_pos = suggestion.suggested_text.find("param2")
        assert param1_pos < param2_pos

    def test_add_kwargs_documentation(self, generator: Any) -> None:
        """Test adding **kwargs documentation."""
        signature = FunctionSignature(
            name="test_function",
            parameters=[
                FunctionParameter(name="param1", type_annotation="str"),
                FunctionParameter(name="**kwargs", type_annotation=None),
            ],
        )

        function = ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

        docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary="Test function",
            parameters=[
                DocstringParameter(
                    name="param1", type_str="str", description="First parameter"
                )
            ],
        )

        issue = InconsistencyIssue(
            issue_type="undocumented_kwargs",
            severity="medium",
            description="Undocumented **kwargs",
            suggestion="Add kwargs documentation",
            line_number=10,
        )

        context = SuggestionContext(issue=issue, function=function, docstring=docstring)

        suggestion = generator.generate(context)

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert "**kwargs" in suggestion.suggested_text
        assert "Additional keyword arguments" in suggestion.suggested_text

    def test_filter_special_parameters(self, generator: Any) -> None:
        """Test filtering special parameters like self and cls."""
        # Test with self parameter (instance method)
        signature = FunctionSignature(
            name="test_method",
            parameters=[
                FunctionParameter(name="self"),
                FunctionParameter(name="param1", type_annotation="str"),
            ],
            is_method=True,
        )

        function = ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

        filtered = generator._filter_special_parameters(signature.parameters, function)

        assert len(filtered) == 1
        assert filtered[0].name == "param1"

    def test_find_parameter_mismatches(self, generator: Any) -> None:
        """Test finding parameter mismatches with fuzzy matching."""
        actual_params = [
            FunctionParameter(name="username", type_annotation="str"),
            FunctionParameter(name="password", type_annotation="str"),
        ]

        documented_params = [
            DocstringParameter(
                name="email", description="User email"
            ),  # Should match username
            DocstringParameter(name="password", description="User password"),
        ]

        with patch("rapidfuzz.fuzz.ratio") as mock_ratio:
            # Mock fuzzy matching to return high similarity for email->username
            mock_ratio.return_value = 75

            mismatches = generator._find_parameter_mismatches(
                actual_params, documented_params
            )

        assert len(mismatches) == 1
        assert mismatches[0] == ("email", "username")

    def test_types_differ(self, generator: Any) -> None:
        """Test type difference detection."""
        # Same types
        assert not generator._types_differ("str", "str")

        # Different types
        assert generator._types_differ("str", "int")

        # One None
        assert generator._types_differ("str", None)
        assert generator._types_differ(None, "str")

        # Both None
        assert not generator._types_differ(None, None)

    def test_normalize_type(self, generator: Any) -> None:
        """Test type normalization."""
        assert generator._normalize_type("str") == "str"
        assert generator._normalize_type("List[str]") == "list"
        assert generator._normalize_type("Optional[str]") == "str"
        assert generator._normalize_type("Optional[str]") == "str"

    def test_generate_parameter_description(self, generator: Any) -> None:
        """Test generating basic parameter descriptions."""
        param = FunctionParameter(
            name="test_param", type_annotation="str", default_value="'default'"
        )

        description = generator._generate_parameter_description(param)

        assert "test_param" in description
        assert "str" in description
        assert "default" in description

    def test_detect_style_from_docstring(self, generator: Any) -> None:
        """Test detecting docstring style."""
        google_docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE, summary="Test"
        )

        style = generator._detect_style(google_docstring)
        assert style == DocstringStyle.GOOGLE

    def test_fallback_suggestion(self, generator: Any) -> None:
        """Test fallback suggestion for unknown issues."""
        issue = InconsistencyIssue(
            issue_type="unknown_issue",
            severity="low",
            description="Unknown issue",
            suggestion="Manual fix needed",
            line_number=10,
        )

        context = SuggestionContext(issue=issue, function=Mock(), docstring=Mock())

        suggestion = generator.generate(context)

        assert suggestion.confidence <= 0.1
        assert "unknown_issue" in suggestion.description

    def test_generic_parameter_fix(self, generator: Any) -> None:
        """Test generic parameter fix for unhandled issue types."""
        issue = InconsistencyIssue(
            issue_type="unhandled_parameter_issue",
            severity="medium",
            description="Unhandled issue",
            suggestion="Fix manually",
            line_number=10,
        )

        context = SuggestionContext(issue=issue, function=Mock(), docstring=Mock())

        suggestion = generator._generic_parameter_fix(context)

        assert suggestion.confidence <= 0.1
        assert "unhandled_parameter_issue" in suggestion.description


class TestParameterGeneratorEdgeCases:
    """Test edge cases for parameter generator."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create parameter generator instance."""
        config = SuggestionConfig(default_style="google")
        return ParameterSuggestionGenerator(config)

    def test_empty_function_parameters(self, generator: Any) -> None:
        """Test handling function with no parameters."""
        signature = FunctionSignature(name="test_function", parameters=[])
        function = ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

        actual_params = generator._get_function_parameters(function)
        assert actual_params == []

    def test_empty_documented_parameters(self, generator: Any) -> None:
        """Test handling docstring with no parameters."""
        docstring = ParsedDocstring(
            format=DocstringFormat.GOOGLE, summary="Test function", parameters=[]
        )

        documented_params = generator._get_documented_parameters(docstring)
        assert documented_params == []

    def test_malformed_function_object(self, generator: Any) -> None:
        """Test handling malformed function object."""
        malformed_function: Mock = Mock()
        # Remove signature attribute to simulate malformed object
        del malformed_function.signature

        actual_params = generator._get_function_parameters(malformed_function)
        assert actual_params == []

    def test_malformed_docstring_object(self, generator: Any) -> None:
        """Test handling malformed docstring object."""
        malformed_docstring: Mock = Mock()
        # Remove parameters attribute
        del malformed_docstring.parameters

        documented_params = generator._get_documented_parameters(malformed_docstring)
        assert documented_params == []

    def test_classmethod_detection(self, generator: Any) -> None:
        """Test detection of classmethod decorator."""
        signature = FunctionSignature(
            name="test_classmethod",
            decorators=["classmethod"],
            parameters=[
                FunctionParameter(name="cls"),
                FunctionParameter(name="param1", type_annotation="str"),
            ],
        )

        function = ParsedFunction(
            signature=signature, docstring=None, file_path="test.py", line_number=10
        )

        assert generator._is_classmethod(function)

        # Test filtering cls parameter
        filtered = generator._filter_special_parameters(signature.parameters, function)
        assert len(filtered) == 1
        assert filtered[0].name == "param1"