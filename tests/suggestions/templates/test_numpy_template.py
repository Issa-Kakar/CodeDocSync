"""
Tests for NumPy-style docstring template.

This module contains comprehensive tests for the NumpyStyleTemplate class,
ensuring proper formatting, parameter rendering, and style-specific features.
"""

from typing import Any

import pytest

from codedocsync.parser.docstring_models import (
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
)
from codedocsync.suggestions.models import DocstringStyle
from codedocsync.suggestions.templates.numpy_template import NumpyStyleTemplate


class TestNumpyStyleTemplate:
    """Test cases for NumPy style template."""

    @pytest.fixture
    def template(self) -> Any:
        """Create a NumPy template instance."""
        return NumpyStyleTemplate()

    @pytest.fixture
    def sample_parameters(self) -> Any:
        """Create sample parameters for testing."""
        return [
            DocstringParameter(
                name="data",
                type_str="np.ndarray",
                description="Input data array with shape (n, m)",
                is_optional=False,
            ),
            DocstringParameter(
                name="axis",
                type_str="int",
                description="Axis along which to operate",
                is_optional=True,
                default_value="0",
            ),
            DocstringParameter(
                name="method",
                type_str="str",
                description="Method to use for processing",
                is_optional=True,
                default_value="'mean'",
            ),
        ]

    @pytest.fixture
    def sample_returns(self) -> Any:
        """Create sample returns for testing."""
        return DocstringReturns(
            type_str="np.ndarray",
            description="Processed array with same shape as input",
        )

    @pytest.fixture
    def sample_raises(self) -> Any:
        """Create sample raises for testing."""
        return [
            DocstringRaises(
                exception_type="ValueError",
                description="If input data has wrong shape",
            ),
            DocstringRaises(
                exception_type="TypeError",
                description="If input is not a numpy array",
            ),
        ]

    def test_initialization(self) -> None:
        """Test template initialization."""
        template = NumpyStyleTemplate()
        assert template.style == DocstringStyle.NUMPY
        assert template.max_line_length == 88
        assert template.indent_size == 4

    def test_initialization_with_custom_params(self) -> None:
        """Test template initialization with custom parameters."""
        template = NumpyStyleTemplate(max_line_length=100)
        assert template.max_line_length == 100
        assert template.style == DocstringStyle.NUMPY

    def test_render_parameters_empty(self, template: Any) -> None:
        """Test rendering empty parameters list."""
        result = template.render_parameters([])
        assert result == []

    def test_render_parameters_single(self, template: Any) -> None:
        """Test rendering single parameter."""
        params = [
            DocstringParameter(
                name="x",
                type_str="int",
                description="Input value",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)
        expected = [
            "Parameters",
            "-" * 10,
            "x : int",
            "    Input value",
        ]

        assert result == expected

    def test_render_parameters_multiple(
        self, template: Any, sample_parameters: Any
    ) -> None:
        """Test rendering multiple parameters."""
        result = template.render_parameters(sample_parameters)

        # Check structure
        assert result[0] == "Parameters"
        assert result[1] == "-" * 10

        # Check that parameters are separated by blank lines
        parameter_starts = []
        for i, line in enumerate(result[2:], 2):
            if line and not line.startswith("    ") and line.strip():
                parameter_starts.append(i)

        # Should have 3 parameters with blank lines between them
        assert len(parameter_starts) == 3

    def test_render_parameters_without_types(self, template: Any) -> None:
        """Test rendering parameters without type annotations."""
        params = [
            DocstringParameter(
                name="param1",
                type_str=None,
                description="First parameter",
                is_optional=False,
            ),
            DocstringParameter(
                name="param2",
                type_str=None,
                description="Second parameter",
                is_optional=False,
            ),
        ]

        result = template.render_parameters(params)

        # Should still have proper structure
        assert "Parameters" in result
        assert "-" * 10 in result
        assert "param1" in result
        assert "param2" in result

    def test_render_parameters_long_description(self, template: Any) -> None:
        """Test rendering parameters with long descriptions."""
        params = [
            DocstringParameter(
                name="very_long_parameter_name",
                type_str="Dict[str, Any]",
                description="This is a very long description that should be wrapped across multiple lines to test the wrapping functionality of the NumPy template",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Check that description is wrapped
        description_lines = [line for line in result if line.startswith("    ")]
        assert len(description_lines) > 1  # Should be wrapped across multiple lines

    def test_render_returns_empty(self, template: Any) -> None:
        """Test rendering empty returns."""
        assert template.render_returns(None) == []

    def test_render_returns_with_type(self, template: Any, sample_returns: Any) -> None:
        """Test rendering returns with type annotation."""
        result = template.render_returns(sample_returns)

        expected = [
            "Returns",
            "-" * 7,
            "result : array_like",  # np.ndarray should be formatted as array_like
            "    Processed array with same shape as input",
        ]

        assert result == expected

    def test_render_returns_without_type(self, template: Any) -> None:
        """Test rendering returns without type annotation."""
        returns = DocstringReturns(
            type_str=None,
            description="Some return value",
        )

        result = template.render_returns(returns)

        assert "Returns" in result
        assert "-" * 7 in result
        assert "result" in result  # Should use default name

    def test_render_raises_empty(self, template: Any) -> None:
        """Test rendering empty raises list."""
        assert template.render_raises([]) == []

    def test_render_raises_single(self, template: Any) -> None:
        """Test rendering single exception."""
        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description="If input is invalid",
            )
        ]

        result = template.render_raises(raises)
        expected = [
            "Raises",
            "-" * 6,
            "ValueError",
            "    If input is invalid",
        ]

        assert result == expected

    def test_render_raises_multiple(self, template: Any, sample_raises: Any) -> None:
        """Test rendering multiple exceptions."""
        result = template.render_raises(sample_raises)

        # Check structure
        assert result[0] == "Raises"
        assert result[1] == "-" * 6

        # Check that exceptions are present
        assert "ValueError" in result
        assert "TypeError" in result

        # Check that there are blank lines between exceptions
        blank_lines = [i for i, line in enumerate(result) if line == ""]
        assert len(blank_lines) >= 1  # At least one blank line between exceptions

    def test_render_raises_without_type(self, template: Any) -> None:
        """Test rendering raises without exception type."""
        raises = [
            DocstringRaises(
                exception_type="Exception",
                description="If something goes wrong",
            )
        ]

        result = template.render_raises(raises)

        # Should use generic "Exception"
        assert "Exception" in result
        assert "If something goes wrong" in result

    def test_format_parameter_line_with_type(self, template: Any) -> None:
        """Test formatting parameter line with type."""
        param = DocstringParameter(
            name="test_param",
            type_str="List[str]",
            description="Test parameter",
            is_optional=False,
        )

        result = template._format_parameter_line(param)
        assert (
            result == "test_param : list of str"
        )  # Should format List[str] appropriately

    def test_format_parameter_line_without_type(self, template: Any) -> None:
        """Test formatting parameter line without type."""
        param = DocstringParameter(
            name="test_param",
            type_str=None,
            description="Test parameter",
            is_optional=False,
        )

        result = template._format_parameter_line(param)
        assert result == "test_param"

    def test_format_return_line_with_type(self, template: Any) -> None:
        """Test formatting return line with type."""
        returns = DocstringReturns(
            type_str="int",
            description="Return value",
        )

        result = template._format_return_line(returns)
        assert result == "result : int"

    def test_format_return_line_without_type(self, template: Any) -> None:
        """Test formatting return line without type."""
        returns = DocstringReturns(
            type_str=None,
            description="Return value",
        )

        result = template._format_return_line(returns)
        assert result == "result"

    def test_match_parameter_line(self, template: Any) -> None:
        """Test parameter line matching."""
        # Valid parameter lines
        assert template._match_parameter_line("param_name : str") == "param_name"
        assert template._match_parameter_line("param") == "param"
        assert template._match_parameter_line("param_name : List[str]") == "param_name"

        # Invalid lines
        assert template._match_parameter_line("    description") is None
        assert template._match_parameter_line("Parameters") is None
        assert template._match_parameter_line("") is None

    def test_merge_descriptions(self, template: Any) -> None:
        """Test merging preserved descriptions."""
        new_lines = [
            "Parameters",
            "-" * 10,
            "param1 : str",
            "    New description for param1",
            "",
            "param2 : int",
            "    New description for param2",
        ]

        descriptions = {
            "param1": "Preserved description for param1",
        }

        result = template._merge_descriptions(new_lines, descriptions)

        # param1 should have preserved description
        assert "Preserved description for param1" in " ".join(result)
        # param2 should keep new description
        assert "New description for param2" in " ".join(result)

    def test_render_examples(self, template: Any) -> None:
        """Test rendering examples."""
        examples = [
            ">>> import numpy as np\n>>> data = np.array([1, 2, 3])\n>>> process(data)",
            ">>> result = process(data, axis=1)\n>>> print(result.shape)",
        ]

        result = template._render_examples(examples)

        assert "Examples" in result
        assert "-" * 8 in result
        assert ">>> import numpy as np" in result

    def test_format_type_annotation_array_types(self, template: Any) -> None:
        """Test formatting array types for NumPy style."""
        # NumPy arrays should become array_like
        assert template._format_type_annotation("np.ndarray") == "array_like"
        assert template._format_type_annotation("ndarray") == "array_like"

        # Lists should become list of type
        assert template._format_type_annotation("List[str]") == "list of str"
        assert template._format_type_annotation("List[int]") == "list of int"

        # Dicts should become dict of type
        assert template._format_type_annotation("Dict[str, Any]") == "dict of Any"

    def test_complete_docstring_rendering(
        self,
        template: Any,
        sample_parameters: Any,
        sample_returns: Any,
        sample_raises: Any,
    ) -> None:
        """Test rendering a complete docstring."""
        docstring = template.render_complete_docstring(
            summary="Process input data using specified method.",
            description="This function processes the input data array along the specified axis using the given method.",
            parameters=sample_parameters,
            returns=sample_returns,
            raises=sample_raises,
        )

        # Check that all sections are present
        assert '"""' in docstring
        assert "Process input data using specified method." in docstring
        assert "Parameters" in docstring
        assert "Returns" in docstring
        assert "Raises" in docstring

        # Check NumPy-specific formatting
        assert "-" * 10 in docstring  # Parameters underline
        assert "-" * 7 in docstring  # Returns underline
        assert "-" * 6 in docstring  # Raises underline

    def test_long_line_wrapping(self, template: Any) -> None:
        """Test that long lines are properly wrapped."""
        template.max_line_length = 50  # Short line for testing

        params = [
            DocstringParameter(
                name="very_long_parameter_name_that_exceeds_limit",
                type_str="Dict[str, List[Tuple[int, str]]]",
                description="This is an extremely long description that definitely exceeds the maximum line length and should be wrapped properly across multiple lines",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Check that no line exceeds the limit (accounting for indentation)
        for line in result:
            assert len(line) <= template.max_line_length or line.strip() == ""


class TestNumpyTemplateEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def template(self) -> Any:
        """Create template for edge case testing."""
        return NumpyStyleTemplate()

    def test_empty_descriptions(self, template: Any) -> None:
        """Test handling of empty descriptions."""
        params = [
            DocstringParameter(
                name="param",
                type_str="str",
                description="",  # Empty description
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should still render parameter name and type
        assert "param : str" in result

    def test_special_characters_in_descriptions(self, template: Any) -> None:
        """Test handling of special characters."""
        params = [
            DocstringParameter(
                name="param",
                type_str="str",
                description="Description with \"quotes\" and 'apostrophes' and <brackets>",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should preserve special characters
        rendered_text = " ".join(result)
        assert '"quotes"' in rendered_text
        assert "'apostrophes'" in rendered_text
        assert "<brackets>" in rendered_text

    def test_unicode_descriptions(self, template: Any) -> None:
        """Test handling of Unicode characters."""
        params = [
            DocstringParameter(
                name="param",
                type_str="str",
                description="Parameter with émoji 🚀 and unicode characters α β γ",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should preserve Unicode
        rendered_text = " ".join(result)
        assert "émoji 🚀" in rendered_text
        assert "α β γ" in rendered_text

    def test_very_long_parameter_names(self, template: Any) -> None:
        """Test handling of very long parameter names."""
        params = [
            DocstringParameter(
                name="extremely_long_parameter_name_that_might_cause_formatting_issues",
                type_str="str",
                description="Description",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should handle gracefully
        assert (
            "extremely_long_parameter_name_that_might_cause_formatting_issues"
            in " ".join(result)
        )

    def test_complex_type_annotations(self, template: Any) -> None:
        """Test handling of complex type annotations."""
        params = [
            DocstringParameter(
                name="callback",
                type_str="Callable[[int, str], Optional[bool]]",
                description="Complex callback function",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should handle complex types gracefully
        assert "callback" in " ".join(result)
        # Type should be simplified or preserved depending on implementation
        rendered_text = " ".join(result)
        assert "Callable" in rendered_text or "callable" in rendered_text
