"""
Tests for Sphinx-style docstring template.

This module contains comprehensive tests for the SphinxStyleTemplate class,
ensuring proper field list formatting and Sphinx-specific features.
"""

import pytest

from codedocsync.parser.docstring_models import (
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
)
from codedocsync.suggestions.models import DocstringStyle
from codedocsync.suggestions.templates.sphinx_template import SphinxStyleTemplate


class TestSphinxStyleTemplate:
    """Test cases for Sphinx style template."""

    @pytest.fixture
    def template(self):
        """Create a Sphinx template instance."""
        return SphinxStyleTemplate()

    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for testing."""
        return [
            DocstringParameter(
                name="path",
                type_annotation="str",
                description="Path to the file to process",
                is_optional=False,
            ),
            DocstringParameter(
                name="encoding",
                type_annotation="str",
                description="Character encoding to use",
                is_optional=True,
                default_value="'utf-8'",
            ),
            DocstringParameter(
                name="mode",
                type_annotation="str",
                description="File mode for opening",
                is_optional=True,
                default_value="'r'",
            ),
        ]

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns for testing."""
        return DocstringReturns(
            type_annotation="List[str]",
            description="List of processed lines from the file",
        )

    @pytest.fixture
    def sample_raises(self):
        """Create sample raises for testing."""
        return [
            DocstringRaises(
                exception_type="FileNotFoundError",
                description="If the specified file does not exist",
            ),
            DocstringRaises(
                exception_type="PermissionError",
                description="If the file cannot be read due to permissions",
            ),
        ]

    def test_initialization(self):
        """Test template initialization."""
        template = SphinxStyleTemplate()
        assert template.style == DocstringStyle.SPHINX
        assert template.max_line_length == 88
        assert template.indent_size == 4

    def test_initialization_with_custom_params(self):
        """Test template initialization with custom parameters."""
        template = SphinxStyleTemplate(max_line_length=100)
        assert template.max_line_length == 100
        assert template.style == DocstringStyle.SPHINX

    def test_render_parameters_empty(self, template):
        """Test rendering empty parameters list."""
        result = template.render_parameters([])
        assert result == []

    def test_render_parameters_single(self, template):
        """Test rendering single parameter."""
        params = [
            DocstringParameter(
                name="value",
                type_annotation="int",
                description="Input value to process",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)
        expected = [
            ":param value: Input value to process",
            ":type value: int",
        ]

        assert result == expected

    def test_render_parameters_multiple(self, template, sample_parameters):
        """Test rendering multiple parameters."""
        result = template.render_parameters(sample_parameters)

        # Should have param and type entries for each parameter
        param_lines = [line for line in result if line.startswith(":param")]
        type_lines = [line for line in result if line.startswith(":type")]

        assert len(param_lines) == 3  # Three parameters
        assert len(type_lines) == 3  # Three type annotations

    def test_render_parameters_without_types(self, template):
        """Test rendering parameters without type annotations."""
        params = [
            DocstringParameter(
                name="param1",
                type_annotation=None,
                description="First parameter",
                is_optional=False,
            ),
            DocstringParameter(
                name="param2",
                type_annotation=None,
                description="Second parameter",
                is_optional=False,
            ),
        ]

        result = template.render_parameters(params)

        # Should have param lines but no type lines
        param_lines = [line for line in result if line.startswith(":param")]
        type_lines = [line for line in result if line.startswith(":type")]

        assert len(param_lines) == 2
        assert len(type_lines) == 0

    def test_render_parameters_without_descriptions(self, template):
        """Test rendering parameters without descriptions."""
        params = [
            DocstringParameter(
                name="param1",
                type_annotation="str",
                description=None,
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should still have type line
        type_lines = [line for line in result if line.startswith(":type")]
        assert len(type_lines) == 1
        assert ":param param1:" in result

    def test_render_returns_empty(self, template):
        """Test rendering empty returns."""
        assert template.render_returns(None) == []

    def test_render_returns_with_type(self, template, sample_returns):
        """Test rendering returns with type annotation."""
        result = template.render_returns(sample_returns)

        expected = [
            ":returns: List of processed lines from the file",
            ":rtype: list of str",  # List[str] should be formatted appropriately
        ]

        assert result == expected

    def test_render_returns_without_type(self, template):
        """Test rendering returns without type annotation."""
        returns = DocstringReturns(
            type_annotation=None,
            description="Some return value",
        )

        result = template.render_returns(returns)

        # Should only have returns line, no rtype
        assert ":returns: Some return value" in result
        assert not any(line.startswith(":rtype:") for line in result)

    def test_render_returns_without_description(self, template):
        """Test rendering returns without description."""
        returns = DocstringReturns(
            type_annotation="int",
            description=None,
        )

        result = template.render_returns(returns)

        # Should only have rtype line
        assert ":rtype: int" in result
        assert not any(line.startswith(":returns:") for line in result)

    def test_render_raises_empty(self, template):
        """Test rendering empty raises list."""
        assert template.render_raises([]) == []

    def test_render_raises_single(self, template):
        """Test rendering single exception."""
        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description="If input is invalid",
            )
        ]

        result = template.render_raises(raises)
        expected = [
            ":raises ValueError: If input is invalid",
        ]

        assert result == expected

    def test_render_raises_multiple(self, template, sample_raises):
        """Test rendering multiple exceptions."""
        result = template.render_raises(sample_raises)

        # Check that both exceptions are present
        assert (
            ":raises FileNotFoundError: If the specified file does not exist" in result
        )
        assert (
            ":raises PermissionError: If the file cannot be read due to permissions"
            in result
        )

    def test_render_raises_without_type(self, template):
        """Test rendering raises without exception type."""
        raises = [
            DocstringRaises(
                exception_type=None,
                description="If something goes wrong",
            )
        ]

        result = template.render_raises(raises)

        # Should use generic "Exception"
        assert ":raises Exception: If something goes wrong" in result

    def test_render_raises_without_description(self, template):
        """Test rendering raises without description."""
        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description=None,
            )
        ]

        result = template.render_raises(raises)

        # Should have empty description
        assert ":raises ValueError:" in result

    def test_wrap_sphinx_field_short_line(self, template):
        """Test wrapping of short Sphinx field lines."""
        field_line = ":param name: Short description"
        result = template._wrap_sphinx_field(field_line, ":param")

        assert result == [field_line]  # Should not be wrapped

    def test_wrap_sphinx_field_long_line(self, template):
        """Test wrapping of long Sphinx field lines."""
        template.max_line_length = 50  # Short line for testing
        field_line = ":param very_long_parameter_name: This is a very long description that definitely exceeds the maximum line length"

        result = template._wrap_sphinx_field(field_line, ":param")

        # Should be wrapped across multiple lines
        assert len(result) > 1
        # First line should start with the field
        assert result[0].startswith(":param very_long_parameter_name:")
        # Continuation lines should be indented
        for line in result[1:]:
            assert line.startswith(" " * len(":param very_long_parameter_name: "))

    def test_match_parameter_line(self, template):
        """Test parameter line matching."""
        # Valid parameter lines
        assert template._match_parameter_line(":param name: description") == "name"
        assert template._match_parameter_line(":param param_name:") == "param_name"
        assert template._match_parameter_line(":param x: desc") == "x"

        # Invalid lines
        assert template._match_parameter_line(":type name: str") is None
        assert template._match_parameter_line(":returns: value") is None
        assert template._match_parameter_line("regular text") is None

    def test_merge_descriptions(self, template):
        """Test merging preserved descriptions."""
        new_lines = [
            ":param param1: New description for param1",
            ":type param1: str",
            ":param param2: New description for param2",
            ":type param2: int",
        ]

        descriptions = {
            "param1": "Preserved description for param1",
        }

        result = template._merge_descriptions(new_lines, descriptions)

        # param1 should have preserved description
        assert ":param param1: Preserved description for param1" in result
        # param2 should keep new description
        assert ":param param2: New description for param2" in result

    def test_render_examples(self, template):
        """Test rendering examples."""
        examples = [
            "from mymodule import process\nresult = process(data)",
            "with open('file.txt') as f:\n    content = process(f.read())",
        ]

        result = template._render_examples(examples)

        # Should use Sphinx rubric and code-block directives
        assert ".. rubric:: Examples" in result
        assert ".. code-block:: python" in result
        assert "from mymodule import process" in result

    def test_format_type_annotation_simple_types(self, template):
        """Test formatting simple types for Sphinx style."""
        # Simple types should be preserved
        assert template._format_type_annotation("str") == "str"
        assert template._format_type_annotation("int") == "int"
        assert template._format_type_annotation("bool") == "bool"

    def test_format_type_annotation_complex_types(self, template):
        """Test formatting complex types for Sphinx style."""
        # Lists should become "list of type"
        assert template._format_type_annotation("List[str]") == "list of str"
        assert template._format_type_annotation("List[int]") == "list of int"

        # Dicts should become "dict"
        assert template._format_type_annotation("Dict[str, Any]") == "dict"

        # Optional should be handled by parent class
        assert template._format_type_annotation("Optional[str]") == "str, optional"

    def test_complete_docstring_rendering(
        self, template, sample_parameters, sample_returns, sample_raises
    ):
        """Test rendering a complete Sphinx docstring."""
        docstring = template.render_complete_docstring(
            summary="Process a file with the specified encoding.",
            description="This function reads a file and processes its contents line by line.",
            parameters=sample_parameters,
            returns=sample_returns,
            raises=sample_raises,
        )

        # Check that all sections are present
        assert '"""' in docstring
        assert "Process a file with the specified encoding." in docstring
        assert ":param path:" in docstring
        assert ":type path:" in docstring
        assert ":returns:" in docstring
        assert ":rtype:" in docstring
        assert ":raises FileNotFoundError:" in docstring

    def test_sphinx_specific_formatting(self, template):
        """Test Sphinx-specific formatting features."""
        # Test that field lists are properly formatted
        params = [
            DocstringParameter(
                name="config",
                type_annotation="Dict[str, Any]",
                description="Configuration dictionary with various settings",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should have proper Sphinx field syntax
        assert any(line.startswith(":param config:") for line in result)
        assert any(line.startswith(":type config:") for line in result)

    def test_long_parameter_names(self, template):
        """Test handling of very long parameter names."""
        params = [
            DocstringParameter(
                name="extremely_long_parameter_name_that_might_cause_issues",
                type_annotation="str",
                description="Description",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should handle gracefully
        assert any(
            "extremely_long_parameter_name_that_might_cause_issues" in line
            for line in result
        )


class TestSphinxTemplateEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def template(self):
        """Create template for edge case testing."""
        return SphinxStyleTemplate()

    def test_empty_field_descriptions(self, template):
        """Test handling of empty field descriptions."""
        params = [
            DocstringParameter(
                name="param",
                type_annotation="str",
                description="",  # Empty description
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should still render field with empty description
        assert ":param param:" in result
        assert ":type param: str" in result

    def test_special_characters_in_sphinx_fields(self, template):
        """Test handling of special characters in Sphinx fields."""
        params = [
            DocstringParameter(
                name="param",
                type_annotation="str",
                description="Description with :ref:`cross-reference` and **bold** text",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should preserve Sphinx markup
        rendered_text = " ".join(result)
        assert ":ref:`cross-reference`" in rendered_text
        assert "**bold**" in rendered_text

    def test_colon_in_descriptions(self, template):
        """Test handling of colons in descriptions (potential parsing issue)."""
        params = [
            DocstringParameter(
                name="param",
                type_annotation="str",
                description="Description with: colons in various: places",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should handle colons in descriptions properly
        rendered_text = " ".join(result)
        assert "Description with: colons in various: places" in rendered_text

    def test_multiline_descriptions_in_fields(self, template):
        """Test handling of multiline descriptions in Sphinx fields."""
        template.max_line_length = 40  # Force wrapping

        params = [
            DocstringParameter(
                name="data",
                type_annotation="str",
                description="This is a very long description that spans multiple lines and should be properly wrapped with continuation indentation",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should have proper continuation indentation
        param_lines = [
            line
            for line in result
            if line.startswith(":param data:")
            or line.startswith(" " * len(":param data: "))
        ]
        assert len(param_lines) > 1  # Should be wrapped

    def test_unicode_in_sphinx_fields(self, template):
        """Test handling of Unicode characters in Sphinx fields."""
        params = [
            DocstringParameter(
                name="αlpha",  # Unicode parameter name
                type_annotation="str",
                description="Parameter with émojis 🚀 and symbols α β γ",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should preserve Unicode
        rendered_text = " ".join(result)
        assert "αlpha" in rendered_text
        assert "émojis 🚀" in rendered_text
        assert "α β γ" in rendered_text

    def test_very_long_field_names(self, template):
        """Test field names that might cause formatting issues."""
        template.max_line_length = 60

        params = [
            DocstringParameter(
                name="parameter_with_extremely_long_name_that_exceeds_normal_limits",
                type_annotation="str",
                description="Short desc",
                is_optional=False,
            )
        ]

        result = template.render_parameters(params)

        # Should handle gracefully without breaking
        assert len(result) > 0
        rendered_text = " ".join(result)
        assert (
            "parameter_with_extremely_long_name_that_exceeds_normal_limits"
            in rendered_text
        )
