"""Tests for Google-style docstring template."""

import pytest

from codedocsync.parser.docstring_models import (
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
)
from codedocsync.suggestions.templates import GoogleStyleTemplate


class TestGoogleStyleTemplate:
    """Test Google-style template functionality."""

    @pytest.fixture
    def template(self):
        """Create Google template instance."""
        return GoogleStyleTemplate()

    def test_render_parameters_simple(self, template) -> None:
        """Test rendering simple parameters."""
        parameters = [
            DocstringParameter(
                name="param1", type_str="str", description="First parameter"
            ),
            DocstringParameter(
                name="param2",
                type_str="int",
                description="Second parameter",
                is_optional=True,
            ),
        ]

        result = template.render_parameters(parameters)

        assert result[0] == "Args:"
        assert "    param1 (str): First parameter" in result
        assert "    param2 (int, optional): Second parameter" in result

    def test_render_parameters_empty(self, template) -> None:
        """Test rendering empty parameters list."""
        result = template.render_parameters([])
        assert result == []

    def test_render_parameters_no_type(self, template) -> None:
        """Test rendering parameters without type annotations."""
        parameters = [
            DocstringParameter(name="param", description="Parameter without type")
        ]

        result = template.render_parameters(parameters)

        assert result[0] == "Args:"
        assert "    param: Parameter without type" in result

    def test_render_parameters_long_description(self, template) -> None:
        """Test parameter with long description that needs wrapping."""
        parameters = [
            DocstringParameter(
                name="long_param",
                type_str="str",
                description="This is a very long parameter description that should be wrapped across multiple lines to test the text wrapping functionality of the template system",
            )
        ]

        result = template.render_parameters(parameters)

        assert result[0] == "Args:"
        assert any("long_param (str):" in line for line in result)
        # Should have continuation lines with proper indentation
        assert any(line.startswith("        ") for line in result[2:])

    def test_render_returns_simple(self, template) -> None:
        """Test rendering simple return documentation."""
        returns = DocstringReturns(type_str="bool", description="Success status")

        result = template.render_returns(returns)

        assert result[0] == "Returns:"
        assert "    bool: Success status" in result

    def test_render_returns_no_type(self, template) -> None:
        """Test rendering return without type."""
        returns = DocstringReturns(description="Something useful")

        result = template.render_returns(returns)

        assert result[0] == "Returns:"
        assert "    Something useful" in result

    def test_render_returns_empty(self, template) -> None:
        """Test rendering empty returns."""
        returns = DocstringReturns()
        result = template.render_returns(returns)
        assert result == []

    def test_render_raises_simple(self, template) -> None:
        """Test rendering simple exception documentation."""
        raises = [
            DocstringRaises(
                exception_type="ValueError", description="When value is invalid"
            ),
            DocstringRaises(
                exception_type="TypeError", description="When type is wrong"
            ),
        ]

        result = template.render_raises(raises)

        assert result[0] == "Raises:"
        assert "    ValueError: When value is invalid" in result
        assert "    TypeError: When type is wrong" in result

    def test_render_raises_empty(self, template) -> None:
        """Test rendering empty raises list."""
        result = template.render_raises([])
        assert result == []

    def test_render_complete_docstring(self, template) -> None:
        """Test rendering complete docstring with all sections."""
        parameters = [
            DocstringParameter(name="param1", type_str="str", description="First param")
        ]
        returns = DocstringReturns(type_str="bool", description="Success")
        raises = [
            DocstringRaises(exception_type="ValueError", description="Invalid input")
        ]
        examples = [">>> example_function('test')\nTrue"]

        result = template.render_complete_docstring(
            summary="Function summary",
            description="Detailed description",
            parameters=parameters,
            returns=returns,
            raises=raises,
            examples=examples,
        )

        lines = result.split("\n")

        # Check structure
        assert lines[0] == '"""'
        assert lines[-1] == '"""'
        assert "Function summary" in result
        assert "Detailed description" in result
        assert "Args:" in result
        assert "Returns:" in result
        assert "Raises:" in result
        assert "Examples:" in result

    def test_format_parameter_line_with_type(self, template) -> None:
        """Test formatting parameter line with type."""
        param = DocstringParameter(
            name="test_param", type_str="str", description="Test description"
        )

        result = template._format_parameter_line(param)
        assert result == "test_param (str): Test description"

    def test_format_parameter_line_without_type(self, template) -> None:
        """Test formatting parameter line without type."""
        param = DocstringParameter(name="test_param", description="Test description")

        result = template._format_parameter_line(param)
        assert result == "test_param: Test description"

    def test_format_parameter_line_no_description(self, template) -> None:
        """Test formatting parameter line without description."""
        param = DocstringParameter(name="test_param", type_str="str")

        result = template._format_parameter_line(param)
        assert result == "test_param (str):"

    def test_match_parameter_line(self, template) -> None:
        """Test matching parameter lines to extract names."""
        test_cases = [
            ("    param_name (str): Description", "param_name"),
            ("    param: Description", "param"),
            ("    *args: Variable arguments", "*args"),
            ("    **kwargs: Keyword arguments", "**kwargs"),
            ("    Regular line without parameter", None),
        ]

        for line, expected in test_cases:
            result = template._match_parameter_line(line)
            assert result == expected

    def test_extract_section_boundaries(self, template) -> None:
        """Test extracting section boundaries from docstring."""
        docstring_lines = [
            '"""Function docstring.',
            "",
            "Args:",
            "    param1: First parameter",
            "    param2: Second parameter",
            "",
            "Returns:",
            "    Success status",
            "",
            "Raises:",
            "    ValueError: Invalid input",
            '"""',
        ]

        boundaries = template.extract_section_boundaries(docstring_lines)

        assert "parameters" in boundaries
        assert "returns" in boundaries
        assert "raises" in boundaries

        # Check parameter section boundaries
        params_start, params_end = boundaries["parameters"]
        assert params_start == 2  # "Args:" line
        assert params_end == 4  # Last parameter line

    def test_replace_section(self, template) -> None:
        """Test replacing a section in docstring."""
        original_lines = [
            '"""Function docstring.',
            "",
            "Args:",
            "    old_param: Old parameter",
            "",
            "Returns:",
            "    Something",
            '"""',
        ]

        new_section_lines = [
            "Args:",
            "    new_param: New parameter",
            "    another_param: Another parameter",
        ]

        result = template.replace_section(
            original_lines, "parameters", new_section_lines
        )

        # Should have replaced the Args section
        assert "Args:" in result
        assert "new_param: New parameter" in "\n".join(result)
        assert "another_param: Another parameter" in "\n".join(result)
        assert "old_param: Old parameter" not in "\n".join(result)
        # Should preserve Returns section
        assert "Returns:" in result

    def test_template_with_max_line_length(self) -> None:
        """Test template respects max line length setting."""
        template = GoogleStyleTemplate(max_line_length=50)

        parameters = [
            DocstringParameter(
                name="param",
                type_str="str",
                description="This is a very long description that should be wrapped",
            )
        ]

        result = template.render_parameters(parameters)

        # Check that lines don't exceed max length (accounting for indentation)
        for line in result:
            assert len(line) <= 50 or line.strip() == ""


class TestTemplateRegistry:
    """Test template registry functionality."""

    def test_register_and_get_template(self) -> None:
        """Test registering and retrieving templates."""
        from codedocsync.suggestions.templates import DocstringStyle, template_registry

        # GoogleStyleTemplate should be registered
        template = template_registry.get_template(DocstringStyle.GOOGLE)
        assert isinstance(template, GoogleStyleTemplate)

    def test_available_styles(self) -> None:
        """Test getting available template styles."""
        from codedocsync.suggestions.templates import DocstringStyle, template_registry

        styles = template_registry.available_styles()
        assert DocstringStyle.GOOGLE in styles

    def test_invalid_style_raises_error(self) -> None:
        """Test that invalid style raises error."""
        from codedocsync.suggestions.templates import DocstringStyle, template_registry

        with pytest.raises(ValueError):
            template_registry.get_template(DocstringStyle.NUMPY)  # Not registered yet


class TestTemplateIntegration:
    """Test template integration with real docstring scenarios."""

    def test_realistic_function_docstring(self) -> None:
        """Test generating realistic function docstring."""
        template = GoogleStyleTemplate()

        parameters = [
            DocstringParameter(
                name="data",
                type_str="List[Dict[str, Any]]",
                description="Input data to process",
            ),
            DocstringParameter(
                name="config",
                type_str="Optional[Config]",
                description="Configuration object",
                is_optional=True,
                default_value="None",
            ),
            DocstringParameter(
                name="verbose",
                type_str="bool",
                description="Enable verbose output",
                is_optional=True,
                default_value="False",
            ),
        ]

        returns = DocstringReturns(
            type_str="ProcessedData",
            description="Processed data object with validation results",
        )

        raises = [
            DocstringRaises(
                exception_type="ValidationError",
                description="When input data fails validation",
            ),
            DocstringRaises(
                exception_type="ConfigurationError",
                description="When configuration is invalid",
            ),
        ]

        result = template.render_complete_docstring(
            summary="Process input data with optional configuration.",
            description="This function validates and processes input data according to the provided configuration. It supports various input formats and provides detailed error reporting.",
            parameters=parameters,
            returns=returns,
            raises=raises,
        )

        # Verify structure and content
        assert '"""' in result
        assert "Process input data with optional configuration." in result
        assert "Args:" in result
        assert "data (List[Dict[str, Any]]): Input data to process" in result
        assert "config (Optional[Config], optional): Configuration object" in result
        assert "Returns:" in result
        assert "ProcessedData: Processed data object with validation results" in result
        assert "Raises:" in result
        assert "ValidationError: When input data fails validation" in result

        # Verify proper formatting
        lines = result.split("\n")
        assert lines[0] == '"""'
        assert lines[-1] == '"""'
