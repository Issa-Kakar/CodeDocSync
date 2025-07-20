"""
Tests for docstring style converter.

This module contains comprehensive tests for the DocstringStyleConverter class,
ensuring proper conversion between different docstring styles while preserving
information and applying appropriate formatting.
"""

import pytest

from codedocsync.suggestions.converter import (
    DocstringStyleConverter,
    convert_docstring,
    batch_convert_docstrings,
    ConversionPreset,
)
from codedocsync.suggestions.models import DocstringStyle
from codedocsync.parser.docstring_models import (
    ParsedDocstring,
    DocstringParameter,
    DocstringReturns,
    DocstringRaises,
)


class TestDocstringStyleConverter:
    """Test cases for DocstringStyleConverter."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return DocstringStyleConverter()

    @pytest.fixture
    def sample_parsed_docstring(self):
        """Create a sample parsed docstring for testing."""
        return ParsedDocstring(
            summary="Process input data using specified method.",
            description="This function processes the input data array along the specified axis using the given method. It supports multiple processing modes.",
            parameters=[
                DocstringParameter(
                    name="data",
                    type_annotation="np.ndarray",
                    description="Input data array with shape (n, m)",
                    is_optional=False,
                ),
                DocstringParameter(
                    name="axis",
                    type_annotation="int",
                    description="Axis along which to operate",
                    is_optional=True,
                    default_value="0",
                ),
                DocstringParameter(
                    name="method",
                    type_annotation="str",
                    description="Method to use for processing",
                    is_optional=True,
                    default_value="'mean'",
                ),
            ],
            returns=DocstringReturns(
                type_annotation="np.ndarray",
                description="Processed array with same shape as input",
            ),
            raises=[
                DocstringRaises(
                    exception_type="ValueError",
                    description="If input data has wrong shape",
                ),
                DocstringRaises(
                    exception_type="TypeError",
                    description="If input is not a numpy array",
                ),
            ],
            raw_text="",  # Original text not needed for conversion tests
            format="google",
        )

    def test_initialization(self):
        """Test converter initialization."""
        converter = DocstringStyleConverter()
        assert isinstance(converter._type_formatters, dict)
        assert isinstance(converter._conversion_stats, dict)
        assert converter._conversion_stats["conversions_performed"] == 0

    def test_convert_same_style(self, converter, sample_parsed_docstring):
        """Test conversion when source and target styles are the same."""
        result = converter.convert(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.GOOGLE
        )

        # Should reformat but preserve content
        assert '"""' in result
        assert "Process input data using specified method." in result
        assert "Args:" in result
        assert "Returns:" in result
        assert "Raises:" in result

    def test_convert_google_to_numpy(self, converter, sample_parsed_docstring):
        """Test conversion from Google to NumPy style."""
        result = converter.convert(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        # Check NumPy-specific formatting
        assert "Parameters" in result
        assert "-" * 10 in result  # Parameters underline
        assert "Returns" in result
        assert "-" * 7 in result  # Returns underline
        assert "Raises" in result
        assert "-" * 6 in result  # Raises underline

        # Check type conversion (np.ndarray should become array_like)
        assert "array_like" in result

    def test_convert_google_to_sphinx(self, converter, sample_parsed_docstring):
        """Test conversion from Google to Sphinx style."""
        result = converter.convert(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.SPHINX
        )

        # Check Sphinx-specific formatting
        assert ":param data:" in result
        assert ":type data:" in result
        assert ":returns:" in result
        assert ":rtype:" in result
        assert ":raises ValueError:" in result

    def test_convert_numpy_to_google(self, converter):
        """Test conversion from NumPy to Google style."""
        numpy_docstring = ParsedDocstring(
            summary="Calculate statistics.",
            description="Compute various statistics from the input data.",
            parameters=[
                DocstringParameter(
                    name="arr",
                    type_annotation="array_like",
                    description="Input array for calculations",
                    is_optional=False,
                ),
            ],
            returns=DocstringReturns(
                type_annotation="dict",
                description="Dictionary containing statistics",
            ),
            raises=None,
            raw_text="",
            format="numpy",
        )

        result = converter.convert(
            numpy_docstring, DocstringStyle.NUMPY, DocstringStyle.GOOGLE
        )

        # Check Google-specific formatting
        assert "Args:" in result
        assert "Returns:" in result
        assert "arr (array_like):" in result or "arr: " in result

    def test_convert_sphinx_to_google(self, converter):
        """Test conversion from Sphinx to Google style."""
        sphinx_docstring = ParsedDocstring(
            summary="Parse configuration file.",
            description="Load and parse a configuration file with validation.",
            parameters=[
                DocstringParameter(
                    name="filename",
                    type_annotation="str",
                    description="Path to configuration file",
                    is_optional=False,
                ),
                DocstringParameter(
                    name="validate",
                    type_annotation="bool",
                    description="Whether to validate the configuration",
                    is_optional=True,
                    default_value="True",
                ),
            ],
            returns=DocstringReturns(
                type_annotation="dict",
                description="Parsed configuration data",
            ),
            raises=[
                DocstringRaises(
                    exception_type="FileNotFoundError",
                    description="If configuration file doesn't exist",
                ),
            ],
            raw_text="",
            format="sphinx",
        )

        result = converter.convert(
            sphinx_docstring, DocstringStyle.SPHINX, DocstringStyle.GOOGLE
        )

        # Check Google formatting
        assert "Args:" in result
        assert "Returns:" in result
        assert "Raises:" in result
        assert "filename" in result
        assert "validate" in result

    def test_convert_with_custom_options(self, converter, sample_parsed_docstring):
        """Test conversion with custom options."""
        result = converter.convert(
            sample_parsed_docstring,
            DocstringStyle.GOOGLE,
            DocstringStyle.NUMPY,
            preserve_formatting=False,
            max_line_length=60,
        )

        # Should respect max line length
        lines = result.split("\n")
        # Most lines should be shorter than limit (allowing some flexibility for structure)
        long_lines = [line for line in lines if len(line) > 65]
        assert (
            len(long_lines) < len(lines) * 0.3
        )  # Less than 30% of lines should be long

    def test_convert_empty_sections(self, converter):
        """Test conversion with empty sections."""
        minimal_docstring = ParsedDocstring(
            summary="Simple function.",
            description=None,
            parameters=None,
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        result = converter.convert(
            minimal_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        # Should handle gracefully
        assert '"""' in result
        assert "Simple function." in result
        # Should not have empty sections
        assert "Parameters" not in result
        assert "Returns" not in result
        assert "Raises" not in result

    def test_convert_batch_success(self, converter):
        """Test successful batch conversion."""
        docstrings = [
            ParsedDocstring(
                summary="Function one.",
                description=None,
                parameters=[
                    DocstringParameter(
                        name="x",
                        type_annotation="int",
                        description="First parameter",
                        is_optional=False,
                    )
                ],
                returns=None,
                raises=None,
                raw_text="",
                format="google",
            ),
            ParsedDocstring(
                summary="Function two.",
                description=None,
                parameters=[
                    DocstringParameter(
                        name="y",
                        type_annotation="str",
                        description="Second parameter",
                        is_optional=False,
                    )
                ],
                returns=None,
                raises=None,
                raw_text="",
                format="google",
            ),
        ]

        results = converter.convert_batch(
            docstrings, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        assert len(results) == 2
        assert all(result is not None for result in results)
        assert all("Parameters" in result for result in results)

    def test_convert_batch_with_failures(self, converter):
        """Test batch conversion with some failures."""
        # Create a mock docstring that will cause conversion failure
        problematic_docstring = ParsedDocstring(
            summary=None,  # This might cause issues
            description=None,
            parameters=None,
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        good_docstring = ParsedDocstring(
            summary="Good function.",
            description=None,
            parameters=None,
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        docstrings = [problematic_docstring, good_docstring]

        # Should handle failures gracefully
        results = converter.convert_batch(
            docstrings, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        assert len(results) == 2
        # One might be None due to failure, one should succeed
        assert any(result is not None for result in results)

    def test_estimate_conversion_quality(self, converter, sample_parsed_docstring):
        """Test conversion quality estimation."""
        quality = converter.estimate_conversion_quality(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        assert isinstance(quality, dict)
        assert "information_loss_risk" in quality
        assert "formatting_changes" in quality
        assert "confidence" in quality
        assert "warnings" in quality
        assert isinstance(quality["confidence"], float)
        assert 0.0 <= quality["confidence"] <= 1.0

    def test_estimate_quality_complex_types(self, converter):
        """Test quality estimation with complex types."""
        complex_docstring = ParsedDocstring(
            summary="Complex function.",
            description=None,
            parameters=[
                DocstringParameter(
                    name="callback",
                    type_annotation="Callable[[int, str], bool]",
                    description="Complex callback function",
                    is_optional=False,
                )
            ],
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        quality = converter.estimate_conversion_quality(
            complex_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        # Should detect complex types and adjust confidence
        assert quality["confidence"] < 1.0
        assert len(quality["warnings"]) > 0

    def test_get_conversion_statistics(self, converter, sample_parsed_docstring):
        """Test getting conversion statistics."""
        # Perform some conversions
        converter.convert(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )
        converter.convert(
            sample_parsed_docstring, DocstringStyle.GOOGLE, DocstringStyle.SPHINX
        )

        stats = converter.get_conversion_statistics()

        assert isinstance(stats, dict)
        assert "conversions_performed" in stats
        assert stats["conversions_performed"] == 2

    def test_type_formatter_caching(self, converter):
        """Test that type formatters are cached."""
        # Access formatter multiple times
        formatter1 = converter._get_type_formatter(DocstringStyle.NUMPY)
        formatter2 = converter._get_type_formatter(DocstringStyle.NUMPY)

        # Should be the same instance
        assert formatter1 is formatter2

    def test_convert_parameters_with_none(self, converter):
        """Test parameter conversion with None input."""
        result = converter._convert_parameters(
            None, converter._get_type_formatter(DocstringStyle.GOOGLE)
        )
        assert result is None

    def test_convert_returns_with_none(self, converter):
        """Test returns conversion with None input."""
        result = converter._convert_returns(
            None, converter._get_type_formatter(DocstringStyle.GOOGLE)
        )
        assert result is None

    def test_convert_raises_with_none(self, converter):
        """Test raises conversion with None input."""
        result = converter._convert_raises(
            None, converter._get_type_formatter(DocstringStyle.GOOGLE)
        )
        assert result is None


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def sample_docstring(self):
        """Create sample docstring for testing."""
        return ParsedDocstring(
            summary="Test function.",
            description="A simple test function.",
            parameters=[
                DocstringParameter(
                    name="param",
                    type_annotation="str",
                    description="Test parameter",
                    is_optional=False,
                )
            ],
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

    def test_convert_docstring_function(self, sample_docstring):
        """Test the convert_docstring convenience function."""
        result = convert_docstring(
            sample_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        assert isinstance(result, str)
        assert "Parameters" in result
        assert "param : str" in result

    def test_batch_convert_docstrings_function(self, sample_docstring):
        """Test the batch_convert_docstrings convenience function."""
        docstrings = [sample_docstring, sample_docstring]

        results = batch_convert_docstrings(
            docstrings, DocstringStyle.GOOGLE, DocstringStyle.SPHINX
        )

        assert len(results) == 2
        assert all(isinstance(result, str) for result in results if result is not None)
        assert all(
            ":param param:" in result for result in results if result is not None
        )


class TestConversionPresets:
    """Test conversion preset configurations."""

    def test_scientific_to_api_preset(self):
        """Test scientific to API conversion preset."""
        preset = ConversionPreset.scientific_to_api()

        assert isinstance(preset, dict)
        assert "preserve_formatting" in preset
        assert "max_line_length" in preset
        assert preset["max_line_length"] == 88

    def test_api_to_sphinx_preset(self):
        """Test API to Sphinx conversion preset."""
        preset = ConversionPreset.api_to_sphinx()

        assert isinstance(preset, dict)
        assert "max_line_length" in preset
        assert preset["max_line_length"] == 100  # Sphinx uses longer lines

    def test_legacy_cleanup_preset(self):
        """Test legacy cleanup preset."""
        preset = ConversionPreset.legacy_cleanup()

        assert isinstance(preset, dict)
        assert "preserve_formatting" in preset
        assert preset["preserve_formatting"] is False  # Reformat everything


class TestErrorHandling:
    """Test error handling in conversion."""

    @pytest.fixture
    def converter(self):
        """Create converter for error testing."""
        return DocstringStyleConverter()

    def test_conversion_with_invalid_style(self, converter):
        """Test conversion with unsupported style combination."""
        docstring = ParsedDocstring(
            summary="Test",
            description=None,
            parameters=None,
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        # This should work but might have warnings in logs
        try:
            result = converter.convert(
                docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
            )
            assert isinstance(result, str)
        except Exception as e:
            # If it fails, should be a ValueError with descriptive message
            assert isinstance(e, ValueError)
            assert "conversion" in str(e).lower()

    def test_conversion_with_malformed_docstring(self, converter):
        """Test conversion with malformed docstring data."""
        # Create docstring with inconsistent data
        malformed_docstring = ParsedDocstring(
            summary="",  # Empty summary
            description="",  # Empty description
            parameters=[],  # Empty parameters
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        # Should handle gracefully
        result = converter.convert(
            malformed_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        assert isinstance(result, str)
        assert '"""' in result


class TestSpecialCases:
    """Test special cases and edge conditions."""

    @pytest.fixture
    def converter(self):
        """Create converter for special case testing."""
        return DocstringStyleConverter()

    def test_convert_with_unicode_content(self, converter):
        """Test conversion with Unicode content."""
        unicode_docstring = ParsedDocstring(
            summary="Función con caracteres especiales 🚀.",
            description="Procesa datos con símbolos α, β, γ.",
            parameters=[
                DocstringParameter(
                    name="données",  # French
                    type_annotation="str",
                    description="Données d'entrée avec émojis 📊",
                    is_optional=False,
                )
            ],
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        result = converter.convert(
            unicode_docstring, DocstringStyle.GOOGLE, DocstringStyle.NUMPY
        )

        # Should preserve Unicode
        assert "🚀" in result
        assert "α, β, γ" in result
        assert "données" in result
        assert "📊" in result

    def test_convert_with_very_long_descriptions(self, converter):
        """Test conversion with very long descriptions."""
        long_description = "This is a very long description that goes on and on " * 20

        long_docstring = ParsedDocstring(
            summary="Function with long description.",
            description=long_description,
            parameters=[
                DocstringParameter(
                    name="param",
                    type_annotation="str",
                    description=long_description,
                    is_optional=False,
                )
            ],
            returns=DocstringReturns(
                type_annotation="str",
                description=long_description,
            ),
            raises=None,
            raw_text="",
            format="google",
        )

        result = converter.convert(
            long_docstring,
            DocstringStyle.GOOGLE,
            DocstringStyle.NUMPY,
            max_line_length=80,
        )

        # Should handle long content and wrap appropriately
        assert isinstance(result, str)
        lines = result.split("\n")
        # Most lines should respect line length
        very_long_lines = [line for line in lines if len(line) > 100]
        assert len(very_long_lines) < len(lines) * 0.2  # Less than 20% very long

    def test_convert_with_special_parameter_types(self, converter):
        """Test conversion with special parameter types (*args, **kwargs)."""
        special_docstring = ParsedDocstring(
            summary="Function with special parameters.",
            description=None,
            parameters=[
                DocstringParameter(
                    name="*args",
                    type_annotation="Any",
                    description="Variable positional arguments",
                    is_optional=True,
                ),
                DocstringParameter(
                    name="**kwargs",
                    type_annotation="Any",
                    description="Variable keyword arguments",
                    is_optional=True,
                ),
            ],
            returns=None,
            raises=None,
            raw_text="",
            format="google",
        )

        result = converter.convert(
            special_docstring, DocstringStyle.GOOGLE, DocstringStyle.SPHINX
        )

        # Should handle special parameters
        assert "*args" in result
        assert "**kwargs" in result
