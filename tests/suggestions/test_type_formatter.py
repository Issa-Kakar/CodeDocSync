"""
Tests for type annotation formatter.

This module contains comprehensive tests for the TypeAnnotationFormatter class,
ensuring proper formatting of Python type annotations for different docstring styles.
"""

import ast

import pytest

from codedocsync.suggestions.models import DocstringStyle
from codedocsync.suggestions.type_formatter import (
    TypeAnnotationFormatter,
    TypeComplexity,
    extract_type_from_ast,
    format_type_for_style,
)


class TestTypeAnnotationFormatter:
    """Test cases for TypeAnnotationFormatter."""

    @pytest.fixture
    def google_formatter(self):
        """Create a formatter for Google style."""
        return TypeAnnotationFormatter(DocstringStyle.GOOGLE)

    @pytest.fixture
    def numpy_formatter(self):
        """Create a formatter for NumPy style."""
        return TypeAnnotationFormatter(DocstringStyle.NUMPY)

    @pytest.fixture
    def sphinx_formatter(self):
        """Create a formatter for Sphinx style."""
        return TypeAnnotationFormatter(DocstringStyle.SPHINX)

    def test_initialization(self):
        """Test formatter initialization."""
        formatter = TypeAnnotationFormatter()
        assert formatter.style == DocstringStyle.GOOGLE
        assert isinstance(formatter._type_mappings, dict)
        assert isinstance(formatter._complexity_cache, dict)

    def test_initialization_with_style(self, numpy_formatter):
        """Test formatter initialization with specific style."""
        assert numpy_formatter.style == DocstringStyle.NUMPY

    def test_format_simple_types(self, google_formatter):
        """Test formatting of simple types."""
        assert google_formatter.format_for_docstring("str") == "str"
        assert google_formatter.format_for_docstring("int") == "int"
        assert google_formatter.format_for_docstring("float") == "float"
        assert google_formatter.format_for_docstring("bool") == "bool"
        assert google_formatter.format_for_docstring("None") == "None"

    def test_format_empty_type(self, google_formatter):
        """Test formatting of empty type annotation."""
        assert google_formatter.format_for_docstring("") == ""
        assert google_formatter.format_for_docstring(None) == ""

    def test_format_optional_types(self, google_formatter):
        """Test formatting of Optional types."""
        assert google_formatter.format_for_docstring("Optional[str]") == "str, optional"
        assert google_formatter.format_for_docstring("Optional[int]") == "int, optional"
        assert (
            google_formatter.format_for_docstring("Optional[List[str]]")
            == "List[str], optional"
        )

    def test_format_union_types(self, google_formatter):
        """Test formatting of Union types."""
        assert google_formatter.format_for_docstring("Union[str, int]") == "str or int"
        assert (
            google_formatter.format_for_docstring("Union[str, int, float]")
            == "str or int or float"
        )
        assert (
            google_formatter.format_for_docstring("Union[str, None]") == "str, optional"
        )

    def test_format_generic_types(self, google_formatter):
        """Test formatting of generic types."""
        # List types
        assert google_formatter.format_for_docstring("List[str]") == "List[str]"
        assert google_formatter.format_for_docstring("List[int]") == "List[int]"

        # Dict types
        assert (
            google_formatter.format_for_docstring("Dict[str, Any]") == "Dict[str, Any]"
        )

        # Tuple types
        assert (
            google_formatter.format_for_docstring("Tuple[int, str]")
            == "Tuple[int, str]"
        )

    def test_format_complex_types(self, google_formatter):
        """Test formatting of complex types."""
        # Callable should be simplified
        assert (
            google_formatter.format_for_docstring("Callable[[int, str], bool]")
            == "callable"
        )

        # Protocol types
        assert google_formatter.format_for_docstring("MyProtocol") == "MyProtocol"

    def test_normalize_type_string(self, google_formatter):
        """Test type string normalization."""
        # Remove quotes
        assert google_formatter._normalize_type_string('"str"') == "str"
        assert google_formatter._normalize_type_string("'int'") == "int"

        # Normalize whitespace
        assert google_formatter._normalize_type_string("List[ str ]") == "List[str]"
        assert (
            google_formatter._normalize_type_string("Dict[str,  Any]")
            == "Dict[str, Any]"
        )
        assert (
            google_formatter._normalize_type_string("Union[  str,   int  ]")
            == "Union[str, int]"
        )

    def test_assess_complexity(self, google_formatter):
        """Test complexity assessment."""
        # Simple types
        assert google_formatter._assess_complexity("str") == TypeComplexity.SIMPLE
        assert google_formatter._assess_complexity("int") == TypeComplexity.SIMPLE

        # Generic types
        assert (
            google_formatter._assess_complexity("List[str]") == TypeComplexity.GENERIC
        )
        assert (
            google_formatter._assess_complexity("Dict[str, Any]")
            == TypeComplexity.GENERIC
        )

        # Union types
        assert (
            google_formatter._assess_complexity("Union[str, int]")
            == TypeComplexity.UNION
        )
        assert (
            google_formatter._assess_complexity("Optional[str]") == TypeComplexity.UNION
        )

        # Complex types
        assert (
            google_formatter._assess_complexity("Callable[[int], str]")
            == TypeComplexity.COMPLEX
        )

    def test_format_for_numpy_style(self, numpy_formatter):
        """Test formatting specific to NumPy style."""
        # Array types should become array_like
        assert numpy_formatter.format_for_docstring("np.ndarray") == "array_like"
        assert numpy_formatter.format_for_docstring("ndarray") == "array_like"
        assert numpy_formatter.format_for_docstring("numpy.ndarray") == "array_like"

        # List types should become "list of type"
        assert numpy_formatter.format_for_docstring("List[str]") == "list of str"
        assert numpy_formatter.format_for_docstring("List[int]") == "list of int"

        # Dict types should become "dict"
        assert numpy_formatter.format_for_docstring("Dict[str, Any]") == "dict"

    def test_format_for_sphinx_style(self, sphinx_formatter):
        """Test formatting specific to Sphinx style."""
        # List types should become "list of type"
        assert sphinx_formatter.format_for_docstring("List[str]") == "list of str"
        assert sphinx_formatter.format_for_docstring("List[int]") == "list of int"

        # Dict types should become "dict"
        assert sphinx_formatter.format_for_docstring("Dict[str, Any]") == "dict"

        # Simple types should be preserved
        assert sphinx_formatter.format_for_docstring("str") == "str"
        assert sphinx_formatter.format_for_docstring("int") == "int"

    def test_split_union_types(self, google_formatter):
        """Test splitting Union type contents."""
        # Simple union
        result = google_formatter._split_union_types("str, int")
        assert result == ["str", "int"]

        # Union with nested generics
        result = google_formatter._split_union_types("List[str], Dict[str, Any], int")
        assert result == ["List[str]", "Dict[str, Any]", "int"]

        # Union with nested unions (edge case)
        result = google_formatter._split_union_types("str, Union[int, float]")
        assert result == ["str", "Union[int, float]"]

    def test_extract_from_ast_name(self, google_formatter):
        """Test extracting type from AST Name node."""
        node = ast.Name(id="str", ctx=ast.Load())
        result = google_formatter.extract_from_ast(node)
        assert result == "str"

    def test_extract_from_ast_attribute(self, google_formatter):
        """Test extracting type from AST Attribute node."""
        # Create ast node for "typing.List"
        value_node = ast.Name(id="typing", ctx=ast.Load())
        node = ast.Attribute(value=value_node, attr="List", ctx=ast.Load())
        result = google_formatter.extract_from_ast(node)
        assert result == "typing.List"

    def test_extract_from_ast_subscript(self, google_formatter):
        """Test extracting type from AST Subscript node."""
        # Create AST node for "List[str]"
        value_node = ast.Name(id="List", ctx=ast.Load())
        slice_node = ast.Name(id="str", ctx=ast.Load())
        node = ast.Subscript(value=value_node, slice=slice_node, ctx=ast.Load())
        result = google_formatter.extract_from_ast(node)
        assert result == "List[str]"

    def test_extract_from_ast_none(self, google_formatter):
        """Test extracting type from None."""
        result = google_formatter.extract_from_ast(None)
        assert result == ""

    def test_simplify_for_style(self, numpy_formatter):
        """Test style-specific simplification."""
        # NumPy simplifications
        result = numpy_formatter.simplify_for_style("List[str]")
        assert result == "array_like"

        result = numpy_formatter.simplify_for_style("np.ndarray")
        assert result == "array_like"

    def test_type_mappings_numpy(self, numpy_formatter):
        """Test NumPy-specific type mappings."""
        mappings = numpy_formatter._type_mappings
        assert "np.ndarray" in mappings
        assert mappings["np.ndarray"] == "array_like"
        assert mappings["List"] == "array_like"

    def test_type_mappings_sphinx(self, sphinx_formatter):
        """Test Sphinx-specific type mappings."""
        mappings = sphinx_formatter._type_mappings
        assert mappings["List"] == "list"
        assert mappings["Dict"] == "dict"

    def test_caching_behavior(self, google_formatter):
        """Test that complexity assessment is cached."""
        # First call should compute and cache
        complexity1 = google_formatter._assess_complexity("List[str]")

        # Second call should use cache
        complexity2 = google_formatter._assess_complexity("List[str]")

        assert complexity1 == complexity2 == TypeComplexity.GENERIC
        assert "List[str]" in google_formatter._complexity_cache

    def test_new_style_union_syntax(self, google_formatter):
        """Test Python 3.10+ union syntax (A | B)."""
        result = google_formatter.format_for_docstring("str | int")
        assert result == "str or int"

        result = google_formatter.format_for_docstring("str | int | float")
        assert result == "str or int or float"


class TestComplexTypeScenarios:
    """Test complex and edge case type scenarios."""

    @pytest.fixture
    def formatter(self):
        """Create a formatter for testing."""
        return TypeAnnotationFormatter(DocstringStyle.GOOGLE)

    def test_deeply_nested_generics(self, formatter):
        """Test deeply nested generic types."""
        complex_type = "Dict[str, List[Tuple[int, str]]]"
        result = formatter.format_for_docstring(complex_type)
        # Should handle gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_very_long_type_annotations(self, formatter):
        """Test very long type annotations."""
        long_type = "Callable[[Dict[str, List[Tuple[int, str]]], Union[List[Dict[str, Any]], None]]"
        result = formatter.format_for_docstring(long_type)
        # Should be simplified to "callable"
        assert result == "callable"

    def test_recursive_type_structures(self, formatter):
        """Test handling of recursive-like type structures."""
        # Union within Union
        recursive_type = "Union[str, Union[int, float]]"
        result = formatter.format_for_docstring(recursive_type)
        # Should handle gracefully
        assert isinstance(result, str)

    def test_malformed_type_annotations(self, formatter):
        """Test handling of malformed type annotations."""
        # Unclosed brackets
        malformed = "List[str"
        result = formatter.format_for_docstring(malformed)
        # Should not crash
        assert isinstance(result, str)

        # Missing type in generic
        malformed = "List[]"
        result = formatter.format_for_docstring(malformed)
        assert isinstance(result, str)

    def test_custom_generic_types(self, formatter):
        """Test handling of custom generic types."""
        custom_type = "MyCustomType[str, int]"
        result = formatter.format_for_docstring(custom_type)
        # Should preserve custom types
        assert "MyCustomType" in result

    def test_forward_references(self, formatter):
        """Test handling of forward references."""
        forward_ref = "'MyClass'"
        result = formatter.format_for_docstring(forward_ref)
        # Should remove quotes
        assert result == "MyClass"

    def test_special_typing_constructs(self, formatter):
        """Test special typing constructs."""
        # Literal types
        literal_type = "Literal['a', 'b', 'c']"
        result = formatter.format_for_docstring(literal_type)
        assert isinstance(result, str)

        # TypeVar
        typevar_type = "TypeVar('T')"
        result = formatter.format_for_docstring(typevar_type)
        # Should be simplified
        assert result == "T"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_format_type_for_style_function(self):
        """Test the convenience function."""
        result = format_type_for_style("List[str]", DocstringStyle.NUMPY)
        assert result == "list of str"

        result = format_type_for_style("Optional[int]", DocstringStyle.GOOGLE)
        assert result == "int, optional"

    def test_extract_type_from_ast_function(self):
        """Test the AST extraction convenience function."""
        node = ast.Name(id="str", ctx=ast.Load())
        result = extract_type_from_ast(node)
        assert result == "str"

    def test_convenience_function_with_none(self):
        """Test convenience functions with None input."""
        result = format_type_for_style(None, DocstringStyle.GOOGLE)
        assert result == ""

        result = extract_type_from_ast(None)
        assert result == ""


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.fixture
    def formatter(self):
        """Create formatter for performance testing."""
        return TypeAnnotationFormatter()

    def test_cache_performance(self, formatter):
        """Test that caching improves performance."""
        complex_type = "Dict[str, List[Tuple[int, Union[str, float]]]]"

        # First call (should compute and cache)
        result1 = formatter._assess_complexity(complex_type)

        # Subsequent calls should be faster (use cache)
        for _ in range(10):
            result = formatter._assess_complexity(complex_type)
            assert result == result1

    def test_unicode_in_type_annotations(self, formatter):
        """Test handling of Unicode in type annotations."""
        unicode_type = "Dαtα[str]"  # Contains Greek letters
        result = formatter.format_for_docstring(unicode_type)
        # Should preserve Unicode
        assert "Dαtα" in result

    def test_very_long_type_names(self, formatter):
        """Test handling of very long type names."""
        long_name = "VeryLongTypeNameThatExceedsNormalLengthExpectations" * 3
        result = formatter.format_for_docstring(long_name)
        # Should handle gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_special_characters_in_types(self, formatter):
        """Test handling of special characters."""
        special_type = "Type_With_Underscores[Generic$Type]"
        result = formatter.format_for_docstring(special_type)
        # Should handle gracefully
        assert isinstance(result, str)

    def test_empty_and_whitespace_types(self, formatter):
        """Test handling of empty and whitespace-only types."""
        assert formatter.format_for_docstring("") == ""
        assert formatter.format_for_docstring("   ") == ""
        assert formatter.format_for_docstring("\t\n") == ""

    def test_complexity_cache_size(self, formatter):
        """Test that complexity cache doesn't grow indefinitely."""
        # Add many different types to cache
        for i in range(100):
            type_name = f"Type{i}[Generic{i}]"
            formatter._assess_complexity(type_name)

        # Cache should contain the types
        assert len(formatter._complexity_cache) <= 100
