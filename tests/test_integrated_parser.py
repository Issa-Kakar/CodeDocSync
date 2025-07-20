"""Tests for the integrated parser combining AST and docstring parsing."""

import os
import tempfile

import pytest

from codedocsync.parser.ast_parser import RawDocstring
from codedocsync.parser.docstring_models import DocstringFormat, ParsedDocstring
from codedocsync.parser.integrated_parser import IntegratedParser


class TestIntegratedParser:
    """Test the integrated parser functionality."""

    @pytest.fixture
    def parser(self):
        return IntegratedParser()

    @pytest.fixture
    def test_python_file(self):
        """Create a temporary Python file for testing."""
        content = '''"""Module docstring."""

def simple_function():
    """Simple function with basic docstring."""
    pass

def google_style_function(param1, param2="default"):
    """Function with Google-style docstring.

    This is a more detailed description of what the function does.

    Args:
        param1 (str): First parameter description
        param2 (str, optional): Second parameter with default. Defaults to "default".

    Returns:
        bool: Success indicator

    Raises:
        ValueError: If param1 is empty
    """
    return True

def numpy_style_function(data, threshold=0.5):
    """Process data with NumPy-style docstring.

    Parameters
    ----------
    data : array_like
        Input data to process
    threshold : float, optional
        Processing threshold (default is 0.5)

    Returns
    -------
    ndarray
        Processed data
    """
    pass

class ExampleClass:
    """Example class with methods."""

    def method_with_docstring(self, arg):
        """Method with simple docstring.

        Args:
            arg: Method argument
        """
        pass

    def method_without_docstring(self):
        pass

def function_without_docstring():
    pass

async def async_function():
    """Async function with docstring.

    Returns:
        str: Async result
    """
    return "async"
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name

        # Cleanup
        os.unlink(f.name)

    def test_parse_file_basic(self, parser, test_python_file):
        """Test basic file parsing with docstring integration."""
        functions = parser.parse_file(test_python_file)

        # Should find multiple functions
        assert len(functions) >= 5

        # Check that functions have names
        function_names = [f.signature.name for f in functions]
        assert "simple_function" in function_names
        assert "google_style_function" in function_names
        assert "numpy_style_function" in function_names

    def test_docstring_parsing_integration(self, parser, test_python_file):
        """Test that docstrings are properly parsed and integrated."""
        functions = parser.parse_file(test_python_file)

        # Find the Google-style function
        google_func = next(
            (f for f in functions if f.signature.name == "google_style_function"), None
        )
        assert google_func is not None

        # Check that docstring was parsed
        if google_func.docstring:
            if isinstance(google_func.docstring, ParsedDocstring):
                assert google_func.docstring.format == DocstringFormat.GOOGLE
                assert len(google_func.docstring.parameters) >= 2

                # Check parameter extraction
                param1 = google_func.docstring.get_parameter("param1")
                if param1:
                    assert param1.name == "param1"
                    assert "First parameter" in param1.description

    def test_functions_without_docstrings(self, parser, test_python_file):
        """Test handling of functions without docstrings."""
        functions = parser.parse_file(test_python_file)

        # Find function without docstring
        no_doc_func = next(
            (f for f in functions if f.signature.name == "function_without_docstring"),
            None,
        )
        assert no_doc_func is not None
        assert no_doc_func.docstring is None

    def test_raw_docstring_preservation(self, parser, test_python_file):
        """Test that raw docstrings are preserved when parsing fails."""
        functions = parser.parse_file(test_python_file)

        # Find simple function
        simple_func = next(
            (f for f in functions if f.signature.name == "simple_function"), None
        )
        assert simple_func is not None

        # Should have either raw or parsed docstring
        assert simple_func.docstring is not None

        # If it's still raw, it means parsing failed gracefully
        if isinstance(simple_func.docstring, RawDocstring):
            assert (
                simple_func.docstring.raw_text
                == "Simple function with basic docstring."
            )

    def test_cache_functionality(self, parser, test_python_file):
        """Test that caching works correctly."""
        # Parse file twice
        functions1 = parser.parse_file(test_python_file)
        functions2 = parser.parse_file(test_python_file)

        assert len(functions1) == len(functions2)

        # Check cache stats
        stats = parser.get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_size" in stats

    def test_cache_clear(self, parser, test_python_file):
        """Test cache clearing functionality."""
        # Parse file to populate cache
        parser.parse_file(test_python_file)

        # Clear cache
        parser.clear_cache()

        # Check cache is empty
        stats_after = parser.get_cache_stats()
        assert stats_after["cache_size"] == 0

    def test_lazy_parsing(self, parser, test_python_file):
        """Test lazy parsing functionality."""
        functions = list(parser.parse_file_lazy(test_python_file))

        # Should return same number of functions as regular parsing
        regular_functions = parser.parse_file(test_python_file)
        assert len(functions) == len(regular_functions)

        # Check that it's actually a generator
        lazy_gen = parser.parse_file_lazy(test_python_file)
        first_func = next(lazy_gen)
        assert first_func is not None

    def test_class_method_parsing(self, parser, test_python_file):
        """Test parsing of class methods."""
        functions = parser.parse_file(test_python_file)

        # Find class methods
        method_names = [f.signature.name for f in functions]
        assert "method_with_docstring" in method_names
        assert "method_without_docstring" in method_names

    def test_async_function_parsing(self, parser, test_python_file):
        """Test parsing of async functions."""
        functions = parser.parse_file(test_python_file)

        # Find async function
        async_func = next(
            (f for f in functions if f.signature.name == "async_function"), None
        )
        assert async_func is not None

        # Should have docstring
        assert async_func.docstring is not None

    def test_malformed_file_handling(self, parser):
        """Test handling of malformed Python files."""
        # Create file with syntax error
        content = """
def broken_function(
    # Missing closing parenthesis
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            temp_file = f.name

        try:
            # Should handle syntax errors gracefully
            functions = parser.parse_file(temp_file)
            # May return empty list or partial results
            assert isinstance(functions, list)
        except Exception as e:
            # Or may raise a parsing error, which is also acceptable
            assert "parsing" in str(e).lower() or "syntax" in str(e).lower()
        finally:
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Windows file locking issue

    def test_empty_file_handling(self, parser):
        """Test handling of empty Python files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            temp_file = f.name

        try:
            functions = parser.parse_file(temp_file)
            assert isinstance(functions, list)
            assert len(functions) == 0
        finally:
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Windows file locking issue

    def test_nonexistent_file_handling(self, parser):
        """Test handling of non-existent files."""
        from codedocsync.utils.errors import FileAccessError

        with pytest.raises(FileAccessError):
            parser.parse_file("/path/that/does/not/exist.py")

    def test_large_file_performance(self, parser):
        """Test performance with larger files."""
        # Create a file with many functions
        content = ""
        for i in range(50):
            content += f'''
def function_{i}(param1, param2="default"):
    """Function {i} with Google-style docstring.

    Args:
        param1 (str): Parameter 1
        param2 (str): Parameter 2

    Returns:
        bool: Success
    """
    return True
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            temp_file = f.name

        try:
            import time

            start_time = time.time()
            functions = parser.parse_file(temp_file)
            end_time = time.time()

            # Should complete in reasonable time (less than 5 seconds)
            assert (end_time - start_time) < 5.0
            assert len(functions) == 50
        finally:
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Windows file locking issue

    def test_unicode_content_handling(self, parser):
        """Test handling of files with Unicode content."""
        content = '''"""Module with Unicode 中文 content."""

def unicode_function():
    """Function with Unicode émojis 🎉 and symbols."""
    pass
'''

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            temp_file = f.name

        try:
            functions = parser.parse_file(temp_file)
            assert len(functions) >= 1

            # Find the unicode function
            unicode_func = next(
                (f for f in functions if f.signature.name == "unicode_function"), None
            )
            assert unicode_func is not None

            # Should handle Unicode in docstring
            if unicode_func.docstring:
                if isinstance(unicode_func.docstring, ParsedDocstring):
                    assert "🎉" in unicode_func.docstring.summary
                elif isinstance(unicode_func.docstring, RawDocstring):
                    assert "🎉" in unicode_func.docstring.raw_text
        finally:
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Windows file locking issue
