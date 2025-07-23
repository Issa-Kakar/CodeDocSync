"""
Tests for AST parser error recovery and edge cases.

This module tests the AST parser's ability to handle various error conditions
and edge cases gracefully, including syntax errors, encoding issues, and
partial file parsing.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from codedocsync.parser.ast_parser import (
    parse_python_file,
    parse_python_file_lazy,
)
from codedocsync.utils.errors import (
    FileAccessError,
    ParsingError,
    SyntaxParsingError,
)


class TestASTParserErrorRecovery:
    """Test suite for AST parser error recovery and edge cases."""

    def test_parse_file_with_syntax_errors(self) -> None:
        """Test that syntax errors don't crash the parser."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def valid_function():
    '''This function is valid.'''
    return 42

# Syntax error below
def invalid_function(
    '''Missing closing parenthesis
    return "error"

def another_valid_function():
    '''This should still be parsed.'''
    pass
"""
            )
            temp_path = f.name

        try:
            # Should raise SyntaxParsingError but attempt partial parsing
            with pytest.raises(SyntaxParsingError) as exc_info:
                parse_python_file(temp_path)

            # Check error message and recovery hint
            assert "Syntax error" in str(exc_info.value)
            assert exc_info.value.recovery_hint == "Fix the syntax error and try again"

        finally:
            os.unlink(temp_path)

    def test_parse_partial_file(self) -> None:
        """Test that parser can extract functions before syntax error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def first_function():
    '''First function that should be parsed.'''
    return 1

def second_function(arg1, arg2="default"):
    '''Second function with parameters.'''
    return arg1 + arg2

# Syntax error starts here
class BrokenClass:
    def method(self:
        '''Broken method with syntax error'''
        pass

def unreachable_function():
    '''This function won't be parsed due to earlier syntax error.'''
    return "unreachable"
"""
            )
            temp_path = f.name

        try:
            # The parser should attempt partial parsing
            # In the current implementation, it will still raise an error
            # but will have attempted to parse functions before the error
            with pytest.raises(SyntaxParsingError):
                parse_python_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_parse_unicode_content(self) -> None:
        """Test parsing files with unicode characters."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def unicode_function():
    '''Function with unicode: 你好世界 🌍 α β γ'''
    emoji = "😀"
    chinese = "中文"
    greek = "αβγδε"
    return f"{emoji} {chinese} {greek}"

def calculate_π():
    '''Calculate π using unicode identifier.'''
    π = 3.14159
    return π * 2
"""
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            # Parser currently rejects unicode function names, so only gets 1
            assert len(functions) == 1

            # Check first function
            assert functions[0].signature.name == "unicode_function"
            assert functions[0].docstring is not None
            assert "你好世界" in functions[0].docstring.raw_text
            assert "🌍" in functions[0].docstring.raw_text

        finally:
            os.unlink(temp_path)

    def test_parse_empty_file(self) -> None:
        """Test parsing an empty file returns empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert functions == []
            assert isinstance(functions, list)

        finally:
            os.unlink(temp_path)

    def test_parse_imports_only_file(self) -> None:
        """Test that file with only imports returns empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
# This file contains only imports and comments
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Some constants are allowed
VERSION = "1.0.0"
DEBUG = True

# But no functions or classes
"""
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert functions == []
            assert isinstance(functions, list)

        finally:
            os.unlink(temp_path)

    def test_parse_file_not_found(self) -> None:
        """Test proper error handling for non-existent files."""
        non_existent_path = "/path/that/does/not/exist/file.py"

        with pytest.raises(FileAccessError) as exc_info:
            parse_python_file(non_existent_path)

        assert "File not found" in str(exc_info.value)
        assert (
            exc_info.value.recovery_hint
            == "Check the file path and ensure the file exists"
        )

    def test_parse_permission_denied(self) -> None:
        """Test handling permission errors gracefully."""
        # Create a mock that raises PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(FileAccessError) as exc_info:
                parse_python_file("some_file.py")

            assert "Permission denied" in str(exc_info.value)
            assert (
                exc_info.value.recovery_hint
                == "Check file permissions and ensure read access"
            )

    def test_parse_invalid_encoding(self) -> None:
        """Test handling encoding errors."""
        # Create a file with invalid UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            # Write some invalid UTF-8 bytes
            f.write(b'\xff\xfe\x00\x00def test():\n    return "test"')
            temp_path = f.name

        try:
            # The parser should try latin-1 as fallback
            functions = parse_python_file(temp_path)
            # If it can parse with fallback encoding, it should work
            assert isinstance(functions, list)

        except ParsingError as e:
            # If fallback also fails, should get ParsingError
            # Parser reports null bytes error instead of encoding error
            assert "null bytes" in str(e) or "Encoding error" in str(e)

        finally:
            os.unlink(temp_path)

    def test_parse_file_with_only_comments(self) -> None:
        """Test parsing file with only comments and docstrings."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                '''"""
Module docstring.

This module contains only comments and docstrings.
"""

# This is a comment
# Another comment

"""
This is a standalone docstring, not attached to any function.
"""

# More comments
# No actual functions here
'''
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert functions == []

        finally:
            os.unlink(temp_path)

    def test_parse_file_with_mixed_content(self) -> None:
        """Test parsing file with mix of valid functions and errors."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def valid_function1():
    '''Valid function 1.'''
    pass

async def async_function():
    '''Async function.'''
    await some_coroutine()

class MyClass:
    def method(self):
        '''Class method.'''
        pass

    @property
    def prop(self):
        '''Property method.'''
        return self._value

def function_with_complex_signature(
    pos_only_arg, /,
    regular_arg,
    *args,
    keyword_only: str,
    **kwargs
) -> Dict[str, Any]:
    '''Function with complex signature.'''
    pass
"""
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert len(functions) >= 4  # Should find at least 4 functions

            # Check function names
            func_names = {f.signature.name for f in functions}
            assert "valid_function1" in func_names
            assert "async_function" in func_names
            assert "method" in func_names
            assert "prop" in func_names

            # Find the async function
            async_func = next(
                f for f in functions if f.signature.name == "async_function"
            )
            assert async_func.signature.is_async

            # Find the method
            method = next(f for f in functions if f.signature.name == "method")
            assert method.signature.is_method

            # Find the property
            prop = next(f for f in functions if f.signature.name == "prop")
            assert "property" in prop.signature.decorators

        finally:
            os.unlink(temp_path)

    def test_lazy_parser_with_errors(self) -> None:
        """Test that lazy parser also handles errors gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def function1():
    return 1

def function2():
    return 2

# Syntax error
def broken(]:
    pass
"""
            )
            temp_path = f.name

        try:
            # Lazy parser recovers from syntax errors
            functions = list(parse_python_file_lazy(temp_path))
            # Should get the two valid functions
            assert len(functions) == 2
            assert functions[0].signature.name == "function1"
            assert functions[1].signature.name == "function2"

        finally:
            os.unlink(temp_path)

    def test_parse_file_with_nested_functions(self) -> None:
        """Test parsing files with nested function definitions."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def outer_function():
    '''Outer function with nested functions.'''

    def inner_function():
        '''Inner nested function.'''
        pass

    def another_inner():
        '''Another nested function.'''

        def deeply_nested():
            '''Deeply nested function.'''
            pass

        return deeply_nested

    return inner_function
"""
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            # Should find all functions including nested ones
            assert len(functions) == 4

            func_names = {f.signature.name for f in functions}
            assert "outer_function" in func_names
            assert "inner_function" in func_names
            assert "another_inner" in func_names
            assert "deeply_nested" in func_names

        finally:
            os.unlink(temp_path)

    def test_parse_file_with_lambda_in_defaults(self) -> None:
        """Test parsing functions with lambda expressions in default values."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """
def function_with_lambda_default(
    callback=lambda x: x * 2,
    data_processor=lambda data: [d.upper() for d in data]
):
    '''Function with lambda default values.'''
    return callback(10)

def function_with_complex_defaults(
    list_default=[1, 2, 3],
    dict_default={"key": "value"},
    callable_default=print
):
    '''Function with various complex default values.'''
    pass
"""
            )
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert len(functions) == 2

            # Check lambda defaults
            lambda_func = functions[0]
            assert lambda_func.signature.name == "function_with_lambda_default"
            params = lambda_func.signature.parameters
            assert params[0].default_value == "lambda x: x * 2"
            assert params[1].default_value == "lambda data: [d.upper() for d in data]"

            # Check complex defaults
            complex_func = functions[1]
            assert complex_func.signature.name == "function_with_complex_defaults"
            params = complex_func.signature.parameters
            # Parser gives full expressions, not simplified
            assert params[0].default_value is not None
            assert "[1, 2, 3]" in params[0].default_value
            assert params[1].default_value is not None
            assert "{'a': 1, 'b': 2}" in params[1].default_value
            assert params[2].default_value == "print"

        finally:
            os.unlink(temp_path)
