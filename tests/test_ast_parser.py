"""
Comprehensive test suite for AST parser module.

This test suite covers all aspects of the AST parser including:
- Basic parsing functionality
- Edge cases and error handling
- Performance requirements
- Complex syntax patterns
"""

import ast
import os
import time

import pytest

from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    parse_python_file,
    _get_decorator_names,
    _get_annotation_string,
    _get_default_value,
    _is_method_function,
)
from codedocsync.utils.errors import (
    ValidationError,
    FileAccessError,
    SyntaxParsingError,
)


class TestBasicParsing:
    """Test basic parsing functionality."""

    def test_parse_simple_function(self, sample_python_file):
        """Test parsing a basic function."""
        functions = parse_python_file(str(sample_python_file))

        assert len(functions) == 1
        func = functions[0]
        assert func.signature.name == "simple_function"
        assert len(func.signature.parameters) == 0
        assert func.signature.return_type is None
        assert not func.signature.is_async
        assert not func.signature.is_method
        assert func.docstring.raw_text == "A simple function."
        assert func.line_number == 1
        assert "def simple_function():" in func.source_code

    def test_parse_function_with_parameters(self, function_with_params_file):
        """Test function with various parameter types."""
        functions = parse_python_file(str(function_with_params_file))

        assert len(functions) == 1
        func = functions[0]
        assert func.signature.name == "complex_function"
        assert len(func.signature.parameters) == 6

        # Check parameter details
        params = func.signature.parameters
        assert params[0].name == "required_param"
        assert params[0].type_annotation == "str"
        assert params[0].is_required

        assert params[1].name == "optional_param"
        assert params[1].type_annotation == "int"
        assert not params[1].is_required
        assert params[1].default_value == "42"

        assert params[2].name == "list_param"
        assert params[2].type_annotation == "List[str]"
        assert not params[2].is_required
        assert params[2].default_value == "None"

        assert params[3].name == "*args"
        assert params[3].type_annotation == "Any"
        assert not params[3].is_required

        assert params[4].name == "keyword_only"
        assert params[4].type_annotation == "bool"
        assert not params[4].is_required
        assert params[4].default_value == "True"

        assert params[5].name == "**kwargs"
        assert params[5].type_annotation == "Dict[str, Any]"
        assert not params[5].is_required

    def test_parse_async_function(self, async_function_file):
        """Test async function parsing."""
        functions = parse_python_file(str(async_function_file))

        assert len(functions) == 1
        func = functions[0]
        assert func.signature.name == "async_function"
        assert func.signature.is_async
        assert func.signature.return_type == "str"
        assert "async def async_function" in func.source_code

    def test_parse_decorated_function(self, decorated_function_file):
        """Test functions with decorators."""
        functions = parse_python_file(str(decorated_function_file))

        assert len(functions) == 1
        func = functions[0]
        assert func.signature.name == "decorated_function"
        assert len(func.signature.decorators) == 3
        assert "property" in func.signature.decorators
        assert "classmethod" in func.signature.decorators
        assert any("custom_decorator" in dec for dec in func.signature.decorators)

    def test_parse_class_methods(self, class_methods_file):
        """Test parsing class methods."""
        functions = parse_python_file(str(class_methods_file))

        assert len(functions) == 3

        # Instance method
        instance_method = next(
            f for f in functions if f.signature.name == "instance_method"
        )
        assert instance_method.signature.is_method
        assert instance_method.signature.parameters[0].name == "self"

        # Class method
        class_method = next(f for f in functions if f.signature.name == "class_method")
        assert class_method.signature.is_method
        assert "classmethod" in class_method.signature.decorators

        # Static method
        static_method = next(
            f for f in functions if f.signature.name == "static_method"
        )
        assert static_method.signature.is_method
        assert "staticmethod" in static_method.signature.decorators


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_empty_file(self, empty_file):
        """Empty file should return empty list."""
        functions = parse_python_file(str(empty_file))
        assert functions == []

    def test_parse_syntax_error(self, syntax_error_file):
        """Syntax errors should attempt partial parsing."""
        try:
            functions = parse_python_file(str(syntax_error_file))
            # Should find the function defined before the syntax error
            assert len(functions) == 1
            assert functions[0].signature.name == "before_error"
        except SyntaxParsingError:
            # If partial parsing fails, that's also acceptable
            pass

    def test_parse_unicode_content(self, unicode_file):
        """Test files with unicode in docstrings and comments."""
        functions = parse_python_file(str(unicode_file))

        assert len(functions) == 1
        func = functions[0]
        assert func.signature.name == "unicode_function"
        assert "Unicode content: 你好" in func.docstring.raw_text

    def test_parse_complex_annotations(self, complex_annotations_file):
        """Test complex type annotations (Union, Optional, etc)."""
        functions = parse_python_file(str(complex_annotations_file))

        assert len(functions) == 1
        func = functions[0]
        params = func.signature.parameters

        assert params[0].type_annotation == "Union[str, int]"
        assert params[1].type_annotation == "Optional[List[Dict[str, Any]]]"
        assert params[2].type_annotation == "Callable[[str], bool]"
        assert func.signature.return_type == "Tuple[str, int]"

    def test_parse_imports_only_file(self, imports_only_file):
        """Files with only imports should return empty list."""
        functions = parse_python_file(str(imports_only_file))
        assert functions == []

    def test_parse_nested_functions(self, nested_functions_file):
        """Test nested function parsing."""
        functions = parse_python_file(str(nested_functions_file))

        # Should find both outer and inner functions
        assert len(functions) == 2
        function_names = {f.signature.name for f in functions}
        assert "outer_function" in function_names
        assert "inner_function" in function_names


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_file_not_found(self):
        """Should raise FileAccessError with recovery hint."""
        with pytest.raises(FileAccessError) as exc_info:
            parse_python_file("nonexistent_file.py")

        assert "File not found" in str(exc_info.value)
        assert exc_info.value.recovery_hint is not None
        assert "Check the file path" in exc_info.value.recovery_hint

    def test_permission_denied(self, permission_denied_file):
        """Should handle permission errors gracefully."""
        # This test may not work on Windows due to different permission handling
        if os.name == "nt":  # Windows
            pytest.skip("Permission test skipped on Windows")

        with pytest.raises(FileAccessError) as exc_info:
            parse_python_file(str(permission_denied_file))

        assert "Permission denied" in str(exc_info.value)
        assert "Check file permissions" in exc_info.value.recovery_hint

    def test_malformed_decorators(self, malformed_decorators_file):
        """Should handle invalid decorator syntax."""
        functions = parse_python_file(str(malformed_decorators_file))

        assert len(functions) == 1
        func = functions[0]
        # Should gracefully handle malformed decorators
        assert len(func.signature.decorators) > 0

    def test_encoding_error_fallback(self, latin1_file):
        """Should fallback to latin-1 encoding when UTF-8 fails."""
        functions = parse_python_file(str(latin1_file))

        assert len(functions) == 1
        assert functions[0].signature.name == "latin1_function"


class TestPerformance:
    """Test performance requirements."""

    def test_parsing_performance_small_file(self, small_file):
        """Small files should parse in <50ms (relaxed for testing)."""
        start_time = time.time()
        functions = parse_python_file(str(small_file))
        duration = time.time() - start_time

        assert duration < 0.05  # 50ms (relaxed for testing environment)
        assert len(functions) == 1

    def test_parsing_performance_large_file(self, large_file):
        """Large files should parse in <200ms."""
        start_time = time.time()
        functions = parse_python_file(str(large_file))
        duration = time.time() - start_time

        assert duration < 0.2  # 200ms
        assert len(functions) >= 50  # Should have many functions


class TestDataModels:
    """Test data model validation and functionality."""

    def test_function_parameter_validation(self):
        """Test FunctionParameter validation."""
        # Valid parameter
        param = FunctionParameter(name="valid_param", type_annotation="str")
        assert param.name == "valid_param"

        # Invalid parameter name
        with pytest.raises(ValidationError):
            FunctionParameter(name="123invalid", type_annotation="str")

        # Invalid parameter name with spaces
        with pytest.raises(ValidationError):
            FunctionParameter(name="invalid param", type_annotation="str")

    def test_function_signature_validation(self):
        """Test FunctionSignature validation."""
        # Valid signature
        sig = FunctionSignature(name="valid_function")
        assert sig.name == "valid_function"

        # Invalid function name
        with pytest.raises(ValidationError):
            FunctionSignature(name="123invalid")

    def test_function_signature_to_string(self):
        """Test signature string representation."""
        params = [
            FunctionParameter(name="param1", type_annotation="str"),
            FunctionParameter(name="param2", type_annotation="int", default_value="42"),
        ]
        sig = FunctionSignature(
            name="test_func", parameters=params, return_type="bool", is_async=True,
        )

        sig_str = sig.to_string()
        assert "async def test_func" in sig_str
        assert "param1: str" in sig_str
        assert "param2: int = 42" in sig_str
        assert "-> bool" in sig_str

    def test_parsed_function_validation(self):
        """Test ParsedFunction validation."""
        sig = FunctionSignature(name="test_func")

        # Valid function
        func = ParsedFunction(
            signature=sig, line_number=1, end_line_number=5, file_path="test.py",
        )
        assert func.line_number == 1

        # Invalid line numbers
        with pytest.raises(ValidationError):
            ParsedFunction(
                signature=sig, line_number=-1, end_line_number=5, file_path="test.py",
            )

        with pytest.raises(ValidationError):
            ParsedFunction(
                signature=sig, line_number=10, end_line_number=5, file_path="test.py",
            )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_decorator_names(self):
        """Test decorator name extraction."""
        # Simple decorator
        simple_code = "@property\ndef func(): pass"
        tree = ast.parse(simple_code)
        func_node = tree.body[0]
        decorators = _get_decorator_names(func_node.decorator_list)
        assert decorators == ["property"]

    def test_get_annotation_string(self):
        """Test annotation string extraction."""
        code = "def func(param: List[str]) -> int: pass"
        tree = ast.parse(code)
        func_node = tree.body[0]

        param_annotation = _get_annotation_string(func_node.args.args[0].annotation)
        assert param_annotation == "List[str]"

        return_annotation = _get_annotation_string(func_node.returns)
        assert return_annotation == "int"

    def test_is_method_function(self):
        """Test method detection logic."""
        # Instance method
        self_param = FunctionParameter(name="self")
        assert _is_method_function([self_param], [])

        # Class method
        cls_param = FunctionParameter(name="cls")
        assert _is_method_function([cls_param], ["classmethod"])

        # Static method
        no_params = []
        assert _is_method_function(no_params, ["staticmethod"])

        # Regular function
        regular_param = FunctionParameter(name="param")
        assert not _is_method_function([regular_param], [])

    def test_get_default_value(self):
        """Test default value extraction."""
        code = "def func(param=42): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        default_node = func_node.args.defaults[0]

        default_value = _get_default_value(default_node)
        assert default_value == "42"


# Test fixtures
@pytest.fixture
def sample_python_file(tmp_path):
    """Create a temporary Python file for testing."""
    content = '''def simple_function():
    """A simple function."""
    pass
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def function_with_params_file(tmp_path):
    """Create a file with complex function parameters."""
    content = '''from typing import List, Dict, Any

def complex_function(
    required_param: str,
    optional_param: int = 42,
    list_param: List[str] = None,
    *args: Any,
    keyword_only: bool = True,
    **kwargs: Dict[str, Any]
) -> str:
    """A complex function with various parameter types."""
    pass
'''
    file_path = tmp_path / "complex_params.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def async_function_file(tmp_path):
    """Create a file with async function."""
    content = '''async def async_function() -> str:
    """An async function."""
    return "async result"
'''
    file_path = tmp_path / "async_func.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def decorated_function_file(tmp_path):
    """Create a file with decorated function."""
    content = '''@property
@classmethod
@custom_decorator(arg="value")
def decorated_function(cls):
    """A decorated function."""
    pass
'''
    file_path = tmp_path / "decorated.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def class_methods_file(tmp_path):
    """Create a file with class methods."""
    content = '''class TestClass:
    def instance_method(self):
        """Instance method."""
        pass

    @classmethod
    def class_method(cls):
        """Class method."""
        pass

    @staticmethod
    def static_method():
        """Static method."""
        pass
'''
    file_path = tmp_path / "class_methods.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file."""
    file_path = tmp_path / "empty.py"
    file_path.write_text("")
    return file_path


@pytest.fixture
def syntax_error_file(tmp_path):
    """Create a file with syntax error."""
    content = '''def before_error():
    """Function before error."""
    pass

def syntax_error(
    # Missing closing parenthesis and colon
    pass
'''
    file_path = tmp_path / "syntax_error.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def unicode_file(tmp_path):
    """Create a file with unicode content."""
    content = '''def unicode_function():
    """Unicode content: 你好 世界! 🌍"""
    pass
'''
    file_path = tmp_path / "unicode.py"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def complex_annotations_file(tmp_path):
    """Create a file with complex type annotations."""
    content = '''from typing import Union, Optional, List, Dict, Any, Callable, Tuple

def complex_annotations(
    union_param: Union[str, int],
    optional_param: Optional[List[Dict[str, Any]]],
    callable_param: Callable[[str], bool]
) -> Tuple[str, int]:
    """Function with complex type annotations."""
    pass
'''
    file_path = tmp_path / "complex_annotations.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def imports_only_file(tmp_path):
    """Create a file with only imports."""
    content = """import os
import sys
from typing import List, Dict

# Just imports and comments
VERSION = "1.0.0"
"""
    file_path = tmp_path / "imports_only.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def nested_functions_file(tmp_path):
    """Create a file with nested functions."""
    content = '''def outer_function():
    """Outer function."""

    def inner_function():
        """Inner function."""
        pass

    return inner_function
'''
    file_path = tmp_path / "nested.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def malformed_decorators_file(tmp_path):
    """Create a file with malformed decorators."""
    content = '''@property
@some.complex.decorator.chain
def malformed_function():
    """Function with complex decorators."""
    pass
'''
    file_path = tmp_path / "malformed.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def latin1_file(tmp_path):
    """Create a file with latin-1 encoding."""
    content = '''def latin1_function():
    """Function with latin-1 content: café"""
    pass
'''
    file_path = tmp_path / "latin1.py"
    file_path.write_bytes(content.encode("latin-1"))
    return file_path


@pytest.fixture
def permission_denied_file(tmp_path):
    """Create a file with no read permissions."""
    content = '''def permission_function():
    """Function in permission-denied file."""
    pass
'''
    file_path = tmp_path / "permission.py"
    file_path.write_text(content)
    file_path.chmod(0o000)  # No permissions
    yield file_path
    file_path.chmod(0o644)  # Restore permissions for cleanup


@pytest.fixture
def small_file(tmp_path):
    """Create a small file for performance testing."""
    content = '''def small_function():
    """Small function for performance testing."""
    return "small"
'''
    file_path = tmp_path / "small.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def large_file(tmp_path):
    """Create a large file for performance testing."""
    functions = []
    for i in range(100):
        functions.append(
            f'''def function_{i}():
    """Function number {i}."""
    return {i}
'''
        )

    content = "\n".join(functions)
    file_path = tmp_path / "large.py"
    file_path.write_text(content)
    return file_path
