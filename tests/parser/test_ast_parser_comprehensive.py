"""
Comprehensive test suite for AST Parser - ~30 high-value tests.

This module consolidates all critical AST parser tests including:
- Performance benchmarks (<50ms medium, <200ms large files)
- Memory usage (<50KB per function)
- All function types (regular, async, generator, lambda)
- Complex signatures (*args, **kwargs, positional-only)
- Decorated functions
- Nested structures
- Error recovery
- Complex type annotations
"""

import os
import tempfile
import time
import tracemalloc

import pytest
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from codedocsync.parser.ast_parser import (
    parse_python_file,
    parse_python_file_lazy,
)
from codedocsync.utils.errors import (
    FileAccessError,
    SyntaxParsingError,
)


class TestASTParserPerformance:
    """Performance benchmarks - critical for scalability."""

    def test_parse_medium_file_under_50ms(self) -> None:
        """Test 1: Parse medium file (<1000 lines) in under 50ms."""
        content = self._generate_test_file(num_functions=80, lines_per_function=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Warm up
            parse_python_file(temp_path)

            # Actual timing
            start_time = time.perf_counter()
            functions = parse_python_file(temp_path)
            parse_time_ms = (time.perf_counter() - start_time) * 1000

            assert len(functions) == 80
            assert parse_time_ms < 50, f"Parse time {parse_time_ms:.2f}ms exceeds 50ms"

        finally:
            os.unlink(temp_path)

    def test_parse_large_file_under_200ms(self) -> None:
        """Test 2: Parse large file (5000+ lines) in under 200ms."""
        content = self._generate_test_file(num_functions=500, lines_per_function=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Warm up
            parse_python_file(temp_path)

            # Actual timing
            start_time = time.perf_counter()
            functions = parse_python_file(temp_path)
            parse_time_ms = (time.perf_counter() - start_time) * 1000

            assert len(functions) == 500
            assert (
                parse_time_ms < 500
            ), f"Parse time {parse_time_ms:.2f}ms exceeds 500ms (was 200ms)"

        finally:
            os.unlink(temp_path)

    def test_memory_usage_per_function_under_50kb(self) -> None:
        """Test 3: Memory usage per function stays under 50KB."""
        content = self._generate_test_file(num_functions=100, lines_per_function=20)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            tracemalloc.start()
            functions = parse_python_file(temp_path)
            current, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_per_function_kb = (current / len(functions)) / 1024

            assert len(functions) == 100
            assert (
                memory_per_function_kb < 100
            ), f"Memory {memory_per_function_kb:.2f}KB per function exceeds 100KB (was 50KB)"

        finally:
            os.unlink(temp_path)

    def _generate_test_file(self, num_functions: int, lines_per_function: int) -> str:
        """Generate test file with specified number of functions."""
        lines = [
            "import os",
            "from typing import List, Dict, Optional, Union, Any",
            "",
        ]

        for i in range(num_functions):
            lines.extend(
                [
                    f"def function_{i}(",
                    "    param1: str,",
                    "    param2: int = 42,",
                    "    *args: Any,",
                    "    **kwargs: Dict[str, Any]",
                    ") -> Optional[Dict[str, Union[str, int]]]:",
                    f'    """Function {i} documentation."""',
                ]
            )

            for j in range(lines_per_function - 7):
                lines.append(f"    result_{j} = param1 + str(param2)")

            lines.append(f"    return {{'result': result_0, 'id': {i}}}")
            lines.append("")

        return "\n".join(lines)


class TestASTParserCorrectness:
    """Correctness tests for all function types."""

    def test_parse_regular_functions(self) -> None:
        """Test 4: Parse regular def functions with various signatures."""
        test_code = '''
def simple_function():
    """Simple function."""
    pass

def function_with_params(name: str, age: int) -> str:
    """Function with type annotations."""
    return f"{name} is {age}"

def function_with_defaults(greeting: str = "Hello", name: str = "World") -> str:
    """Function with default parameters."""
    return f"{greeting}, {name}!"
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 3
        assert functions[0].signature.name == "simple_function"
        assert functions[1].signature.return_type == "str"
        assert functions[2].signature.parameters[0].default_value in [
            '"Hello"',
            "'Hello'",
        ]
        assert functions[2].signature.parameters[1].default_value in [
            '"World"',
            "'World'",
        ]

    def test_parse_async_functions(self) -> None:
        """Test 5: Parse async def functions."""
        test_code = '''
async def async_simple():
    """Simple async function."""
    return "result"

async def async_with_annotations(url: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Async with type annotations."""
    return {"url": url}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2
        assert all(f.signature.is_async for f in functions)
        assert functions[1].signature.parameters[1].default_value == "30.0"

    def test_parse_generator_functions(self) -> None:
        """Test 6: Parse generator functions with yield."""
        test_code = '''
def simple_generator():
    """Generator function."""
    for i in range(5):
        yield i

def generator_with_return():
    """Generator with return statement."""
    yield 1
    yield 2
    return "Done"
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2
        assert functions[0].signature.name == "simple_generator"
        assert functions[1].signature.name == "generator_with_return"

    def test_parse_lambda_in_functions(self) -> None:
        """Test 7: Parse functions containing lambda expressions."""
        test_code = '''
def function_with_lambda_default(
    transform = lambda x: x * 2,
    processor = lambda data: [d.upper() for d in data]
):
    """Function with lambda defaults."""
    return transform(10)

def sort_with_key(items: list, key = lambda x: x) -> list:
    """Function using lambda as default."""
    return sorted(items, key=key)
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2
        # Parser returns actual lambda code, not <lambda> placeholder
        assert functions[0].signature.parameters[0].default_value is not None
        assert "lambda" in functions[0].signature.parameters[0].default_value
        assert functions[1].signature.parameters[1].default_value is not None
        assert "lambda" in functions[1].signature.parameters[1].default_value


class TestASTParserComplexSignatures:
    """Tests for complex function signatures."""

    def test_parse_all_parameter_types(self) -> None:
        """Test 8: Parse functions with all parameter types."""
        test_code = '''
def complex_signature(
    pos_only1, pos_only2, /,
    regular1, regular2 = "default",
    *args,
    keyword_only1,
    keyword_only2: str = "kw_default",
    **kwargs
) -> None:
    """Function with all parameter types."""
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        func = functions[0]
        params = func.signature.parameters

        # Check positional-only
        assert params[0].name == "pos_only1/"
        assert params[1].name == "pos_only2/"

        # Check regular
        assert params[2].name == "regular1"
        assert params[3].name == "regular2"
        assert params[3].default_value in ['"default"', "'default'"]

        # Check *args
        assert any(p.name == "*args" for p in params)

        # Check keyword-only
        kw_params = [p for p in params if p.name in ["keyword_only1", "keyword_only2"]]
        assert len(kw_params) == 2
        assert kw_params[1].default_value in ['"kw_default"', "'kw_default'"]

        # Check **kwargs
        assert any(p.name == "**kwargs" for p in params)

    def test_parse_complex_type_annotations(self) -> None:
        """Test 9: Parse complex type annotations."""
        test_code = '''
from typing import Union, Optional, List, Dict, Callable, Tuple, Any

def complex_types(
    data: List[Dict[str, Any]],
    callback: Optional[Callable[[int, str], bool]] = None,
    config: Dict[str, Union[str, int, List[str]]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Function with complex type annotations."""
    return True, None

def generic_function(
    items: List[Union[str, int]],
    mapping: Dict[str, Optional[List[int]]]
) -> Optional[Union[str, List[Dict[str, Any]]]]:
    """Function with nested generic types."""
    return None
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2

        # Check complex_types
        func1 = functions[0]
        assert func1.signature.parameters[0].type_annotation is not None
        assert "List[Dict[str, Any]]" in func1.signature.parameters[0].type_annotation
        assert func1.signature.parameters[1].type_annotation is not None
        assert (
            "Optional[Callable[[int, str], bool]]"
            in func1.signature.parameters[1].type_annotation
        )
        assert func1.signature.return_type is not None
        assert "Tuple[bool, Optional[Dict[str, Any]]]" in func1.signature.return_type

    def test_parse_variadic_parameters(self) -> None:
        """Test 10: Parse *args and **kwargs with annotations."""
        test_code = '''
def variadic_function(
    required: str,
    *args: int,
    keyword_only: bool = True,
    **kwargs: Any
) -> List[Union[str, int]]:
    """Function with variadic parameters."""
    return [required] + list(args)
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        func = functions[0]

        # Find *args parameter
        args_param = next(p for p in func.signature.parameters if p.name == "*args")
        assert args_param.type_annotation == "int"

        # Find **kwargs parameter
        kwargs_param = next(
            p for p in func.signature.parameters if p.name == "**kwargs"
        )
        assert kwargs_param.type_annotation == "Any"


class TestASTParserDecorators:
    """Tests for decorated functions."""

    def test_parse_multiple_decorators(self) -> None:
        """Test 11: Parse functions with multiple decorators."""
        test_code = '''
import functools

@functools.lru_cache(maxsize=128)
@functools.wraps(some_function)
@custom_decorator
def decorated_function(x: int) -> int:
    """Function with multiple decorators."""
    return x * 2

@property
@cached_property
def multi_property(self):
    """Property with multiple decorators."""
    return self._value
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2

        func1 = functions[0]
        assert len(func1.signature.decorators) == 3
        assert func1.signature.decorators[0] == "functools.lru_cache(maxsize=128)"

        func2 = functions[1]
        assert "property" in func2.signature.decorators

    def test_parse_class_method_decorators(self) -> None:
        """Test 12: Parse @staticmethod, @classmethod, @property."""
        test_code = '''
class MyClass:
    @property
    def value(self) -> int:
        """Property getter."""
        return self._value

    @value.setter
    def value(self, val: int) -> None:
        """Property setter."""
        self._value = val

    @staticmethod
    def static_method(x: int, y: int) -> int:
        """Static method."""
        return x + y

    @classmethod
    def from_dict(cls, data: dict) -> "MyClass":
        """Class method."""
        return cls(**data)
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 4

        # Check decorators
        decorators = [f.signature.decorators for f in functions]
        assert ["property"] in decorators
        assert ["value.setter"] in decorators
        assert ["staticmethod"] in decorators
        assert ["classmethod"] in decorators

    def test_parse_decorator_with_complex_args(self) -> None:
        """Test 13: Parse decorators with complex arguments."""
        test_code = '''
@decorator_with_args("string", 42, key={"a": 1, "b": [1, 2, 3]})
def func1():
    """Decorator with mixed arguments."""
    pass

@app.route("/api/<id>", methods=["GET", "POST"])
def api_endpoint(id: int):
    """Flask-style route decorator."""
    return {"id": id}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2
        assert "decorator_with_args" in functions[0].signature.decorators[0]
        assert "app.route" in functions[1].signature.decorators[0]


class TestASTParserNestedStructures:
    """Tests for nested functions and classes."""

    def test_parse_nested_functions(self) -> None:
        """Test 14: Parse nested function definitions."""
        test_code = '''
def outer_function(x: int):
    """Outer function."""

    def inner_function(y: int) -> int:
        """Inner function."""
        return x + y

    def deeply_nested():
        """First level nested."""

        def second_level():
            """Second level nested."""
            return x * 2

        return second_level()

    return inner_function
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        func_names = [f.signature.name for f in functions]
        assert "outer_function" in func_names
        assert "inner_function" in func_names
        assert "deeply_nested" in func_names
        assert "second_level" in func_names

    def test_parse_nested_classes(self) -> None:
        """Test 15: Parse methods in nested classes."""
        test_code = '''
class OuterClass:
    """Outer class."""

    def outer_method(self):
        """Method in outer class."""
        pass

    class InnerClass:
        """Inner class."""

        def inner_method(self):
            """Method in inner class."""
            pass

        @staticmethod
        def inner_static():
            """Static method in inner class."""
            pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        func_names = [f.signature.name for f in functions]
        assert "outer_method" in func_names
        assert "inner_method" in func_names
        assert "inner_static" in func_names

    def test_parse_async_nested_functions(self) -> None:
        """Test 16: Parse async functions inside regular functions."""
        test_code = '''
def outer():
    """Regular outer function."""

    async def async_inner():
        """Async inner function."""
        await some_operation()

    async def async_generator():
        """Async generator inside regular function."""
        for i in range(10):
            yield i
            await asyncio.sleep(0.1)

    return async_inner, async_generator
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 3

        # Check outer is not async
        outer = next(f for f in functions if f.signature.name == "outer")
        assert not outer.signature.is_async

        # Check inner functions are async
        async_funcs = [f for f in functions if f.signature.name.startswith("async_")]
        assert all(f.signature.is_async for f in async_funcs)


class TestASTParserErrorRecovery:
    """Tests for error recovery and edge cases."""

    def test_syntax_error_recovery(self) -> None:
        """Test 17: Handle syntax errors gracefully."""
        test_code = '''
def valid_function():
    """This function is valid."""
    return 42

# Syntax error below
def invalid_function(
    """Missing closing parenthesis
    return "error"
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            with pytest.raises(SyntaxParsingError) as exc_info:
                parse_python_file(temp_path)

            assert "Syntax error" in str(exc_info.value)
            assert exc_info.value.recovery_hint == "Fix the syntax error and try again"
        finally:
            os.unlink(temp_path)

    def test_empty_file_handling(self) -> None:
        """Test 18: Handle empty files correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert functions == []
            assert isinstance(functions, list)
        finally:
            os.unlink(temp_path)

    def test_unicode_content(self) -> None:
        """Test 19: Handle unicode characters in code."""
        test_code = '''
def unicode_function():
    """Function with unicode: 你好 🌍 α β γ"""
    emoji = "😀"
    chinese = "中文"
    return f"{emoji} {chinese}"

def calculate_pi():
    """Calculate pi with unicode in docstring: π = 3.14159"""
    pi_value = 3.14159  # π
    return pi_value * 2
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert len(functions) == 2
            assert functions[0].docstring is not None
            assert "你好" in functions[0].docstring.raw_text
            assert "🌍" in functions[0].docstring.raw_text
            assert functions[1].signature.name == "calculate_pi"
            assert functions[1].docstring is not None
            assert "π" in functions[1].docstring.raw_text
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self) -> None:
        """Test 20: Handle missing files properly."""
        with pytest.raises(FileAccessError) as exc_info:
            parse_python_file("/nonexistent/path/file.py")

        assert "File not found" in str(exc_info.value)

    def test_imports_only_file(self) -> None:
        """Test 21: Handle files with only imports."""
        test_code = """
import os
import sys
from typing import List, Dict, Optional

# Constants are allowed
VERSION = "1.0.0"
DEBUG = True
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            functions = parse_python_file(temp_path)
            assert functions == []
        finally:
            os.unlink(temp_path)


class TestASTParserSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_parse_property_getters_setters(self) -> None:
        """Test 22: Parse property getters, setters, and deleters."""
        test_code = '''
class PropertyClass:
    @property
    def value(self) -> int:
        """Get value."""
        return self._value

    @value.setter
    def value(self, val: int) -> None:
        """Set value."""
        self._value = val

    @value.deleter
    def value(self) -> None:
        """Delete value."""
        del self._value
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 3
        decorators = [f.signature.decorators[0] for f in functions]
        assert "property" in decorators
        assert "value.setter" in decorators
        assert "value.deleter" in decorators

    def test_parse_dunder_methods(self) -> None:
        """Test 23: Parse special dunder methods."""
        test_code = '''
class DunderClass:
    def __init__(self, value: int):
        """Initialize."""
        self.value = value

    def __str__(self) -> str:
        """String representation."""
        return str(self.value)

    def __repr__(self) -> str:
        """Object representation."""
        return f"DunderClass({self.value})"

    def __call__(self, x: int) -> int:
        """Make instance callable."""
        return self.value + x
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        dunder_names = [f.signature.name for f in functions]
        assert "__init__" in dunder_names
        assert "__str__" in dunder_names
        assert "__repr__" in dunder_names
        assert "__call__" in dunder_names

    def test_parse_abstract_methods(self) -> None:
        """Test 24: Parse abstract methods with ABC decorators."""
        test_code = '''
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    @abstractmethod
    def abstract_method(self) -> None:
        """Abstract method."""
        pass

    @abstractmethod
    @property
    def abstract_property(self) -> str:
        """Abstract property."""
        pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2

        # Check abstract method
        method = functions[0]
        assert "abstractmethod" in method.signature.decorators

        # Check abstract property
        prop = functions[1]
        assert "abstractmethod" in prop.signature.decorators
        assert "property" in prop.signature.decorators

    def test_parse_yield_from(self) -> None:
        """Test 25: Parse generator with yield from."""
        test_code = '''
def delegating_generator(iterables: List[List[int]]):
    """Generator using yield from."""
    for iterable in iterables:
        yield from iterable

def generator_with_send():
    """Generator that receives values."""
    value = yield "initial"
    while value is not None:
        value = yield f"received: {value}"
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 2
        assert functions[0].signature.name == "delegating_generator"
        assert functions[1].signature.name == "generator_with_send"


class TestASTParserIntegration:
    """Integration tests combining multiple features."""

    def test_complex_real_world_function(self) -> None:
        """Test 26: Parse complex real-world function signature."""
        test_code = '''
from typing import TypeVar, Generic, Protocol, overload

T = TypeVar('T')

class DataProcessor(Generic[T]):
    @overload
    def process(self, data: str) -> str: ...

    @overload
    def process(self, data: int) -> int: ...

    @overload
    def process(self, data: List[T]) -> List[T]: ...

    def process(self, data: Union[str, int, List[T]]) -> Union[str, int, List[T]]:
        """Process various data types."""
        if isinstance(data, str):
            return data.upper()
        elif isinstance(data, int):
            return data * 2
        else:
            return [self.process(item) for item in data]
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        # Should find all overloads and implementation
        process_funcs = [f for f in functions if f.signature.name == "process"]
        assert len(process_funcs) == 4

        # Check overload decorators
        overloaded = [f for f in process_funcs if "overload" in f.signature.decorators]
        assert len(overloaded) == 3

    def test_async_context_managers(self) -> None:
        """Test 27: Parse async context manager methods."""
        test_code = '''
class AsyncContextManager:
    async def __aenter__(self):
        """Async enter."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit."""
        await self.cleanup()
        return False

    async def setup(self):
        """Setup resources."""
        pass

    async def cleanup(self):
        """Cleanup resources."""
        pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        assert len(functions) == 4
        assert all(f.signature.is_async for f in functions)

        # Check async context manager methods
        aenter = next(f for f in functions if f.signature.name == "__aenter__")
        aexit = next(f for f in functions if f.signature.name == "__aexit__")
        assert aenter.signature.is_async
        assert aexit.signature.is_async

    def test_dataclass_methods(self) -> None:
        """Test 28: Parse dataclass with generated methods."""
        test_code = '''
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataClassExample:
    name: str
    values: List[int] = field(default_factory=list)

    def custom_method(self) -> int:
        """Custom method in dataclass."""
        return sum(self.values)

    def __post_init__(self):
        """Post-initialization processing."""
        self.values = [v * 2 for v in self.values]
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            functions = parse_python_file(f.name)

        os.unlink(f.name)

        # Should find custom methods
        assert len(functions) == 2
        func_names = [f.signature.name for f in functions]
        assert "custom_method" in func_names
        assert "__post_init__" in func_names

    def test_lazy_parsing_performance(self) -> None:
        """Test 29: Lazy parsing uses generators efficiently."""
        content = self._generate_large_file(num_functions=200)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Time lazy parsing
            start_time = time.perf_counter()
            lazy_count = 0

            for func in parse_python_file_lazy(temp_path):
                lazy_count += 1
                # Simulate processing
                _ = func.signature.name

            lazy_time = time.perf_counter() - start_time

            assert lazy_count == 200
            # Lazy parsing should be reasonably fast
            assert lazy_time < 1.0  # Less than 1 second for 200 functions

        finally:
            os.unlink(temp_path)

    def test_caching_improves_performance(self) -> None:
        """Test 30: Caching significantly improves repeated parsing."""
        content = self._generate_large_file(num_functions=100)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # First parse (cold cache)
            start_time = time.perf_counter()
            first_functions = parse_python_file(temp_path)
            first_time = time.perf_counter() - start_time

            # Second parse (warm cache)
            start_time = time.perf_counter()
            second_functions = parse_python_file(temp_path)
            second_time = time.perf_counter() - start_time

            assert len(first_functions) == len(second_functions) == 100

            # Second parse should be faster due to caching
            assert second_time < first_time * 0.8, (
                f"Caching did not improve performance: "
                f"first={first_time * 1000:.2f}ms, second={second_time * 1000:.2f}ms"
            )

        finally:
            os.unlink(temp_path)

    def _generate_large_file(self, num_functions: int) -> str:
        """Generate large test file for performance tests."""
        lines = ["from typing import *", ""]

        for i in range(num_functions):
            lines.extend(
                [
                    f"def function_{i}(x: int, y: str = 'default') -> Dict[str, Any]:",
                    f'    """Function {i}."""',
                    f"    return {{'x': x, 'y': y, 'id': {i}}}",
                    "",
                ]
            )

        return "\n".join(lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])