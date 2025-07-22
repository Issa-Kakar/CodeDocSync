"""
Comprehensive tests for AST parser covering all function types.

This module tests parsing of various function types including regular functions,
async functions, generator functions, lambda functions, nested functions, and
class methods (regular, static, and class methods).
"""

import tempfile
from pathlib import Path

from codedocsync.parser.ast_parser import parse_python_file
from typing import Any, Callable, Dict, List, Optional


class TestASTParserFunctionTypes:
    """Test suite for AST parser function type handling."""

    def test_parse_regular_functions(self) -> None:
        """Test parsing of regular def functions with various signatures."""
        test_code = '''
def simple_function():
    """A simple function with no parameters."""
    pass

def function_with_params(name, age):
    """Function with positional parameters."""
    return f"{name} is {age} years old"

def function_with_defaults(greeting="Hello", name="World"):
    """Function with default parameters."""
    return f"{greeting}, {name}!"

def function_with_annotations(x: int, y: int) -> int:
    """Function with type annotations."""
    return x + y

def function_with_complex_annotations(
    data: List[Dict[str, Any]],
    callback: Callable[[int], str] | None = None
) -> Dict[str, List[int]]:
    """Function with complex type annotations."""
    return {}

def function_with_all_param_types(
    pos_only_param, /,
    regular_param,
    *args,
    keyword_only: str,
    **kwargs
) -> None:
    """Function with all parameter types."""
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(functions) == 6

        # Test simple function
        simple_func = functions[0]
        assert simple_func.signature.name == "simple_function"
        assert len(simple_func.signature.parameters) == 0
        assert not simple_func.signature.is_async
        assert not simple_func.signature.is_method
        assert simple_func.docstring is not None
        assert simple_func.docstring is not None
        assert "simple function" in simple_func.docstring.raw_text

        # Test function with params
        param_func = functions[1]
        assert param_func.signature.name == "function_with_params"
        assert len(param_func.signature.parameters) == 2
        assert param_func.signature.parameters[0].name == "name"
        assert param_func.signature.parameters[1].name == "age"
        assert all(p.is_required for p in param_func.signature.parameters)

        # Test function with defaults
        default_func = functions[2]
        assert default_func.signature.name == "function_with_defaults"
        assert len(default_func.signature.parameters) == 2
        assert default_func.signature.parameters[0].default_value == '"Hello"'
        assert default_func.signature.parameters[1].default_value == '"World"'
        assert all(not p.is_required for p in default_func.signature.parameters)

        # Test function with annotations
        annotated_func = functions[3]
        assert annotated_func.signature.name == "function_with_annotations"
        assert annotated_func.signature.parameters[0].type_annotation == "int"
        assert annotated_func.signature.parameters[1].type_annotation == "int"
        assert annotated_func.signature.return_type == "int"

        # Test function with complex annotations
        complex_func = functions[4]
        assert complex_func.signature.name == "function_with_complex_annotations"
        assert complex_func.signature.parameters[0].type_annotation is not None
        assert (
            "List[Dict[str, Any]]"
            in complex_func.signature.parameters[0].type_annotation
        )
        assert complex_func.signature.parameters[1].default_value == "None"
        assert complex_func.signature.return_type is not None
        assert "Dict[str, List[int]]" in complex_func.signature.return_type

        # Test function with all parameter types
        all_params_func = functions[5]
        assert all_params_func.signature.name == "function_with_all_param_types"
        # Check positional-only parameter
        assert all_params_func.signature.parameters[0].name == "pos_only_param/"
        # Check regular parameter
        assert all_params_func.signature.parameters[1].name == "regular_param"
        # Check *args
        assert any(p.name == "*args" for p in all_params_func.signature.parameters)
        # Check keyword-only
        assert any(
            p.name == "keyword_only" and p.type_annotation == "str"
            for p in all_params_func.signature.parameters
        )
        # Check **kwargs
        assert any(p.name == "**kwargs" for p in all_params_func.signature.parameters)

    def test_parse_async_functions(self) -> None:
        """Test parsing of async def functions."""
        test_code = '''
async def async_simple():
    """Simple async function."""
    return "async result"

async def async_with_await():
    """Async function using await."""
    result = await some_async_call()
    return result

async def async_generator():
    """Async generator function."""
    for i in range(10):
        yield i
        await asyncio.sleep(0.1)

async def async_with_annotations(
    url: str,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Async function with type annotations."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            return await response.json()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(functions) == 4

        # All functions should be async
        assert all(func.signature.is_async for func in functions)

        # Test simple async
        async_simple = functions[0]
        assert async_simple.signature.name == "async_simple"
        assert async_simple.signature.is_async
        assert not async_simple.signature.is_method

        # Test async with await
        async_await = functions[1]
        assert async_await.signature.name == "async_with_await"
        assert async_await.docstring is not None
        assert "await" in async_await.docstring.raw_text

        # Test async generator
        async_gen = functions[2]
        assert async_gen.signature.name == "async_generator"
        assert async_gen.docstring is not None
        assert "generator" in async_gen.docstring.raw_text

        # Test async with annotations
        async_annotated = functions[3]
        assert async_annotated.signature.name == "async_with_annotations"
        assert async_annotated.signature.parameters[0].type_annotation == "str"
        assert async_annotated.signature.parameters[1].type_annotation == "float"
        assert async_annotated.signature.parameters[1].default_value == "30.0"
        assert async_annotated.signature.return_type is not None
        assert "Dict[str, Any]" in async_annotated.signature.return_type

    def test_parse_generator_functions(self) -> None:
        """Test parsing of functions with yield statements."""
        test_code = '''
def simple_generator():
    """Simple generator function."""
    for i in range(5):
        yield i

def generator_with_params(start: int, stop: int, step: int = 1):
    """Generator with parameters and type hints."""
    current = start
    while current < stop:
        yield current
        current += step

def generator_with_return():
    """Generator that also uses return."""
    yield 1
    yield 2
    return "Done"

def generator_expression_factory(n: int) -> Generator[int, None, None]:
    """Function returning a generator expression."""
    return (i ** 2 for i in range(n))

def yield_from_generator(iterables: List[List[int]]):
    """Generator using yield from."""
    for iterable in iterables:
        yield from iterable
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        assert len(functions) == 5

        # Test simple generator
        simple_gen = functions[0]
        assert simple_gen.signature.name == "simple_generator"
        assert simple_gen.docstring is not None
        assert "generator" in simple_gen.docstring.raw_text

        # Test generator with params
        param_gen = functions[1]
        assert param_gen.signature.name == "generator_with_params"
        assert len(param_gen.signature.parameters) == 3
        assert param_gen.signature.parameters[2].default_value == "1"

        # Test generator with return
        return_gen = functions[2]
        assert return_gen.signature.name == "generator_with_return"

        # Test generator expression factory
        expr_factory = functions[3]
        assert expr_factory.signature.name == "generator_expression_factory"
        assert expr_factory.signature.return_type is not None
        assert "Generator[int, None, None]" in expr_factory.signature.return_type

        # Test yield from generator
        yield_from = functions[4]
        assert yield_from.signature.name == "yield_from_generator"
        assert yield_from.docstring is not None
        assert "yield from" in yield_from.docstring.raw_text

    def test_parse_lambda_functions(self) -> None:
        """Test parsing of lambda expressions within functions."""
        test_code = '''
def function_with_lambdas():
    """Function containing lambda expressions."""
    # Simple lambda
    square = lambda x: x ** 2

    # Lambda with multiple params
    add = lambda x, y: x + y

    # Lambda with default param
    greet = lambda name="World": f"Hello, {name}!"

    # Lambda in list comprehension
    operations = [lambda x: x + i for i in range(5)]

    # Lambda as default value
    def process(data, transform=lambda x: x):
        return transform(data)

    return square, add, greet, operations

# Lambda as module-level variable
module_lambda = lambda x, y=10: x * y

# Function using lambda as default parameter
def sort_with_key(items: list, key: Callable = lambda x: x) -> list:
    """Sort items using a key function."""
    return sorted(items, key=key)
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        # We should find the regular functions, not the lambdas themselves
        assert len(functions) == 3  # function_with_lambdas, process, sort_with_key

        # Test function containing lambdas
        lambda_container = functions[0]
        assert lambda_container.signature.name == "function_with_lambdas"
        assert lambda_container.docstring is not None
        assert "lambda" in lambda_container.docstring.raw_text

        # Test nested function with lambda default
        process_func = functions[1]
        assert process_func.signature.name == "process"
        assert process_func.signature.parameters[1].default_value == "<lambda>"

        # Test function with lambda as default parameter
        sort_func = functions[2]
        assert sort_func.signature.name == "sort_with_key"
        assert sort_func.signature.parameters[1].default_value == "<lambda>"

    def test_parse_nested_functions(self) -> None:
        """Test parsing of functions defined inside other functions."""
        test_code = '''
def outer_function(x: int):
    """Outer function containing nested functions."""

    def inner_function(y: int) -> int:
        """Inner function accessing outer scope."""
        return x + y

    def deeply_nested():
        """First level nested function."""

        def second_level():
            """Second level nested function."""

            def third_level():
                """Third level nested function."""
                return x * 3

            return third_level()

        return second_level()

    async def async_inner():
        """Async nested function."""
        return await some_async_operation(x)

    # Nested generator
    def nested_generator():
        """Generator nested inside function."""
        for i in range(x):
            yield i * x

    # Return nested functions
    return inner_function, deeply_nested, async_inner, nested_generator

def decorator_factory(prefix: str):
    """Factory function creating decorators."""

    def decorator(func):
        """Decorator function."""

        def wrapper(*args, **kwargs):
            """Wrapper function."""
            print(f"{prefix}: calling {func.__name__}")
            return func(*args, **kwargs)

        return wrapper

    return decorator
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        # The parser walks all nodes, so it finds all functions including nested ones
        function_names = [f.signature.name for f in functions]

        # Check that we found all functions
        assert "outer_function" in function_names
        assert "inner_function" in function_names
        assert "deeply_nested" in function_names
        assert "second_level" in function_names
        assert "third_level" in function_names
        assert "async_inner" in function_names
        assert "nested_generator" in function_names
        assert "decorator_factory" in function_names
        assert "decorator" in function_names
        assert "wrapper" in function_names

        # Find specific functions and check properties
        async_inner = next(f for f in functions if f.signature.name == "async_inner")
        assert async_inner.signature.is_async

        inner_func = next(f for f in functions if f.signature.name == "inner_function")
        assert inner_func.signature.return_type == "int"
        assert inner_func.signature.parameters[0].name == "y"

    def test_parse_class_methods(self) -> None:
        """Test parsing of methods inside classes (regular, static, class methods)."""
        test_code = '''
class MyClass:
    """Test class with various method types."""

    def __init__(self, name: str):
        """Constructor method."""
        self.name = name

    def regular_method(self, value: int) -> str:
        """Regular instance method."""
        return f"{self.name}: {value}"

    @classmethod
    def class_method(cls, data: dict) -> "MyClass":
        """Class method creating instance."""
        return cls(data.get("name", "default"))

    @staticmethod
    def static_method(x: float, y: float) -> float:
        """Static method performing calculation."""
        return x + y

    @property
    def name_upper(self) -> str:
        """Property getter."""
        return self.name.upper()

    @name_upper.setter
    def name_upper(self, value: str) -> None:
        """Property setter."""
        self.name = value.lower()

    async def async_method(self, url: str) -> dict:
        """Async instance method."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

    def __str__(self) -> str:
        """String representation."""
        return self.name

    def __repr__(self) -> str:
        """Object representation."""
        return f"MyClass(name={self.name!r})"

    @classmethod
    async def async_class_method(cls) -> List["MyClass"]:
        """Async class method."""
        data = await fetch_data()
        return [cls(item["name"]) for item in data]

class ChildClass(MyClass):
    """Child class with method overrides."""

    def regular_method(self, value: int) -> str:
        """Override of parent method."""
        return f"Child {super().regular_method(value)}"

    @staticmethod
    def new_static_method() -> None:
        """New static method in child."""
        pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()

            functions = parse_python_file(f.name)

        # Clean up
        Path(f.name).unlink()

        # Check that we found all methods
        method_names = [f.signature.name for f in functions]

        # MyClass methods
        assert "__init__" in method_names
        assert "regular_method" in method_names
        assert "class_method" in method_names
        assert "static_method" in method_names
        assert "name_upper" in method_names  # Both getter and setter
        assert "async_method" in method_names
        assert "__str__" in method_names
        assert "__repr__" in method_names
        assert "async_class_method" in method_names
        assert "new_static_method" in method_names

        # Test regular method
        regular_methods = [f for f in functions if f.signature.name == "regular_method"]
        assert len(regular_methods) == 2  # One in parent, one in child
        for method in regular_methods:
            assert method.signature.is_method
            assert method.signature.parameters[0].name == "self"

        # Test class method
        class_method = next(f for f in functions if f.signature.name == "class_method")
        assert class_method.signature.is_method
        assert "classmethod" in class_method.signature.decorators
        assert class_method.signature.parameters[0].name == "cls"
        assert class_method.signature.return_type == '"MyClass"'

        # Test static method
        static_method = next(
            f for f in functions if f.signature.name == "static_method"
        )
        assert static_method.signature.is_method
        assert "staticmethod" in static_method.signature.decorators
        assert len(static_method.signature.parameters) == 2
        assert all(p.name in ["x", "y"] for p in static_method.signature.parameters)

        # Test property methods
        property_methods = [f for f in functions if f.signature.name == "name_upper"]
        assert len(property_methods) == 2  # getter and setter
        getter = next(
            f for f in property_methods if "property" in f.signature.decorators
        )
        setter = next(
            f for f in property_methods if "name_upper.setter" in f.signature.decorators
        )
        assert getter.signature.is_method
        assert setter.signature.is_method

        # Test async method
        async_method = next(f for f in functions if f.signature.name == "async_method")
        assert async_method.signature.is_async
        assert async_method.signature.is_method

        # Test async class method
        async_class = next(
            f for f in functions if f.signature.name == "async_class_method"
        )
        assert async_class.signature.is_async
        assert async_class.signature.is_method
        assert "classmethod" in async_class.signature.decorators

        # Test dunder methods
        init_method = next(f for f in functions if f.signature.name == "__init__")
        assert init_method.signature.is_method
        assert init_method.signature.parameters[0].name == "self"
        assert init_method.signature.parameters[1].name == "name"
        assert init_method.signature.parameters[1].type_annotation == "str"