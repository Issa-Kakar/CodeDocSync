"""
Comprehensive tests for AST parser decorator handling.

Tests parsing of decorated functions and complex structures including:
- Single decorators
- Multiple decorators
- Decorators with arguments
- Class decorators (@property, @staticmethod, @classmethod)
- Nested classes and functions
- Async functions with decorators
"""

import os
import tempfile

from codedocsync.parser.ast_parser import parse_python_file


class TestASTParserDecorators:
    """Test AST parser handling of decorators and complex structures."""

    def test_parse_single_decorator(self) -> None:
        """Test parsing functions with one decorator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
import functools

@functools.cache
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@deprecated
def old_function():
    """This function is deprecated."""
    pass
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Should find 2 functions
                assert len(functions) == 2

                # Check fibonacci function
                fib_func = functions[0]
                assert fib_func.signature.name == "fibonacci"
                assert len(fib_func.signature.decorators) == 1
                assert fib_func.signature.decorators[0] == "functools.cache"
                assert fib_func.line_number == 4
                assert fib_func.docstring is not None
                assert fib_func.docstring.raw_text == "Calculate fibonacci number."

                # Check old_function
                old_func = functions[1]
                assert old_func.signature.name == "old_function"
                assert len(old_func.signature.decorators) == 1
                assert old_func.signature.decorators[0] == "deprecated"
                assert old_func.line_number == 11
                assert old_func.docstring is not None
                assert old_func.docstring.raw_text == "This function is deprecated."

            finally:
                os.unlink(f.name)

    def test_parse_multiple_decorators(self) -> None:
        """Test parsing functions with multiple decorators."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
import functools
import logging

@functools.lru_cache(maxsize=128)
@functools.wraps(some_function)
@logging_decorator
def complex_function(x: int, y: int) -> int:
    """Function with multiple decorators."""
    return x + y

@property
@cached_property
@synchronized
def multi_decorated_property(self):
    """Property with multiple decorators."""
    return self._value
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Should find 2 functions
                assert len(functions) == 2

                # Check complex_function
                complex_func = functions[0]
                assert complex_func.signature.name == "complex_function"
                assert len(complex_func.signature.decorators) == 3
                # Decorators should be in order
                assert (
                    complex_func.signature.decorators[0]
                    == "functools.lru_cache(maxsize=128)"
                )
                assert (
                    complex_func.signature.decorators[1]
                    == "functools.wraps(some_function)"
                )
                assert complex_func.signature.decorators[2] == "logging_decorator"
                assert complex_func.line_number == 5

                # Check multi_decorated_property
                prop_func = functions[1]
                assert prop_func.signature.name == "multi_decorated_property"
                assert len(prop_func.signature.decorators) == 3
                assert prop_func.signature.decorators[0] == "property"
                assert prop_func.signature.decorators[1] == "cached_property"
                assert prop_func.signature.decorators[2] == "synchronized"
                assert (
                    prop_func.signature.is_method is True
                )  # property decorator makes it a method

            finally:
                os.unlink(f.name)

    def test_parse_decorator_with_arguments(self) -> None:
        """Test parsing decorators with arguments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
import functools
from typing import Optional

@decorator_with_args("string_arg", 42, keyword=True)
def func1():
    """Function with decorator arguments."""
    pass

@functools.lru_cache(maxsize=None)
def func2(n: int) -> int:
    """LRU cache with None maxsize."""
    return n * 2

@retry(max_attempts=3, delay=1.5, exceptions=(ValueError, TypeError))
def func3(data: dict) -> Optional[str]:
    """Retry decorator with multiple arguments."""
    return data.get("key")

@app.route("/api/users", methods=["GET", "POST"])
def api_endpoint():
    """Flask-style route decorator."""
    return {"status": "ok"}

@validate_schema(
    input_schema=UserSchema,
    output_schema=ResponseSchema,
    strict=True
)
def func4(user_data: dict) -> dict:
    """Decorator with multiline arguments."""
    return {"id": 1, **user_data}
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Should find 5 functions
                assert len(functions) == 5

                # Check func1
                func1 = functions[0]
                assert func1.signature.name == "func1"
                assert len(func1.signature.decorators) == 1
                assert (
                    func1.signature.decorators[0]
                    == 'decorator_with_args("string_arg", 42, keyword=True)'
                )

                # Check func2
                func2 = functions[1]
                assert func2.signature.name == "func2"
                assert (
                    func2.signature.decorators[0] == "functools.lru_cache(maxsize=None)"
                )

                # Check func3
                func3 = functions[2]
                assert func3.signature.name == "func3"
                assert (
                    func3.signature.decorators[0]
                    == "retry(max_attempts=3, delay=1.5, exceptions=(ValueError, TypeError))"
                )

                # Check api_endpoint
                api_func = functions[3]
                assert api_func.signature.name == "api_endpoint"
                assert (
                    api_func.signature.decorators[0]
                    == 'app.route("/api/users", methods=["GET", "POST"])'
                )

                # Check func4
                func4 = functions[4]
                assert func4.signature.name == "func4"
                assert "validate_schema" in func4.signature.decorators[0]

            finally:
                os.unlink(f.name)

    def test_parse_class_decorators(self) -> None:
        """Test parsing @property, @staticmethod, @classmethod decorators."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
class MyClass:
    """Test class with various method decorators."""

    def __init__(self, value):
        self._value = value
        self._cached = None

    @property
    def value(self):
        """Get the value."""
        return self._value

    @value.setter
    def value(self, new_value):
        """Set the value."""
        self._value = new_value

    @value.deleter
    def value(self):
        """Delete the value."""
        del self._value

    @staticmethod
    def static_method(x: int, y: int) -> int:
        """Static method example."""
        return x + y

    @classmethod
    def from_string(cls, string_value: str):
        """Create instance from string."""
        return cls(int(string_value))

    @property
    @functools.lru_cache(maxsize=1)
    def expensive_property(self):
        """Cached property calculation."""
        import time
        time.sleep(0.1)
        return self._value * 2
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Should find 7 functions (including __init__)
                assert len(functions) == 7

                # Find each function by name
                func_dict = {f.signature.name: f for f in functions}

                # Check __init__
                init_func = func_dict["__init__"]
                assert init_func.signature.is_method is True
                assert len(init_func.signature.decorators) == 0

                # Check property getter
                value_getter = func_dict["value"]
                assert value_getter.signature.decorators == ["property"]
                assert value_getter.signature.is_method is True
                assert value_getter.docstring is not None
                assert value_getter.docstring.raw_text == "Get the value."

                # Check static method
                static_func = func_dict["static_method"]
                assert static_func.signature.decorators == ["staticmethod"]
                assert static_func.signature.is_method is True
                assert len(static_func.signature.parameters) == 2
                assert (
                    static_func.signature.parameters[0].name == "x"
                )  # No self parameter

                # Check class method
                class_func = func_dict["from_string"]
                assert class_func.signature.decorators == ["classmethod"]
                assert class_func.signature.is_method is True
                assert class_func.signature.parameters[0].name == "cls"

                # Check expensive_property with multiple decorators
                expensive_prop = func_dict["expensive_property"]
                assert len(expensive_prop.signature.decorators) == 2
                assert expensive_prop.signature.decorators[0] == "property"
                assert (
                    expensive_prop.signature.decorators[1]
                    == "functools.lru_cache(maxsize=1)"
                )

            finally:
                os.unlink(f.name)

    def test_parse_nested_classes_and_functions(self) -> None:
        """Test parsing deeply nested structures."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
def outer_function(x):
    """Outer function with nested functions."""

    @decorator1
    def inner_function(y):
        """First level nested function."""

        @decorator2
        @decorator3
        def deeply_nested(z):
            """Second level nested function."""
            return x + y + z

        return deeply_nested

    class NestedClass:
        """Nested class inside function."""

        @property
        def nested_property(self):
            """Property in nested class."""
            return x

        @staticmethod
        def nested_static():
            """Static method in nested class."""

            def triple_nested():
                """Function inside static method."""
                return 42

            return triple_nested()

    return inner_function

class OuterClass:
    """Outer class with nested structures."""

    class InnerClass:
        """Inner class."""

        @classmethod
        def inner_classmethod(cls):
            """Class method in inner class."""

            def method_nested_func():
                """Function inside class method."""
                return cls.__name__

            return method_nested_func()

        class DoublyNestedClass:
            """Doubly nested class."""

            @property
            def deep_property(self):
                """Property in doubly nested class."""
                return "deep"
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Count all functions including nested ones
                func_names = [f.signature.name for f in functions]

                # Should find all functions at all nesting levels
                assert "outer_function" in func_names
                assert "inner_function" in func_names
                assert "deeply_nested" in func_names
                assert "nested_property" in func_names
                assert "nested_static" in func_names
                assert "triple_nested" in func_names
                assert "inner_classmethod" in func_names
                assert "method_nested_func" in func_names
                assert "deep_property" in func_names

                # Check decorators on nested functions
                func_dict = {f.signature.name: f for f in functions}

                # Check inner_function decorators
                inner_func = func_dict["inner_function"]
                assert inner_func.signature.decorators == ["decorator1"]
                assert inner_func.line_number == 5

                # Check deeply_nested decorators
                deeply_nested = func_dict["deeply_nested"]
                assert len(deeply_nested.signature.decorators) == 2
                assert deeply_nested.signature.decorators == [
                    "decorator2",
                    "decorator3",
                ]
                assert deeply_nested.line_number == 10

                # Check nested class methods
                nested_prop = func_dict["nested_property"]
                assert nested_prop.signature.decorators == ["property"]
                assert nested_prop.signature.is_method is True

                # Check line numbers are correct for nested functions
                assert inner_func.line_number < deeply_nested.line_number
                outer_function = func_dict["outer_function"]
                assert outer_function.line_number < inner_func.line_number

            finally:
                os.unlink(f.name)

    def test_parse_decorated_async_functions(self) -> None:
        """Test parsing async functions with decorators."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
import asyncio
from typing import Optional, List

@async_decorator
async def simple_async():
    """Simple async function with decorator."""
    await asyncio.sleep(0.1)

@retry_async(max_attempts=3)
@log_async_calls
async def decorated_async(url: str) -> Optional[str]:
    """Async function with multiple decorators."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

class AsyncClass:
    """Class with async methods."""

    @property
    async def async_property(self):
        """Async property (not typical but valid syntax)."""
        await asyncio.sleep(0.1)
        return self._value

    @staticmethod
    @async_timeout(5.0)
    async def async_static(items: List[str]) -> List[str]:
        """Decorated async static method."""
        results = []
        for item in items:
            await asyncio.sleep(0.01)
            results.append(item.upper())
        return results

    @classmethod
    @cache_async(ttl=300)
    async def async_classmethod(cls, config: dict):
        """Decorated async class method."""
        await cls.validate_config(config)
        return cls(**config)

    @synchronized_async
    @validate_async
    async def complex_async_method(self, data: dict) -> dict:
        """Async method with multiple decorators."""
        async with self.lock:
            processed = await self.process(data)
            return processed
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Find all async functions
                async_functions = [f for f in functions if f.signature.is_async]

                # Should have 6 async functions
                assert len(async_functions) == 6

                # Check simple_async
                simple_async = next(
                    f for f in functions if f.signature.name == "simple_async"
                )
                assert simple_async.signature.is_async is True
                assert simple_async.signature.decorators == ["async_decorator"]
                assert simple_async.docstring is not None
                assert (
                    simple_async.docstring.raw_text
                    == "Simple async function with decorator."
                )

                # Check decorated_async
                decorated_async = next(
                    f for f in functions if f.signature.name == "decorated_async"
                )
                assert decorated_async.signature.is_async is True
                assert len(decorated_async.signature.decorators) == 2
                assert (
                    decorated_async.signature.decorators[0]
                    == "retry_async(max_attempts=3)"
                )
                assert decorated_async.signature.decorators[1] == "log_async_calls"
                assert decorated_async.signature.return_type == "Optional[str]"

                # Check async_property
                async_prop = next(
                    f for f in functions if f.signature.name == "async_property"
                )
                assert async_prop.signature.is_async is True
                assert async_prop.signature.decorators == ["property"]
                assert async_prop.signature.is_method is True

                # Check async_static
                async_static = next(
                    f for f in functions if f.signature.name == "async_static"
                )
                assert async_static.signature.is_async is True
                assert len(async_static.signature.decorators) == 2
                assert async_static.signature.decorators[0] == "staticmethod"
                assert async_static.signature.decorators[1] == "async_timeout(5.0)"
                assert async_static.signature.is_method is True

                # Check async_classmethod
                async_classmethod = next(
                    f for f in functions if f.signature.name == "async_classmethod"
                )
                assert async_classmethod.signature.is_async is True
                assert len(async_classmethod.signature.decorators) == 2
                assert async_classmethod.signature.decorators[0] == "classmethod"
                assert (
                    async_classmethod.signature.decorators[1] == "cache_async(ttl=300)"
                )

                # Check complex_async_method
                complex_method = next(
                    f for f in functions if f.signature.name == "complex_async_method"
                )
                assert complex_method.signature.is_async is True
                assert len(complex_method.signature.decorators) == 2
                assert complex_method.signature.decorators[0] == "synchronized_async"
                assert complex_method.signature.decorators[1] == "validate_async"
                assert complex_method.signature.return_type == "dict"

            finally:
                os.unlink(f.name)

    def test_edge_cases(self) -> None:
        """Test edge cases for decorator parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''
# Decorator with complex expressions
@decorator(lambda x: x * 2, key={"a": 1, "b": [1, 2, 3]})
def func_with_lambda_decorator():
    """Function with lambda in decorator."""
    pass

# Chained attribute decorators
@app.api.v2.route("/users/<int:id>")
@requires.auth.admin
def chained_decorator_func(id: int):
    """Function with chained attribute decorators."""
    pass

# Decorator with generator expression
@parametrize("test_input,expected", [(i, i*2) for i in range(5)])
def test_parametrized(test_input, expected) -> None:
    """Parametrized test function."""
    assert test_input * 2 == expected

# No decorators but complex signature
def no_decorator_complex(
    pos_only_arg: int,
    /,
    regular_arg: str,
    *args: tuple,
    keyword_only: bool = True,
    **kwargs: dict
) -> Optional[List[str]]:
    """Function with complex signature but no decorators."""
    pass

# Empty decorator list should work
def undecorated_function():
    """Simple function without decorators."""
    return 42
'''
            )
            f.flush()

            try:
                functions = parse_python_file(f.name)

                # Should find 5 functions
                assert len(functions) == 5

                func_dict = {f.signature.name: f for f in functions}

                # Check lambda decorator
                lambda_func = func_dict["func_with_lambda_decorator"]
                assert len(lambda_func.signature.decorators) == 1
                assert "decorator" in lambda_func.signature.decorators[0]
                assert "lambda" in lambda_func.signature.decorators[0]

                # Check chained decorators
                chained_func = func_dict["chained_decorator_func"]
                assert len(chained_func.signature.decorators) == 2
                assert (
                    chained_func.signature.decorators[0]
                    == 'app.api.v2.route("/users/<int:id>")'
                )
                assert chained_func.signature.decorators[1] == "requires.auth.admin"

                # Check parametrized decorator
                param_func = func_dict["test_parametrized"]
                assert len(param_func.signature.decorators) == 1
                assert "parametrize" in param_func.signature.decorators[0]

                # Check no decorators
                no_dec_func = func_dict["no_decorator_complex"]
                assert len(no_dec_func.signature.decorators) == 0
                assert len(no_dec_func.signature.parameters) == 5

                # Check undecorated function
                undec_func = func_dict["undecorated_function"]
                assert len(undec_func.signature.decorators) == 0
                assert undec_func.docstring is not None
                assert (
                    undec_func.docstring.raw_text
                    == "Simple function without decorators."
                )

            finally:
                os.unlink(f.name)
