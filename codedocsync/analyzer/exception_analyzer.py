"""
Exception analysis functionality for detecting exceptions in code.

This module provides tools for analyzing Python code to identify
exceptions that can be raised, supporting the analyzer module's
exception documentation checks.
"""

import ast
from dataclasses import dataclass


@dataclass
class ExceptionInfo:
    """Information about an exception that can be raised."""

    exception_type: str
    description: str
    condition: str | None = None
    line_number: int | None = None
    is_re_raised: bool = False
    confidence: float = 1.0


class ExceptionAnalyzer:
    """Analyze function code to find all possible exceptions."""

    def __init__(self) -> None:
        self.builtin_exceptions = {
            "ValueError": "When an invalid value is provided",
            "TypeError": "When an invalid type is provided",
            "KeyError": "When a key is not found",
            "IndexError": "When an index is out of range",
            "AttributeError": "When an attribute is not found",
            "FileNotFoundError": "When a file is not found",
            "PermissionError": "When permission is denied",
            "RuntimeError": "When a runtime error occurs",
            "NotImplementedError": "When functionality is not implemented",
            "OSError": "When an OS-related error occurs",
            "IOError": "When an I/O operation fails",
        }

    def analyze_exceptions(self, source_code: str) -> list[ExceptionInfo]:
        """Find all exceptions that can be raised."""
        exceptions: list[ExceptionInfo] = []

        try:
            tree = ast.parse(source_code)
            self._analyze_ast(tree, exceptions)
        except SyntaxError:
            # If we can't parse, return basic exceptions
            exceptions.append(
                ExceptionInfo(
                    exception_type="Exception",
                    description="When an error occurs",
                    confidence=0.3,
                )
            )

        return self._deduplicate_exceptions(exceptions)

    def _analyze_ast(self, tree: ast.AST, exceptions: list[ExceptionInfo]) -> None:
        """Analyze AST for exception patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                self._analyze_function_node(node, exceptions)
                break

    def _analyze_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        exceptions: list[ExceptionInfo],
    ) -> None:
        """Analyze a function node for exception patterns."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                self._analyze_raise_statement(child, exceptions)
            elif isinstance(child, ast.Call):
                self._analyze_function_call(child, exceptions)
            elif isinstance(child, ast.Subscript):
                self._analyze_subscript(child, exceptions)
            elif isinstance(child, ast.Attribute):
                self._analyze_attribute_access(child, exceptions)

    def _analyze_raise_statement(
        self, node: ast.Raise, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze a raise statement."""
        if node.exc is None:
            # Bare raise - re-raising current exception
            exceptions.append(
                ExceptionInfo(
                    exception_type="Exception",
                    description="Re-raised exception",
                    is_re_raised=True,
                    line_number=node.lineno,
                    confidence=0.8,
                )
            )
            return

        exception_type = None
        if isinstance(node.exc, ast.Call):
            # raise ExceptionType(message)
            if isinstance(node.exc.func, ast.Name):
                exception_type = node.exc.func.id
        elif isinstance(node.exc, ast.Name):
            # raise exception_instance
            exception_type = node.exc.id

        if exception_type:
            description = self.builtin_exceptions.get(
                exception_type, f"When a {exception_type} condition occurs"
            )
            exceptions.append(
                ExceptionInfo(
                    exception_type=exception_type,
                    description=description,
                    line_number=node.lineno,
                    confidence=0.95,
                )
            )

    def _analyze_function_call(
        self, node: ast.Call, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze function calls that might raise exceptions."""
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            # Common functions that raise specific exceptions
            function_exceptions = {
                "open": ["FileNotFoundError", "PermissionError", "IOError"],
                "int": ["ValueError"],
                "float": ["ValueError"],
                "len": ["TypeError"],
                "max": ["ValueError"],
                "min": ["ValueError"],
                "next": ["StopIteration"],
                "iter": ["TypeError"],
                "getattr": ["AttributeError"],
                "setattr": ["AttributeError"],
                "delattr": ["AttributeError"],
            }

            if func_name in function_exceptions:
                for exc_type in function_exceptions[func_name]:
                    description = self.builtin_exceptions.get(
                        exc_type, f"When {func_name} fails"
                    )
                    exceptions.append(
                        ExceptionInfo(
                            exception_type=exc_type,
                            description=description,
                            condition=f"When calling {func_name}",
                            line_number=node.lineno,
                            confidence=0.6,
                        )
                    )

    def _analyze_subscript(
        self, node: ast.Subscript, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze subscript operations that might raise exceptions."""
        # Dictionary/list access can raise KeyError/IndexError
        exceptions.append(
            ExceptionInfo(
                exception_type="KeyError",
                description="When accessing a non-existent key",
                condition="When accessing dictionary keys",
                line_number=node.lineno,
                confidence=0.4,
            )
        )
        exceptions.append(
            ExceptionInfo(
                exception_type="IndexError",
                description="When accessing an invalid index",
                condition="When accessing list/tuple indices",
                line_number=node.lineno,
                confidence=0.4,
            )
        )

    def _analyze_attribute_access(
        self, node: ast.Attribute, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze attribute access that might raise exceptions."""
        # Attribute access can raise AttributeError
        exceptions.append(
            ExceptionInfo(
                exception_type="AttributeError",
                description="When accessing a non-existent attribute",
                condition="When accessing object attributes",
                line_number=node.lineno,
                confidence=0.3,
            )
        )

    def _deduplicate_exceptions(
        self, exceptions: list[ExceptionInfo]
    ) -> list[ExceptionInfo]:
        """Remove duplicate exceptions and merge similar ones."""
        unique_exceptions: dict[str, ExceptionInfo] = {}

        for exc in exceptions:
            key = exc.exception_type
            if key in unique_exceptions:
                # Keep the one with higher confidence
                if exc.confidence > unique_exceptions[key].confidence:
                    unique_exceptions[key] = exc
            else:
                unique_exceptions[key] = exc

        # Sort by confidence and exception type
        return sorted(
            unique_exceptions.values(), key=lambda x: (-x.confidence, x.exception_type)
        )
