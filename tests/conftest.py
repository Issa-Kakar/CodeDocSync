"""Shared test fixtures and helpers."""

import pytest

from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)


def create_test_function(
    name: str = "test_func",
    params: list[FunctionParameter] | None = None,
    docstring: RawDocstring | None = None,
    line_number: int = 1,
    source_code: str | None = None,
) -> ParsedFunction:
    """Helper to create valid ParsedFunction for tests."""
    if params is None:
        params = []

    signature = FunctionSignature(
        name=name,
        parameters=params,
        return_type=None,
        is_async=False,
        is_method=False,
        decorators=[],
    )

    if source_code is None:
        # Generate source code from signature
        param_str = ", ".join(p.name for p in params)
        source_code = f"def {name}({param_str}):\n    pass"

    return ParsedFunction(
        signature=signature,
        docstring=docstring,
        file_path="test.py",
        line_number=line_number,
        end_line_number=line_number + source_code.count("\n") + 1,
        source_code=source_code,
    )


# Make it available to all tests
pytest.create_test_function = create_test_function
