"""Test helpers for creating valid test objects."""

from codedocsync.parser import FunctionSignature, ParsedDocstring, ParsedFunction
from codedocsync.parser.models import FunctionParameter, ParameterKind


def create_test_function(
    name: str = "test_func",
    params: list[str] = None,
    return_type: str = "None",
    docstring: str = None,
    file_path: str = "test.py",
    line_number: int = 1,
    end_line_number: int = 10,
    source_code: str = None,
) -> ParsedFunction:
    """Create a valid ParsedFunction for tests."""
    if params is None:
        params = []

    if source_code is None:
        source_code = f"def {name}({', '.join(params)}): pass"

    # Create FunctionParameter objects
    parameters = []
    for param_name in params:
        parameters.append(
            FunctionParameter(
                name=param_name,
                type_annotation=None,
                default_value=None,
                is_required=True,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
            )
        )

    signature = FunctionSignature(
        name=name,
        parameters=parameters,
        return_type=return_type,
        is_method=False,
        is_async=False,
        decorators=[],
    )

    # Create ParsedDocstring if docstring is provided
    parsed_docstring = None
    if docstring:
        parsed_docstring = ParsedDocstring(
            description=docstring,
            parameters=[],
            returns=None,
            raises=[],
            style="google",
        )

    return ParsedFunction(
        signature=signature,
        docstring=parsed_docstring,
        file_path=file_path,
        line_number=line_number,
        end_line_number=end_line_number,
        source_code=source_code,
    )
