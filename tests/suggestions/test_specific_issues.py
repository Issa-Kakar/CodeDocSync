"""
Test suite for issue-specific suggestion generators.
Tests parameter, return, raises, and other specific issue types
to ensure each generator handles its specific issues correctly.
"""

from textwrap import dedent

from codedocsync.analyzer.models import InconsistencyIssue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.suggestions.generators.behavior_generator import (
    BehaviorSuggestionGenerator,
)
from codedocsync.suggestions.generators.example_generator import (
    ExampleSuggestionGenerator,
)
from codedocsync.suggestions.generators.parameter_generator import (
    ParameterSuggestionGenerator,
)
from codedocsync.suggestions.generators.raises_generator import (
    RaisesSuggestionGenerator,
)
from codedocsync.suggestions.generators.return_generator import (
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext


class TestSpecificIssueFixes:
    """Test fixes for specific issue types."""

    def test_fix_parameter_name_mismatch(self) -> None:
        """Test fixing parameter name mismatches."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="calculate_area",
                parameters=[
                    FunctionParameter(
                        name="width", type_annotation="float", is_required=True
                    ),
                    FunctionParameter(
                        name="height", type_annotation="float", is_required=True
                    ),
                ],
                return_type="float",
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Calculate area of rectangle.
                    Args:
                        w (float): Width of rectangle.
                        h (float): Height of rectangle.
                    Returns:
                        float: Area value.
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="parameter_name_mismatch",
            severity="critical",
            description="Parameter names don't match",
            suggestion="Fix parameter names",
            line_number=4,
            details={
                "mismatches": [
                    {"code": "width", "doc": "w"},
                    {"code": "height", "doc": "h"},
                ]
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "width (float):" in suggestion.suggested_text
        assert "height (float):" in suggestion.suggested_text
        assert "w (float)" not in suggestion.suggested_text
        assert "h (float)" not in suggestion.suggested_text

    def test_fix_return_type_mismatch(self) -> None:
        """Test fixing return type documentation mismatches."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="get_config",
                return_type="Optional[Dict[str, Any]]",
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Get configuration settings.
                    Returns:
                        dict: Configuration dictionary.
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="return_type_mismatch",
            severity="high",
            description="Return type doesn't match documentation",
            suggestion="Update return type",
            line_number=4,
            details={
                "expected": "Optional[Dict[str, Any]]",
                "documented": "dict",
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ReturnSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "Optional[Dict[str, Any]]:" in suggestion.suggested_text
        assert "Configuration dictionary" in suggestion.suggested_text
        assert (
            "None if" in suggestion.suggested_text.lower()
            or "not found" in suggestion.suggested_text.lower()
        )

    def test_fix_missing_raises_complex(self) -> None:
        """Test fixing missing exception documentation with multiple exceptions."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="parse_json_file",
                parameters=[
                    FunctionParameter(
                        name="filepath", type_annotation="str", is_required=True
                    ),
                ],
                return_type="Dict[str, Any]",
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Parse JSON file and return contents.
                    Args:
                        filepath (str): Path to JSON file.
                    Returns:
                        Dict[str, Any]: Parsed JSON data.
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="missing_raises",
            severity="medium",
            description="Exceptions not documented",
            suggestion="Document exceptions",
            line_number=8,
            details={
                "exceptions": [
                    "FileNotFoundError",
                    "JSONDecodeError",
                    "PermissionError",
                ],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = RaisesSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "Raises:" in suggestion.suggested_text
        assert "FileNotFoundError:" in suggestion.suggested_text
        assert "JSONDecodeError:" in suggestion.suggested_text
        assert "PermissionError:" in suggestion.suggested_text
        assert "file" in suggestion.suggested_text.lower()
        assert "json" in suggestion.suggested_text.lower()
        assert "permission" in suggestion.suggested_text.lower()

    def test_fix_parameter_order_different(self) -> None:
        """Test fixing when parameter order in docs doesn't match code."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="create_connection",
                parameters=[
                    FunctionParameter(
                        name="host", type_annotation="str", is_required=True
                    ),
                    FunctionParameter(
                        name="port", type_annotation="int", is_required=True
                    ),
                    FunctionParameter(
                        name="timeout",
                        type_annotation="float",
                        default_value="30.0",
                        is_required=False,
                    ),
                ],
                return_type="Connection",
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Create network connection.
                    Args:
                        port (int): Port number.
                        timeout (float): Timeout in seconds.
                        host (str): Host address.
                    Returns:
                        Connection: Active connection object.
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="parameter_order_different",
            severity="medium",
            description="Parameter order doesn't match code",
            suggestion="Reorder parameters",
            line_number=4,
            details={
                "expected_order": ["host", "port", "timeout"],
                "documented_order": ["port", "timeout", "host"],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        # Check parameters are in correct order
        host_pos = suggestion.suggested_text.find("host (str):")
        port_pos = suggestion.suggested_text.find("port (int):")
        timeout_pos = suggestion.suggested_text.find("timeout (float")
        assert host_pos < port_pos < timeout_pos

    def test_fix_missing_params_with_complex_types(self) -> None:
        """Test fixing missing parameters with complex type annotations."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter(
                        name="data", type_annotation="pd.DataFrame", is_required=True
                    ),
                    FunctionParameter(
                        name="columns",
                        type_annotation="Optional[List[str]]",
                        default_value="None",
                        is_required=False,
                    ),
                    FunctionParameter(
                        name="callback",
                        type_annotation="Callable[[int], None]",
                        default_value="None",
                        is_required=False,
                    ),
                ],
                return_type="pd.DataFrame",
            ),
            docstring=RawDocstring(raw_text='""""""'),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="critical",
            description="Parameters not documented",
            suggestion="Add parameter documentation",
            line_number=2,
            details={
                "missing_params": ["data", "columns", "callback"],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ParameterSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "data (pd.DataFrame):" in suggestion.suggested_text
        assert "columns (Optional[List[str]], optional):" in suggestion.suggested_text
        assert "callback (Callable[[int], None], optional):" in suggestion.suggested_text

    def test_fix_example_invalid(self) -> None:
        """Test fixing invalid examples in docstring."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="square",
                parameters=[
                    FunctionParameter(
                        name="x", type_annotation="float", is_required=True
                    ),
                ],
                return_type="float",
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Calculate square of a number.
                    Args:
                        x (float): Number to square.
                    Returns:
                        float: Square of x.
                    Examples:
                        >>> square(2)
                        5
                        >>> square(-3)
                        -9
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="example_invalid",
            severity="low",
            description="Examples show incorrect output",
            suggestion="Fix examples",
            line_number=10,
            details={
                "invalid_examples": [
                    {"input": "square(2)", "expected": "4", "shown": "5"},
                    {"input": "square(-3)", "expected": "9", "shown": "-9"},
                ],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ExampleSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert ">>> square(2)" in suggestion.suggested_text
        assert "4" in suggestion.suggested_text
        assert ">>> square(-3)" in suggestion.suggested_text
        assert "9" in suggestion.suggested_text
        assert "5" not in suggestion.suggested_text
        assert "-9" not in suggestion.suggested_text

    def test_fix_description_outdated(self) -> None:
        """Test fixing outdated function descriptions."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="save_data",
                parameters=[
                    FunctionParameter(
                        name="data", type_annotation="Dict[str, Any]", is_required=True
                    ),
                    FunctionParameter(
                        name="filepath", type_annotation="Path", is_required=True
                    ),
                    FunctionParameter(
                        name="compress",
                        type_annotation="bool",
                        default_value="False",
                        is_required=False,
                    ),
                ],
                return_type="None",
                is_async=True,
            ),
            docstring=RawDocstring(
                raw_text=dedent(
                    '''
                    """
                    Save data to CSV file.
                    Args:
                        data (dict): Data to save.
                        filepath (str): Output file path.
                    Returns:
                        bool: True if successful.
                    """
                '''
                ).strip(),
                line_number=1,
            ),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="description_outdated",
            severity="medium",
            description="Multiple inconsistencies in documentation",
            suggestion="Update documentation",
            line_number=2,
            details={
                "issues": [
                    "Function is async but not mentioned",
                    "Says CSV but handles dict/JSON data",
                    "Missing 'compress' parameter",
                    "Returns None not bool",
                    "filepath type is Path not str",
                ],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = BehaviorSuggestionGenerator()
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "async" in suggestion.suggested_text.lower()
        assert "compress" in suggestion.suggested_text
        assert "Path" in suggestion.suggested_text
        assert "None" in suggestion.suggested_text or "nothing" in suggestion.suggested_text.lower()
        assert "CSV" not in suggestion.suggested_text or "JSON" in suggestion.suggested_text