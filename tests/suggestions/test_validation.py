"""
Test suite for template syntax validation.
Ensures all generated docstrings have 100% valid syntax
and can be parsed correctly by docstring parsers.
"""

import ast
from typing import Any

from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.parser.docstring_models import DocstringParameter
from codedocsync.parser.docstring_parser import DocstringParser
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
from codedocsync.suggestions.templates.google_template import GoogleStyleTemplate
from codedocsync.suggestions.templates.numpy_template import NumpyStyleTemplate
from codedocsync.suggestions.templates.sphinx_template import SphinxStyleTemplate


class TestTemplateSyntaxValidation:
    """Validate all generated templates produce valid Python docstrings."""

    def validate_python_syntax(self, docstring: str) -> bool:
        """Validate that docstring can be used in Python code."""
        # Create a simple function with the docstring
        code = f'''
def test_func() -> None:
    """{docstring}"""
    pass
'''
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def validate_docstring_parse(self, docstring: str, style: str) -> bool:
        """Validate that docstring can be parsed by docstring parser."""
        parser = DocstringParser()
        try:
            parsed = parser.parse(docstring)
            return parsed is not None
        except Exception:
            return False

    def test_google_style_template_validity(self) -> None:
        """Test Google style templates produce valid syntax."""
        template = GoogleStyleTemplate()
        # Test various parameter configurations
        test_cases = [
            # Simple parameters
            {
                "params": [
                    {"name": "x", "type": "int", "description": "X value"},
                    {"name": "y", "type": "int", "description": "Y value"},
                ]
            },
            # Complex types
            {
                "params": [
                    {
                        "name": "data",
                        "type": "Dict[str, List[int]]",
                        "description": "Complex data structure",
                    },
                    {
                        "name": "callback",
                        "type": "Callable[[int], str]",
                        "description": "Callback function",
                    },
                ]
            },
            # Optional parameters
            {
                "params": [
                    {
                        "name": "required",
                        "type": "str",
                        "description": "Required param",
                    },
                    {
                        "name": "optional",
                        "type": "int",
                        "description": "Optional param",
                        "default": "10",
                    },
                ]
            },
        ]
        for case in test_cases:
            # Convert test case params to DocstringParameter objects
            params = [
                DocstringParameter(
                    name=p["name"],
                    type_str=p.get("type"),
                    description=p.get("description", ""),
                )
                for p in case["params"]
            ]
            lines = template.render_parameters(params)
            docstring = "\n".join(lines)
            assert self.validate_python_syntax(docstring)
            assert self.validate_docstring_parse(docstring, "google")

    def test_numpy_style_template_validity(self) -> None:
        """Test NumPy style templates produce valid syntax."""
        template = NumpyStyleTemplate()
        # Test various configurations
        test_cases = [
            # Basic parameters
            {
                "params": [
                    {
                        "name": "array",
                        "type": "np.ndarray",
                        "description": "Input array",
                    },
                    {
                        "name": "axis",
                        "type": "int",
                        "description": "Axis to operate on",
                    },
                ]
            },
            # With shape information
            {
                "params": [
                    {
                        "name": "matrix",
                        "type": "np.ndarray",
                        "description": "2D matrix\n    Shape: (n, m)",
                    },
                ]
            },
        ]
        for case in test_cases:
            # Convert test case params to DocstringParameter objects
            params = [
                DocstringParameter(
                    name=p["name"],
                    type_str=p.get("type"),
                    description=p.get("description", ""),
                )
                for p in case["params"]
            ]
            lines = template.render_parameters(params)
            docstring = "\n".join(lines)
            assert self.validate_python_syntax(docstring)
            assert self.validate_docstring_parse(docstring, "numpy")

    def test_sphinx_style_template_validity(self) -> None:
        """Test Sphinx style templates produce valid syntax."""
        template = SphinxStyleTemplate()
        test_cases = [
            # Simple types
            {
                "params": [
                    {"name": "path", "type": "str", "description": "File path"},
                    {"name": "mode", "type": "str", "description": "File mode"},
                ]
            },
            # Union types
            {
                "params": [
                    {
                        "name": "value",
                        "type": "Union[int, float]",
                        "description": "Numeric value",
                    },
                ]
            },
        ]
        for case in test_cases:
            # Convert test case params to DocstringParameter objects
            params = [
                DocstringParameter(
                    name=p["name"],
                    type_str=p.get("type"),
                    description=p.get("description", ""),
                )
                for p in case["params"]
            ]
            lines = template.render_parameters(params)
            docstring = "\n".join(lines)
            assert self.validate_python_syntax(docstring)
            assert self.validate_docstring_parse(docstring, "sphinx")

    def test_return_documentation_validity(self) -> None:
        """Test return documentation produces valid syntax."""
        generator = ReturnSuggestionGenerator()
        test_returns = [
            {"type": "int", "description": "Count of items"},
            {"type": "List[str]", "description": "List of names"},
            {"type": "Optional[Dict[str, Any]]", "description": "Data or None"},
            {"type": "Generator[int, None, None]", "description": "Number generator"},
        ]
        for style in ["google", "numpy", "sphinx"]:
            for ret in test_returns:
                func = ParsedFunction(
                    signature=FunctionSignature(
                        name="test_func", return_type=ret["type"]
                    ),
                    docstring=RawDocstring(raw_text='""""""'),
                    file_path="test.py",
                    line_number=1,
                )
                issue = InconsistencyIssue(
                    issue_type="missing_returns",
                    severity="high",
                    description="Missing return",
                    suggestion="Add return",
                    line_number=1,
                    details={"return_type": ret["type"]},
                )
                context = SuggestionContext(
                    issue=issue, function=func, project_style=style
                )
                suggestion = generator.generate(context)
                assert suggestion is not None
                assert self.validate_python_syntax(suggestion.suggested_text)

    def test_raises_documentation_validity(self) -> None:
        """Test exception documentation produces valid syntax."""
        generator = RaisesSuggestionGenerator()
        exceptions = [
            ["ValueError", "TypeError"],
            ["FileNotFoundError", "PermissionError", "OSError"],
            ["CustomError"],
        ]
        for style in ["google", "numpy", "sphinx"]:
            for exc_list in exceptions:
                func = ParsedFunction(
                    signature=FunctionSignature(name="test_func"),
                    docstring=RawDocstring(raw_text='""""""'),
                    file_path="test.py",
                    line_number=1,
                )
                issue = InconsistencyIssue(
                    issue_type="missing_raises",
                    severity="medium",
                    description="Missing raises",
                    suggestion="Add raises",
                    line_number=1,
                    details={"exceptions": exc_list},
                )
                context = SuggestionContext(
                    issue=issue, function=func, project_style=style
                )
                suggestion = generator.generate(context)
                assert suggestion is not None
                assert self.validate_python_syntax(suggestion.suggested_text)

    def test_complete_docstring_validity(self) -> None:
        """Test complete docstrings with all sections are valid."""
        # Create a complex function
        func = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="List[Dict[str, Any]]",
                        is_required=True,
                    ),
                    FunctionParameter(
                        name="validate",
                        type_annotation="bool",
                        default_value="True",
                        is_required=False,
                    ),
                ],
                return_type="ProcessedResult",
            ),
            docstring=RawDocstring(raw_text='""""""'),
            file_path="test.py",
            line_number=1,
        )
        # Generate suggestions for all missing parts
        generators = {
            "params": ParameterSuggestionGenerator(),
            "returns": ReturnSuggestionGenerator(),
            "raises": RaisesSuggestionGenerator(),
        }
        issues = {
            "params": InconsistencyIssue(
                issue_type="missing_params",
                severity="critical",
                description="Missing params",
                suggestion="Add params",
                line_number=1,
                details={"missing_params": ["data", "validate"]},
            ),
            "returns": InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="Missing returns",
                suggestion="Add returns",
                line_number=1,
                details={"return_type": "ProcessedResult"},
            ),
            "raises": InconsistencyIssue(
                issue_type="missing_raises",
                severity="medium",
                description="Missing raises",
                suggestion="Add raises",
                line_number=1,
                details={"exceptions": ["ValueError", "ProcessingError"]},
            ),
        }
        for style in ["google", "numpy", "sphinx"]:
            # Combine all suggestions
            combined = '"""Process data with validation.\n\n'
            for issue_type, issue in issues.items():
                context = SuggestionContext(
                    issue=issue, function=func, project_style=style
                )
                suggestion = generators[issue_type].generate(context)
                if suggestion:
                    combined += suggestion.suggested_text + "\n\n"
            combined += '"""'
            # Validate complete docstring
            assert self.validate_python_syntax(combined)

    def test_edge_case_validity(self) -> None:
        """Test edge cases produce valid syntax."""
        edge_cases: list[dict[str, Any]] = [
            # Empty parameter list
            {
                "params": [],
                "returns": None,
                "raises": [],
            },
            # Very long type annotations
            {
                "params": [
                    {
                        "name": "complex_param",
                        "type": "Dict[str, List[Tuple[int, str, Dict[str, Any]]]]",
                        "description": "Very complex nested type",
                    }
                ],
                "returns": "Tuple[bool, Dict[str, List[int]], Optional[str]]",
                "raises": ["Exception"],
            },
            # Special characters in descriptions
            {
                "params": [
                    {
                        "name": "pattern",
                        "type": "str",
                        "description": "Pattern with special chars: \"quotes\", 'single', \\backslash",
                    }
                ],
                "returns": "bool",
                "raises": [],
            },
        ]
        for style in ["google", "numpy", "sphinx"]:
            for case in edge_cases:
                # Build a complete docstring
                if style == "google":
                    docstring = '"""Test function.\n\n'
                    if case["params"]:
                        docstring += "Args:\n"
                        for p in case["params"]:
                            docstring += (
                                f"    {p['name']} ({p['type']}): {p['description']}\n"
                            )
                    if case["returns"]:
                        docstring += f"\nReturns:\n    {case['returns']}: Result.\n"
                    if case["raises"]:
                        docstring += "\nRaises:\n"
                        for exc in case["raises"]:
                            docstring += f"    {exc}: Error.\n"
                    docstring += '"""'
                    assert self.validate_python_syntax(docstring)

    def test_multiline_descriptions_validity(self) -> None:
        """Test multiline descriptions produce valid syntax."""
        func = ParsedFunction(
            signature=FunctionSignature(
                name="complex_function",
                parameters=[
                    FunctionParameter(
                        name="config",
                        type_annotation="Dict[str, Any]",
                        is_required=True,
                    ),
                ],
            ),
            docstring=RawDocstring(raw_text='""""""'),
            file_path="test.py",
            line_number=1,
        )
        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="critical",
            description="Missing params",
            suggestion="Add params",
            line_number=1,
            details={
                "missing_params": ["config"],
                "param_descriptions": {
                    "config": """Configuration dictionary with the following keys:
        - 'timeout': Request timeout in seconds
        - 'retries': Number of retry attempts
        - 'verbose': Enable verbose logging"""
                },
            },
        )
        for style in ["google", "numpy", "sphinx"]:
            context = SuggestionContext(issue=issue, function=func, project_style=style)
            generator = ParameterSuggestionGenerator()
            suggestion = generator.generate(context)
            assert suggestion is not None
            assert self.validate_python_syntax(suggestion.suggested_text)
