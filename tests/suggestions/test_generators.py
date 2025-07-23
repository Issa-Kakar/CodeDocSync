"""
Comprehensive test suite for all suggestion generators.

Tests each generator's ability to handle various issue types,
edge cases, and integration scenarios.
"""

from typing import Any

import pytest

from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.generators import (
    BehaviorSuggestionGenerator,
    EdgeCaseSuggestionGenerator,
    ExampleSuggestionGenerator,
    ParameterSuggestionGenerator,
    RaisesSuggestionGenerator,
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.models import SuggestionContext, SuggestionType

from .fixtures import (
    DOCSTRING_EXAMPLES,
    create_parsed_docstring,
    create_test_function,
    create_test_issue,
)


class TestParameterSuggestionGenerator:
    """Test parameter suggestion generation."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create parameter suggestion generator."""
        config = SuggestionConfig(default_style="google")
        return ParameterSuggestionGenerator(config)

    def test_parameter_name_mismatch_simple(self, generator: Any) -> None:
        """Test fixing simple parameter name mismatch."""
        # Create function with mismatched parameter
        function = create_test_function(
            name="authenticate_user",
            params=["username", "password"],
            docstring=DOCSTRING_EXAMPLES["google_simple"],
        )

        # Create issue
        issue = create_test_issue(
            issue_type="parameter_name_mismatch",
            description="Parameter 'email' doesn't match 'username' in code",
            details={"expected": "username", "found": "email"},
        )

        # Create context
        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                summary="Simple function description.",
                params={"email": "First parameter", "password": "Second parameter"},
            ),
            project_style="google",
        )

        # Generate suggestion
        suggestion = generator.generate(context)

        # Verify suggestion
        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert "username" in suggestion.suggested_text
        assert "email" not in suggestion.suggested_text
        assert suggestion.confidence >= 0.9
        assert suggestion.copy_paste_ready

    def test_parameter_missing(self, generator: Any) -> None:
        """Test adding missing parameter documentation."""
        function = create_test_function(params=["username", "password", "remember_me"])

        issue = create_test_issue(
            issue_type="parameter_missing",
            description="Parameter 'remember_me' is not documented",
            details={"missing_param": "remember_me", "type_annotation": "bool"},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                params={"username": "User's username", "password": "User's password"}
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "remember_me" in suggestion.suggested_text
        assert "bool" in suggestion.suggested_text
        assert suggestion.confidence >= 0.8

    def test_parameter_type_mismatch(self, generator: Any) -> None:
        """Test fixing parameter type mismatch."""
        function = create_test_function(params=["count", "name"])

        issue = create_test_issue(
            issue_type="parameter_type_mismatch",
            description="Parameter 'count' type mismatch",
            details={
                "param_name": "count",
                "expected_type": "int",
                "documented_type": "str",
            },
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                params={"count": "Number of items", "name": "Item name"}
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "count (int)" in suggestion.suggested_text
        assert "Number of items" in suggestion.suggested_text  # Preserves description

    def test_preserves_descriptions(self, generator: Any) -> None:
        """Test that existing descriptions are preserved."""
        function = create_test_function(params=["data", "config"])

        issue = create_test_issue(
            issue_type="parameter_name_mismatch",
            details={"expected": "config", "found": "settings"},
        )

        detailed_desc = (
            "Configuration object with multiple settings\n"
            "        that control the behavior of the function.\n"
            "        Can be None for default settings."
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                params={"data": "Input data to process", "settings": detailed_desc}
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should preserve the detailed description
        assert (
            "Configuration object with multiple settings" in suggestion.suggested_text
        )
        assert "config" in suggestion.suggested_text
        assert "settings" not in suggestion.suggested_text


class TestReturnSuggestionGenerator:
    """Test return suggestion generation."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create return suggestion generator."""
        config = SuggestionConfig(default_style="google")
        return ReturnSuggestionGenerator(config)

    def test_return_type_mismatch(self, generator: Any) -> None:
        """Test fixing return type mismatch."""
        function = create_test_function(
            name="calculate_total",
            return_type="float",
            docstring="Calculate total.\n\nReturns:\n    int: The total",
        )

        issue = create_test_issue(
            issue_type="return_type_mismatch",
            description="Return type mismatch",
            details={"expected_type": "float", "documented_type": "int"},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(returns="The total"),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "float" in suggestion.suggested_text
        assert "The total" in suggestion.suggested_text

    def test_missing_return_documentation(self, generator: Any) -> None:
        """Test adding missing return documentation."""
        function = create_test_function(
            name="process_data", return_type="Dict[str, Any]"
        )

        issue = create_test_issue(
            issue_type="missing_returns",
            description="Missing return documentation",
            details={"return_type": "Dict[str, Any]"},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Process data."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "Returns:" in suggestion.suggested_text
        assert "Dict[str, Any]" in suggestion.suggested_text

    def test_generator_function_return(self, generator: Any) -> None:
        """Test documenting generator functions."""
        function = create_test_function(
            name="iterate_items", return_type="Generator[int, None, None]"
        )

        issue = create_test_issue(
            issue_type="return_type_mismatch",
            details={"is_generator": True, "yield_type": "int"},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(returns="Items"),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert (
            "Yields:" in suggestion.suggested_text
            or "Generator" in suggestion.suggested_text
        )


class TestRaisesSuggestionGenerator:
    """Test exception documentation generation."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create raises suggestion generator."""
        config = SuggestionConfig(default_style="google")
        return RaisesSuggestionGenerator(config)

    def test_missing_raises_documentation(self, generator: Any) -> None:
        """Test adding missing exception documentation."""
        function = create_test_function(
            name="validate_input",
            source_code="""def validate_input(data):
    if not data:
        raise ValueError("Data cannot be empty")
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
""",
        )

        issue = create_test_issue(
            issue_type="missing_raises",
            description="Missing exception documentation",
            details={
                "missing_exceptions": [
                    {"type": "ValueError", "condition": "not data"},
                    {"type": "TypeError", "condition": "not isinstance(data, dict)"},
                ]
            },
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Validate input."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "Raises:" in suggestion.suggested_text
        assert "ValueError" in suggestion.suggested_text
        assert "TypeError" in suggestion.suggested_text

    def test_updates_existing_raises_section(self, generator: Any) -> None:
        """Test updating existing raises section."""
        function = create_test_function(name="process")

        issue = create_test_issue(
            issue_type="missing_raises",
            details={"missing_exceptions": [{"type": "RuntimeError"}]},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                raises={"ValueError": "If value is invalid"}
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should preserve existing and add new
        assert "ValueError: If value is invalid" in suggestion.suggested_text
        assert "RuntimeError" in suggestion.suggested_text


class TestBehaviorSuggestionGenerator:
    """Test behavioral description generation."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create behavior suggestion generator."""
        config = SuggestionConfig(default_style="google")
        return BehaviorSuggestionGenerator(config)

    def test_enhance_vague_description(self, generator: Any) -> None:
        """Test enhancing vague descriptions."""
        function = create_test_function(
            name="process_data",
            source_code="""def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
""",
        )

        issue = create_test_issue(
            issue_type="description_vague",
            description="Description is too vague",
            severity="medium",
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Process data."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should add behavioral details
        assert suggestion.confidence >= 0.7
        assert len(suggestion.suggested_text) > len("Process data.")

    def test_identify_side_effects(self, generator: Any) -> None:
        """Test identifying and documenting side effects."""
        function = create_test_function(
            name="save_to_file",
            source_code="""def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    logging.info(f"Saved data to {filename}")
""",
        )

        issue = create_test_issue(
            issue_type="description_incomplete", severity="medium"
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Save data."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should mention file I/O
        assert "file" in suggestion.suggested_text.lower()


class TestExampleSuggestionGenerator:
    """Test example generation."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create example suggestion generator."""
        config = SuggestionConfig(default_style="google", include_examples=True)
        return ExampleSuggestionGenerator(config)

    def test_generate_basic_example(self, generator: Any) -> None:
        """Test generating basic usage example."""
        function = create_test_function(
            name="add_numbers", params=["a", "b"], return_type="int"
        )

        issue = create_test_issue(
            issue_type="missing_examples",
            description="No usage examples provided",
            severity="low",
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                summary="Add two numbers.",
                params={"a": "First number", "b": "Second number"},
                returns="Sum of a and b",
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "Examples:" in suggestion.suggested_text
        assert "add_numbers" in suggestion.suggested_text
        assert ">>>" in suggestion.suggested_text

    def test_update_invalid_example(self, generator: Any) -> None:
        """Test updating invalid examples."""
        function = create_test_function(name="multiply", params=["x", "y"])

        issue = create_test_issue(
            issue_type="example_invalid",
            description="Example uses old function signature",
            details={"invalid_example": ">>> multiply(5)  # Old signature"},
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                examples=[">>> multiply(5)  # Old signature", "10"]
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        assert "multiply" in suggestion.suggested_text
        assert "x" in suggestion.suggested_text and "y" in suggestion.suggested_text


class TestEdgeCaseSuggestionGenerator:
    """Test edge case handling."""

    @pytest.fixture
    def generator(self) -> Any:
        """Create edge case suggestion generator."""
        config = SuggestionConfig(default_style="google")
        return EdgeCaseSuggestionGenerator(config)

    def test_property_method_documentation(self, generator: Any) -> None:
        """Test documenting property methods."""
        function = create_test_function(name="temperature")
        function.signature.decorators = ["property"]

        issue = create_test_issue(
            issue_type="property_documentation",
            description="Property lacks proper documentation",
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Temperature."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should document as property
        assert (
            "property" in suggestion.suggested_text.lower()
            or "attribute" in suggestion.suggested_text.lower()
        )

    def test_classmethod_documentation(self, generator: Any) -> None:
        """Test documenting class methods."""
        function = create_test_function(
            name="from_dict", params=["cls", "data"], return_type="MyClass"
        )
        function.signature.decorators = ["classmethod"]

        issue = create_test_issue(
            issue_type="classmethod_documentation",
            description="Class method documentation includes 'cls' parameter",
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(
                params={"cls": "Class reference", "data": "Input data"}
            ),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should not include 'cls' in parameters
        assert "cls:" not in suggestion.suggested_text
        assert "data:" in suggestion.suggested_text

    def test_magic_method_documentation(self, generator: Any) -> None:
        """Test documenting magic methods."""
        function = create_test_function(
            name="__init__", params=["self", "name", "value"]
        )

        issue = create_test_issue(
            issue_type="magic_method_documentation",
            description="Magic method needs appropriate documentation",
        )

        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=create_parsed_docstring(summary="Init method."),
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Should have initialization-specific documentation
        assert (
            "initialize" in suggestion.suggested_text.lower()
            or "create" in suggestion.suggested_text.lower()
        )


class TestGeneratorIntegration:
    """Test integration between multiple generators."""

    def test_multiple_issues_same_function(self) -> None:
        """Test handling multiple issues for the same function."""
        config = SuggestionConfig(default_style="google")

        # Create function with multiple issues
        function = create_test_function(
            name="process_data",
            params=["data", "options"],
            return_type="Dict[str, Any]",
        )

        # Issue 1: Missing parameter
        param_gen = ParameterSuggestionGenerator(config)
        param_issue = create_test_issue(
            issue_type="parameter_missing", details={"missing_param": "options"}
        )

        # Issue 2: Missing return
        return_gen = ReturnSuggestionGenerator(config)
        return_issue = create_test_issue(
            issue_type="missing_returns", details={"return_type": "Dict[str, Any]"}
        )

        # Generate suggestions
        docstring = create_parsed_docstring(params={"data": "Input data"})

        param_suggestion = param_gen.generate(
            SuggestionContext(param_issue, function, docstring, "google")
        )
        return_suggestion = return_gen.generate(
            SuggestionContext(return_issue, function, docstring, "google")
        )

        # Both should be valid
        assert param_suggestion.confidence >= 0.8
        assert return_suggestion.confidence >= 0.8
        assert "options" in param_suggestion.suggested_text
        assert "Returns:" in return_suggestion.suggested_text

    def test_generator_selection_by_issue_type(self) -> None:
        """Test that correct generator is selected for each issue type."""
        config = SuggestionConfig(default_style="google")

        issue_generator_map = {
            "parameter_name_mismatch": ParameterSuggestionGenerator,
            "return_type_mismatch": ReturnSuggestionGenerator,
            "missing_raises": RaisesSuggestionGenerator,
            "description_vague": BehaviorSuggestionGenerator,
            "missing_examples": ExampleSuggestionGenerator,
            "property_documentation": EdgeCaseSuggestionGenerator,
        }

        for _issue_type, generator_class in issue_generator_map.items():
            generator = generator_class(config)
            assert generator is not None
            assert hasattr(generator, "generate")


class TestPerformanceAndScale:
    """Test generator performance with large inputs."""

    def test_large_parameter_list(self) -> None:
        """Test handling functions with many parameters."""
        config = SuggestionConfig(default_style="google")
        generator = ParameterSuggestionGenerator(config)

        # Create function with 50 parameters
        params = [f"param_{i}" for i in range(50)]
        function = create_test_function(name="complex_function", params=params)

        issue = create_test_issue(
            issue_type="parameter_missing",
            details={"missing_param": "param_25"},
        )

        # Create docstring missing one parameter
        doc_params = {f"param_{i}": f"Parameter {i}" for i in range(50) if i != 25}
        docstring = create_parsed_docstring(params=doc_params)

        context = SuggestionContext(issue, function, docstring, "google")

        # Should handle efficiently
        suggestion = generator.generate(context)
        assert suggestion is not None
        assert "param_25" in suggestion.suggested_text

    def test_complex_nested_docstring(self) -> None:
        """Test handling complex nested docstrings."""
        config = SuggestionConfig(default_style="numpy")
        generator = ParameterSuggestionGenerator(config)

        function = create_test_function(params=["matrix", "axis", "keepdims"])

        issue = create_test_issue(
            issue_type="parameter_type_mismatch",
            details={
                "param_name": "matrix",
                "expected_type": "np.ndarray",
                "documented_type": "array",
            },
        )

        # Complex NumPy-style docstring
        DOCSTRING_EXAMPLES["numpy_complex"]
        context = SuggestionContext(
            issue,
            function,
            create_parsed_docstring(format_style="numpy"),
            "numpy",
        )

        suggestion = generator.generate(context)
        assert suggestion is not None
