"""
Comprehensive tests for suggestion generator functionality.

Tests all docstring styles, smart updates, and performance benchmarks.
"""

import time

import pytest
from docstring_parser import parse
from docstring_parser.google import GoogleParser
from docstring_parser.numpydoc import NumpydocParser

from codedocsync.analyzer.models import AnalysisResult, InconsistencyIssue
from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser import ParsedDocstring
from codedocsync.parser.docstring_models import (
from typing import Any, Callable, Dict, List, Optional, Tuple
    DocstringFormat,
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
)
from codedocsync.parser.models import (
    FunctionParameter,
    FunctionSignature,
    ParameterKind,
    ParsedFunction,
)
from codedocsync.suggestions import (
    SuggestionContext,
    SuggestionType,
    enhance_with_suggestions,
)
from codedocsync.suggestions.generators import (
    ParameterSuggestionGenerator,
    RaisesSuggestionGenerator,
    ReturnSuggestionGenerator,
)
from codedocsync.suggestions.templates import (
    GoogleStyleTemplate,
    NumpyStyleTemplate,
    SphinxStyleTemplate,
)


class TestDocstringStyleGeneration:
    """Test generation of different docstring styles."""

    def create_test_function(self) -> ParsedFunction:
        """Create a test function with parameters, return, and raises."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="calculate_average",
                parameters=[
                    FunctionParameter(
                        name="numbers",
                        type_annotation="List[float]",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                    FunctionParameter(
                        name="weights",
                        type_annotation="Optional[List[float]]",
                        default_value="None",
                        is_required=False,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="float",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=10,
        )

    def test_generate_google_style_docstring(self) -> None:
        """Test Google style docstring generation."""
        template = GoogleStyleTemplate()

        # Create proper docstring components
        parameters = [
            DocstringParameter(
                name="numbers",
                type_str="List[float]",
                description="List of numbers to average",
                is_optional=False,
            ),
            DocstringParameter(
                name="weights",
                type_str="Optional[List[float]]",
                description="Optional weights for each number",
                is_optional=True,
            ),
        ]

        returns = DocstringReturns(
            type_str="float",
            description="The weighted average",
        )

        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description="If lists have different lengths",
            )
        ]

        # Generate full docstring
        docstring = template.render_complete_docstring(
            summary="Calculate weighted average of numbers.",
            parameters=parameters,
            returns=returns,
            raises=raises,
        )

        # Verify structure
        assert "Calculate weighted average of numbers." in docstring
        assert "Args:" in docstring
        assert "numbers (List[float]): List of numbers to average" in docstring
        assert (
            "weights (Optional[List[float]]): Optional weights for each number"
            in docstring
        )
        assert "Returns:" in docstring
        assert "float: The weighted average" in docstring
        assert "Raises:" in docstring
        assert "ValueError: If lists have different lengths" in docstring

        # Verify it's valid Google style
        parsed = GoogleParser().parse(docstring)
        assert parsed.short_description == "Calculate weighted average of numbers."
        assert len(parsed.params) == 2
        assert parsed.returns is not None
        assert len(parsed.raises) == 1

    def test_generate_numpy_style_docstring(self) -> None:
        """Test NumPy style docstring generation."""
        template = NumpyStyleTemplate()

        # Create proper docstring components
        parameters = [
            DocstringParameter(
                name="numbers",
                type_str="List[float]",
                description="List of numbers to average",
                is_optional=False,
            ),
            DocstringParameter(
                name="weights",
                type_str="Optional[List[float]]",
                description="Optional weights for each number",
                is_optional=True,
            ),
        ]

        returns = DocstringReturns(
            type_str="float",
            description="The weighted average",
        )

        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description="If lists have different lengths",
            )
        ]

        # Generate full docstring
        docstring = template.render_complete_docstring(
            summary="Calculate weighted average of numbers.",
            parameters=parameters,
            returns=returns,
            raises=raises,
        )

        # Verify structure
        assert "Calculate weighted average of numbers." in docstring
        assert "Parameters" in docstring
        assert "----------" in docstring
        assert "numbers : List[float]" in docstring
        assert "List of numbers to average" in docstring
        assert "weights : Optional[List[float]], optional" in docstring
        assert "Returns" in docstring
        assert "-------" in docstring
        assert "float" in docstring
        assert "The weighted average" in docstring
        assert "Raises" in docstring
        assert "ValueError" in docstring

        # Verify it's valid NumPy style
        parsed = NumpydocParser().parse(docstring)
        assert parsed.short_description == "Calculate weighted average of numbers."
        assert len(parsed.params) == 2
        assert parsed.returns is not None
        assert len(parsed.raises) == 1

    def test_generate_sphinx_style_docstring(self) -> None:
        """Test Sphinx style docstring generation."""
        template = SphinxStyleTemplate()

        # Create proper docstring components
        parameters = [
            DocstringParameter(
                name="numbers",
                type_str="List[float]",
                description="List of numbers to average",
                is_optional=False,
            ),
            DocstringParameter(
                name="weights",
                type_str="Optional[List[float]]",
                description="Optional weights for each number",
                is_optional=True,
            ),
        ]

        returns = DocstringReturns(
            type_str="float",
            description="The weighted average",
        )

        raises = [
            DocstringRaises(
                exception_type="ValueError",
                description="If lists have different lengths",
            )
        ]

        # Generate full docstring
        docstring = template.render_complete_docstring(
            summary="Calculate weighted average of numbers.",
            parameters=parameters,
            returns=returns,
            raises=raises,
        )

        # Verify structure
        assert "Calculate weighted average of numbers." in docstring
        assert ":param numbers:" in docstring
        assert "List of numbers to average" in docstring
        assert ":type numbers: List[float]" in docstring
        assert ":param weights:" in docstring
        assert ":type weights: Optional[List[float]]" in docstring
        assert ":returns:" in docstring
        assert "The weighted average" in docstring
        assert ":rtype: float" in docstring
        assert ":raises ValueError:" in docstring

        # Verify it's valid Sphinx style
        parsed = parse(docstring)
        assert parsed.short_description == "Calculate weighted average of numbers."
        assert len(parsed.params) == 2
        assert parsed.returns is not None
        assert len(parsed.raises) == 1


class TestSmartUpdates:
    """Test smart update features for preserving and merging content."""

    def test_preserve_existing_content(self) -> None:
        """Test that existing docstring content is preserved during updates."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="process_data",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="Dict[str, Any]",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="Dict[str, Any]",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=20,
        )

        # Existing docstring with some content
        existing_docstring = '''"""
        Process data with advanced algorithms.

        This function applies various transformations to the input data
        including normalization, validation, and enrichment.

        Args:
            data: Input data dictionary

        Returns:
            Processed data

        Examples:
            >>> result = process_data({'key': 'value'})
            >>> print(result)
            {'key': 'VALUE', 'processed': True}
        """'''

        # Parse existing docstring
        parsed_existing_raw = parse(existing_docstring.strip().strip('"').strip())
        # Convert to ParsedDocstring
        parsed_existing = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary=parsed_existing_raw.short_description or "",
            description=parsed_existing_raw.long_description,
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
        )

        # Create issue for missing parameter type
        issue = InconsistencyIssue(
            issue_type="parameter_type_missing",
            severity="medium",
            description="Parameter 'data' is missing type annotation in docstring",
            suggestion="Add type annotation for parameter 'data'",
            line_number=25,
        )

        # Generate suggestion using parameter generator
        generator = ParameterSuggestionGenerator()
        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=parsed_existing,
            project_style="google",
        )

        suggestion = generator.generate(context)

        # Verify existing content is preserved
        assert "Process data with advanced algorithms" in suggestion.suggested_text
        assert (
            "This function applies various transformations" in suggestion.suggested_text
        )
        assert "Examples:" in suggestion.suggested_text
        assert (
            ">>> result = process_data({'key': 'value'})" in suggestion.suggested_text
        )

        # Verify new content is added
        assert "Dict[str, Any]" in suggestion.suggested_text

    def test_merge_partial_updates(self) -> None:
        """Test merging partial docstring updates."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="validate_input",
                parameters=[
                    FunctionParameter(
                        name="value",
                        type_annotation="str",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                    FunctionParameter(
                        name="strict",
                        type_annotation="bool",
                        default_value="False",
                        is_required=False,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="bool",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=30,
        )

        # Existing partial docstring
        existing_docstring = '''"""
        Validate input value.

        Args:
            value: The value to validate
        """'''

        parsed_existing_raw = parse(existing_docstring.strip().strip('"').strip())
        # Convert to ParsedDocstring
        parsed_existing = ParsedDocstring(
            format=DocstringFormat.GOOGLE,
            summary=parsed_existing_raw.short_description or "",
            description=parsed_existing_raw.long_description,
            parameters=[],
            returns=None,
            raises=[],
            examples=[],
        )

        # Multiple issues to fix
        issues = [
            InconsistencyIssue(
                issue_type="parameter_missing",
                severity="high",
                description="Parameter 'strict' is missing from docstring",
                suggestion="Add parameter 'strict' to docstring",
                line_number=35,
            ),
            InconsistencyIssue(
                issue_type="return_missing",
                severity="high",
                description="Return documentation is missing",
                suggestion="Add return documentation",
                line_number=36,
            ),
        ]

        # Create analysis result
        matched_pair = MatchedPair(
            function=function,
            docstring=parsed_existing,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,
                location_score=1.0,
                signature_similarity=0.8,
            ),
            match_type=MatchType.EXACT,
            match_reason="Same file match",
        )

        analysis_result = AnalysisResult(
            matched_pair=matched_pair,
            issues=issues,
            used_llm=False,
            analysis_time_ms=10.0,
        )

        # Enhance with suggestions
        enhanced_result = enhance_with_suggestions(analysis_result)

        # Verify all issues have suggestions
        assert len(enhanced_result.issues) == 2
        assert all(
            issue.rich_suggestion is not None for issue in enhanced_result.issues
        )

        # Check that suggestions can be merged
        param_suggestion = next(
            issue.rich_suggestion
            for issue in enhanced_result.issues
            if issue.issue_type == "parameter_missing"
        )
        return_suggestion = next(
            issue.rich_suggestion
            for issue in enhanced_result.issues
            if issue.issue_type == "return_missing"
        )

        assert param_suggestion is not None
        assert return_suggestion is not None
        assert "strict" in param_suggestion.suggested_text
        assert (
            "Returns:" in return_suggestion.suggested_text
            or "bool" in return_suggestion.suggested_text
        )

    def test_fix_specific_issues(self) -> None:
        """Test fixing specific issues: parameter, return, and raises."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="divide",
                parameters=[
                    FunctionParameter(
                        name="a",
                        type_annotation="float",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                    FunctionParameter(
                        name="b",
                        type_annotation="float",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="float",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=40,
        )

        # Test parameter issue
        param_issue = InconsistencyIssue(
            issue_type="parameter_type_mismatch",
            severity="high",
            description="Parameter 'a' type mismatch",
            suggestion="Update parameter type",
            line_number=45,
            details={
                "parameter": "a",
                "expected_type": "float",
                "documented_type": "int",
            },
        )

        param_generator = ParameterSuggestionGenerator()
        param_context = SuggestionContext(
            issue=param_issue,
            function=function,
            docstring=None,
            project_style="google",
        )
        param_suggestion = param_generator.generate(param_context)

        assert param_suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert "float" in param_suggestion.suggested_text
        assert param_suggestion.confidence >= 0.8

        # Test return issue
        return_issue = InconsistencyIssue(
            issue_type="return_type_mismatch",
            severity="high",
            description="Return type mismatch",
            suggestion="Update return type",
            line_number=46,
            details={"expected_type": "float", "documented_type": "int"},
        )

        return_generator = ReturnSuggestionGenerator()
        return_context = SuggestionContext(
            issue=return_issue,
            function=function,
            docstring=None,
            project_style="google",
        )
        return_suggestion = return_generator.generate(return_context)

        assert return_suggestion.suggestion_type == SuggestionType.RETURN_UPDATE
        assert "float" in return_suggestion.suggested_text
        assert return_suggestion.confidence >= 0.8

        # Test raises issue
        raises_issue = InconsistencyIssue(
            issue_type="raises_missing",
            severity="medium",
            description="Missing exception documentation",
            suggestion="Document ZeroDivisionError",
            line_number=47,
            details={"exception_type": "ZeroDivisionError"},
        )

        raises_generator = RaisesSuggestionGenerator()
        raises_context = SuggestionContext(
            issue=raises_issue,
            function=function,
            docstring=None,
            project_style="google",
        )
        raises_suggestion = raises_generator.generate(raises_context)

        assert raises_suggestion.suggestion_type == SuggestionType.RAISES_UPDATE
        assert "ZeroDivisionError" in raises_suggestion.suggested_text
        assert raises_suggestion.confidence >= 0.7


class TestPerformanceBenchmarks:
    """Test performance benchmarks for suggestion generation."""

    def test_suggestion_generation_performance(self) -> None:
        """Test that suggestion generation is < 100ms."""
        function = ParsedFunction(
            signature=FunctionSignature(
                name="test_func",
                parameters=[
                    FunctionParameter(
                        name="param1",
                        type_annotation="str",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="str",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=50,
        )

        issue = InconsistencyIssue(
            issue_type="parameter_missing",
            severity="high",
            description="Parameter missing from docstring",
            suggestion="Add parameter",
            line_number=55,
        )

        generator = ParameterSuggestionGenerator()
        context = SuggestionContext(
            issue=issue,
            function=function,
            docstring=None,
            project_style="google",
        )

        # Measure generation time
        start_time = time.time()
        suggestion = generator.generate(context)
        end_time = time.time()

        generation_time_ms = (end_time - start_time) * 1000

        assert generation_time_ms < 100, f"Generation took {generation_time_ms:.2f}ms"
        assert suggestion is not None

    def test_template_accuracy(self) -> None:
        """Test that all templates produce 100% valid syntax."""
        templates = [
            GoogleStyleTemplate(),
            NumpyStyleTemplate(),
            SphinxStyleTemplate(),
        ]

        parsers = {
            "google": GoogleParser(),
            "numpy": NumpydocParser(),
            "sphinx": parse,  # Sphinx uses general parser
        }

        ParsedFunction(
            signature=FunctionSignature(
                name="complex_function",
                parameters=[
                    FunctionParameter(
                        name="required_param",
                        type_annotation="str",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                    FunctionParameter(
                        name="optional_param",
                        type_annotation="int",
                        default_value="0",
                        is_required=False,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="Tuple[str, int]",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=60,
        )

        for template in templates:
            # Generate comprehensive docstring
            parameters = [
                DocstringParameter(
                    name="required_param",
                    type_str="str",
                    description="A required string parameter",
                    is_optional=False,
                ),
                DocstringParameter(
                    name="optional_param",
                    type_str="int",
                    description="An optional integer parameter",
                    is_optional=True,
                ),
            ]

            returns = DocstringReturns(
                type_str="Tuple[str, int]",
                description="A tuple containing the processed string and count",
            )

            raises = [
                DocstringRaises(
                    exception_type="ValueError",
                    description="If required_param is empty",
                ),
                DocstringRaises(
                    exception_type="TypeError",
                    description="If parameters have wrong types",
                ),
            ]

            docstring = template.render_complete_docstring(
                summary="A complex function with multiple features.",
                parameters=parameters,
                returns=returns,
                raises=raises,
            )

            # Verify docstring is not empty
            assert docstring.strip()

            # Get the appropriate parser
            style_name = template.__class__.__name__.lower().replace(
                "docstringtemplate", ""
            )
            parser = parsers.get(style_name)

            if parser:
                try:
                    # Parse the generated docstring
                    # Handle different parser types
                    if hasattr(parser, "parse"):
                        parsed = parser.parse(docstring)
                    elif callable(parser):
                        parsed = parser(docstring)
                    else:
                        pytest.fail(
                            f"Parser {parser} is not callable and has no parse method"
                        )

                    # Verify basic structure
                    assert parsed.short_description is not None
                    assert len(parsed.params) >= 2  # Should have both parameters
                    assert parsed.returns is not None
                    assert len(parsed.raises) >= 1  # Should have at least one exception

                except Exception as e:
                    pytest.fail(
                        f"Template {template.__class__.__name__} produced invalid syntax: {e}"
                    )


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of suggestion generation."""

    def test_empty_function_docstring_generation(self) -> None:
        """Test generating docstring for function with no parameters or return."""
        ParsedFunction(
            signature=FunctionSignature(
                name="do_nothing",
                parameters=[],
                return_type="None",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=70,
        )

        template = GoogleStyleTemplate()
        docstring = template.render_complete_docstring(
            summary="Does nothing.",
            parameters=[],
            returns=None,
            raises=[],
        )

        assert "Does nothing." in docstring
        # Should not have Args, Returns, or Raises sections
        assert "Args:" not in docstring
        assert "Returns:" not in docstring
        assert "Raises:" not in docstring

    def test_complex_type_annotations(self) -> None:
        """Test handling of complex type annotations."""
        ParsedFunction(
            signature=FunctionSignature(
                name="process_complex_types",
                parameters=[
                    FunctionParameter(
                        name="data",
                        type_annotation="Dict[str, List[Tuple[int, str]]]",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                    FunctionParameter(
                        name="callback",
                        type_annotation="Callable[[str], Awaitable[None]]",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="AsyncIterator[Dict[str, Any]]",
                decorators=[],
                is_async=True,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=80,
        )

        template = GoogleStyleTemplate()
        parameters = [
            DocstringParameter(
                name="data",
                type_str="Dict[str, List[Tuple[int, str]]]",
                description="Nested dictionary with tuple lists",
                is_optional=False,
            ),
            DocstringParameter(
                name="callback",
                type_str="Callable[[str], Awaitable[None]]",
                description="Async callback function",
                is_optional=False,
            ),
        ]

        returns = DocstringReturns(
            type_str="AsyncIterator[Dict[str, Any]]",
            description="Async iterator of processed dictionaries",
        )

        docstring = template.render_complete_docstring(
            summary="Process complex data structures asynchronously.",
            parameters=parameters,
            returns=returns,
            raises=[],
        )

        # Verify complex types are preserved correctly
        assert "Dict[str, List[Tuple[int, str]]]" in docstring
        assert "Callable[[str], Awaitable[None]]" in docstring
        assert "AsyncIterator[Dict[str, Any]]" in docstring

    def test_multiline_descriptions(self) -> None:
        """Test handling of multiline descriptions."""
        ParsedFunction(
            signature=FunctionSignature(
                name="complex_algorithm",
                parameters=[
                    FunctionParameter(
                        name="input_data",
                        type_annotation="np.ndarray",
                        default_value=None,
                        is_required=True,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    ),
                ],
                return_type="np.ndarray",
                decorators=[],
                is_async=False,
                is_method=False,
                is_classmethod=False,
                is_staticmethod=False,
            ),
            docstring=None,
            file_path="test.py",
            line_number=90,
        )

        multiline_desc = """Apply complex algorithm to input data.

        This algorithm implements a sophisticated transformation
        that involves multiple steps:
        1. Normalization
        2. Feature extraction
        3. Dimensionality reduction
        4. Final transformation"""

        template = NumpyStyleTemplate()
        parameters = [
            DocstringParameter(
                name="input_data",
                type_str="np.ndarray",
                description="""Input array to process.
                Must be 2-dimensional with shape (n_samples, n_features)""",
                is_optional=False,
            ),
        ]

        returns = DocstringReturns(
            type_str="np.ndarray",
            description="""Transformed array.
            Has the same shape as input but with transformed values""",
        )

        docstring = template.render_complete_docstring(
            summary=multiline_desc.split("\n")[0],  # First line as summary
            description="\n".join(
                multiline_desc.split("\n")[1:]
            ).strip(),  # Rest as description
            parameters=parameters,
            returns=returns,
            raises=[],
        )

        # Verify multiline content is preserved
        assert "Apply complex algorithm to input data." in docstring
        assert "1. Normalization" in docstring
        assert "Must be 2-dimensional" in docstring
        assert "Has the same shape" in docstring