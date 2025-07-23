"""
Performance benchmarks for suggestion generation.
Ensures suggestion generation meets performance requirements:
- Suggestion generation: < 100ms
- Batch processing: efficient scaling
"""

import time
from typing import Any

import pytest

from codedocsync.analyzer.models import AnalysisResult, InconsistencyIssue
from codedocsync.matcher import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser.ast_parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from codedocsync.suggestions.config import get_minimal_config
from codedocsync.suggestions.generators.parameter_generator import (
    ParameterSuggestionGenerator,
)
from codedocsync.suggestions.integration import (
    enhance_multiple_with_suggestions,
    enhance_with_suggestions,
)
from codedocsync.suggestions.models import SuggestionContext


class TestSuggestionPerformance:
    """Performance benchmarks for suggestion generation."""

    def create_test_function(self, num_params: int = 5) -> ParsedFunction:
        """Create a test function with specified number of parameters."""
        params = []
        for i in range(num_params):
            params.append(
                FunctionParameter(
                    name=f"param{i}", type_annotation=f"type{i}", is_required=True
                )
            )
        return ParsedFunction(
            signature=FunctionSignature(
                name="test_function", parameters=params, return_type="ReturnType"
            ),
            docstring=RawDocstring(raw_text='""""""'),
            file_path="test.py",
            line_number=1,
        )

    def test_single_suggestion_performance(self) -> None:
        """Test that single suggestion generation is under 100ms."""
        func = self.create_test_function(num_params=10)
        matched_pair = MatchedPair(
            function=func,
            match_type=MatchType.EXACT,
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=0.9,
                location_score=0.9,
                signature_similarity=0.9,
            ),
            match_reason="Direct match",
            docstring=None,
        )
        issues = [
            InconsistencyIssue(
                issue_type="missing_params",
                severity="critical",
                description="Missing parameters",
                suggestion="Add parameters",
                line_number=2,
                details={
                    "missing_params": [f"param{i}" for i in range(10)],
                },
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="Missing return documentation",
                suggestion="Add return documentation",
                line_number=2,
                details={"return_type": "ReturnType"},
            ),
        ]
        analysis_result = AnalysisResult(
            matched_pair=matched_pair, issues=issues, analysis_time_ms=5.0
        )
        # Use minimal config for performance
        config = get_minimal_config()
        # Measure performance
        start_time = time.perf_counter()
        enhanced = enhance_with_suggestions(analysis_result, config)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        # Verify performance
        assert (
            elapsed_ms < 100
        ), f"Suggestion generation took {elapsed_ms:.2f}ms (> 100ms)"
        assert enhanced is not None
        assert len(enhanced.issues) == 2
        assert all(issue.rich_suggestion is not None for issue in enhanced.issues)

    def test_batch_suggestion_performance(self) -> None:
        """Test performance with multiple analysis results."""
        results = []
        # Create 20 analysis results
        for _ in range(20):
            func = self.create_test_function(num_params=5)
            matched_pair = MatchedPair(
                function=func,
                match_type=MatchType.EXACT,
                confidence=MatchConfidence(
                    overall=0.9,
                    name_similarity=0.9,
                    location_score=0.9,
                    signature_similarity=0.9,
                ),
                match_reason="Direct match",
                docstring=None,
            )
            issues = [
                InconsistencyIssue(
                    issue_type="missing_params",
                    severity="critical",
                    description="Missing parameters",
                    suggestion="Add parameters",
                    line_number=2,
                    details={
                        "missing_params": [f"param{j}" for j in range(5)],
                    },
                )
            ]
            results.append(
                AnalysisResult(
                    matched_pair=matched_pair, issues=issues, analysis_time_ms=3.0
                )
            )
        config = get_minimal_config()
        # Measure batch performance
        start_time = time.perf_counter()
        enhanced_results = enhance_multiple_with_suggestions(results, config)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        avg_ms_per_result = elapsed_ms / len(results)
        # Verify performance
        assert (
            avg_ms_per_result < 50
        ), f"Average time per result: {avg_ms_per_result:.2f}ms (> 50ms)"
        assert len(enhanced_results) == 20
        assert all(
            result.issues[0].rich_suggestion is not None for result in enhanced_results
        )

    def test_generator_direct_performance(self) -> None:
        """Test direct generator performance without integration layer."""
        func = self.create_test_function(num_params=15)
        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="critical",
            description="Missing parameters",
            suggestion="Add parameters",
            line_number=2,
            details={
                "missing_params": [f"param{i}" for i in range(15)],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style="google")
        generator = ParameterSuggestionGenerator()
        # Measure direct generation
        start_time = time.perf_counter()
        suggestion = generator.generate(context)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        # Direct generation should be very fast
        assert elapsed_ms < 50, f"Direct generation took {elapsed_ms:.2f}ms (> 50ms)"
        assert suggestion is not None
        assert suggestion.is_actionable

    def test_complex_function_performance(self) -> None:
        """Test performance with complex function signatures."""
        # Create complex function with many parameters and complex types
        params = [
            FunctionParameter(
                name="data",
                type_annotation="Dict[str, List[Tuple[int, str]]]",
                is_required=True,
            ),
            FunctionParameter(
                name="processor",
                type_annotation="Callable[[Any], Tuple[bool, Dict[str, Any]]]",
                is_required=True,
            ),
            FunctionParameter(
                name="options",
                type_annotation="Dict[str, Union[str, int, float, List[str]]]",
                default_value="None",
                is_required=False,
            ),
            FunctionParameter(name="*args", type_annotation="Any", is_required=False),
            FunctionParameter(
                name="**kwargs", type_annotation="Any", is_required=False
            ),
        ]
        func = ParsedFunction(
            signature=FunctionSignature(
                name="complex_processor",
                parameters=params,
                return_type="Generator[Dict[str, Any], None, None]",
                is_async=True,
            ),
            docstring=RawDocstring(raw_text='""""""'),
            file_path="test.py",
            line_number=1,
        )
        matched_pair = MatchedPair(
            function=func,
            match_type=MatchType.EXACT,
            confidence=MatchConfidence(
                overall=0.85,
                name_similarity=0.85,
                location_score=0.85,
                signature_similarity=0.85,
            ),
            match_reason="Direct match",
            docstring=None,
        )
        issues = [
            InconsistencyIssue(
                issue_type="missing_params",
                severity="critical",
                description="Complex parameters not documented",
                suggestion="Add parameter documentation",
                line_number=2,
                details={
                    "missing_params": [
                        "data",
                        "processor",
                        "options",
                        "*args",
                        "**kwargs",
                    ],
                },
            ),
            InconsistencyIssue(
                issue_type="missing_returns",
                severity="high",
                description="Complex return type not documented",
                suggestion="Add return documentation",
                line_number=2,
                details={
                    "return_type": "Generator[Dict[str, Any], None, None]",
                },
            ),
            InconsistencyIssue(
                issue_type="description_outdated",
                severity="medium",
                description="Description doesn't mention async nature",
                suggestion="Update description",
                line_number=1,
                details={"is_async": True},
            ),
        ]
        analysis_result = AnalysisResult(
            matched_pair=matched_pair, issues=issues, analysis_time_ms=10.0
        )
        config = get_minimal_config()
        # Measure performance with complex function
        start_time = time.perf_counter()
        enhanced = enhance_with_suggestions(analysis_result, config)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        # Even complex functions should be processed quickly
        assert elapsed_ms < 100, f"Complex function took {elapsed_ms:.2f}ms (> 100ms)"
        assert enhanced is not None
        assert len(enhanced.issues) == 3
        assert all(issue.rich_suggestion is not None for issue in enhanced.issues)

    @pytest.mark.parametrize("style", ["google", "numpy", "sphinx"])
    def test_style_generation_performance(self, style: Any) -> None:
        """Test performance across different docstring styles."""
        func = self.create_test_function(num_params=8)
        issue = InconsistencyIssue(
            issue_type="missing_params",
            severity="critical",
            description="Missing parameters",
            suggestion="Add parameters",
            line_number=2,
            details={
                "missing_params": [f"param{i}" for i in range(8)],
            },
        )
        context = SuggestionContext(issue=issue, function=func, project_style=style)
        generator = ParameterSuggestionGenerator()
        # Measure performance for each style
        start_time = time.perf_counter()
        suggestion = generator.generate(context)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        # All styles should have similar performance
        assert elapsed_ms < 50, f"{style} style took {elapsed_ms:.2f}ms (> 50ms)"
        assert suggestion is not None
        assert suggestion.is_actionable
