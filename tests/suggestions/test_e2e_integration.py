"""
End-to-end integration tests for the complete suggestion pipeline.

Tests the full workflow from parsing to suggestion generation,
including CLI integration and production scenarios.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from codedocsync.analyzer.models import AnalysisResult
from codedocsync.main import app
from codedocsync.matcher.models import (
    MatchConfidence,
    MatchedPair,
    MatchResult,
    MatchType,
)
from codedocsync.suggestions import (
    EnhancedAnalysisResult,
    SuggestionConfig,
    enhance_multiple_with_suggestions,
    enhance_with_suggestions,
)
from codedocsync.suggestions.errors import SuggestionError, get_error_handler
from codedocsync.suggestions.performance import get_performance_monitor

from .fixtures import (
    create_parsed_docstring,
    create_test_function,
    create_test_issue,
)

runner = CliRunner()


class TestFullPipeline:
    """Test complete suggestion pipeline."""

    def test_parse_analyze_suggest_workflow(self) -> None:
        """Test full workflow from parsing to suggestion generation."""
        # Step 1: Create test function
        function = create_test_function(
            name="calculate_total",
            params=["items", "tax_rate"],
            return_type="float",
            docstring="""Calculate total with tax.

            Args:
                items: List of items
                rate: Tax rate

            Returns:
                Total amount
            """,
        )

        # Step 2: Create matched pair
        pair = MatchedPair(
            function=function,
            docstring=create_parsed_docstring(
                summary="Calculate total with tax.",
                params={
                    "items": "List of items",
                    "rate": "Tax rate",
                },
                returns="Total amount",
            ),
            confidence=MatchConfidence(
                overall=0.9,
                name_similarity=1.0,
                location_score=1.0,
                signature_similarity=0.7,
            ),
            match_type=MatchType.EXACT,
            match_reason="Same file documentation",
        )

        # Step 3: Create analysis result with issues
        issues = [
            create_test_issue(
                issue_type="parameter_name_mismatch",
                description="Parameter 'rate' doesn't match 'tax_rate' in code",
            ),
            create_test_issue(
                issue_type="return_type_mismatch",
                description="Return type not documented",
                severity="medium",
            ),
        ]

        analysis_result = AnalysisResult(
            matched_pair=pair,
            issues=issues,
            used_llm=False,
            analysis_time_ms=15.3,
        )

        # Step 4: Generate suggestions
        config = SuggestionConfig(default_style="google")
        enhanced_result = enhance_with_suggestions(analysis_result, config)

        # Verify complete pipeline
        assert enhanced_result is not None
        assert len(enhanced_result.issues) == 2

        # Check parameter mismatch suggestion
        param_issue = enhanced_result.issues[0]
        assert param_issue.rich_suggestion is not None
        assert "tax_rate" in param_issue.rich_suggestion.suggested_text
        assert param_issue.rich_suggestion.confidence >= 0.8

        # Check return type suggestion
        return_issue = enhanced_result.issues[1]
        assert return_issue.rich_suggestion is not None
        assert "float" in return_issue.rich_suggestion.suggested_text

    def test_batch_processing(self) -> None:
        """Test processing multiple functions in batch."""
        # Create multiple test scenarios
        test_cases: list[dict[str, Any]] = [
            {
                "name": "func1",
                "params": ["x", "y"],
                "issue": "parameter_missing",
                "missing": "y",
            },
            {
                "name": "func2",
                "params": ["data"],
                "issue": "missing_returns",
                "return_type": "Dict[str, Any]",
            },
            {
                "name": "func3",
                "params": ["value"],
                "issue": "missing_raises",
                "exceptions": ["ValueError"],
            },
        ]

        results = []
        for case in test_cases:
            function = create_test_function(
                name=case["name"],
                params=case["params"],
                return_type=case.get("return_type"),
            )

            issue = create_test_issue(
                issue_type=case["issue"],
            )

            pair = MatchedPair(
                function=function,
                docstring=None,
                confidence=MatchConfidence(
                    overall=0.9,
                    name_similarity=1.0,
                    location_score=1.0,
                    signature_similarity=0.9,
                ),
                match_type=MatchType.EXACT,
                match_reason="Test",
            )

            result = AnalysisResult(
                matched_pair=pair,
                issues=[issue],
                used_llm=False,
                analysis_time_ms=10.0,
            )

            results.append(result)

        # Process batch
        config = SuggestionConfig()

        enhanced_results = enhance_multiple_with_suggestions(results, config)

        # Verify batch results
        assert len(enhanced_results) == 3
        enhanced_result: EnhancedAnalysisResult
        for enhanced_result in enhanced_results:
            assert len(enhanced_result.issues) > 0
            assert enhanced_result.issues[0].rich_suggestion is not None


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_suggest_command_help(self) -> None:
        """Test suggest command help text."""
        result = runner.invoke(app, ["suggest", "--help"])
        assert result.exit_code == 0
        assert "Generate fix suggestions" in result.stdout
        assert "--style" in result.stdout
        assert "--apply" in result.stdout
        assert "--interactive" in result.stdout

    def test_suggest_command_with_file(self, tmp_path: Path) -> None:
        """Test suggest command with a Python file."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''
def process_data(items, threshold=0.5):
    """Process data items.

    Args:
        items: List of items
        limit: Threshold value

    Returns:
        Processed items
    """
    return [item for item in items if item > threshold]
'''
        )

        # Mock the analysis pipeline
        with patch("codedocsync.main.UnifiedMatchingFacade") as mock_facade:
            mock_match_result = Mock(spec=MatchResult)
            mock_match_result.matched_pairs = [
                MatchedPair(
                    function=create_test_function(
                        name="process_data",
                        params=["items", "threshold"],
                    ),
                    docstring=create_parsed_docstring(
                        params={"items": "List of items", "limit": "Threshold value"}
                    ),
                    confidence=MatchConfidence(
                        overall=0.9,
                        name_similarity=1.0,
                        location_score=1.0,
                        signature_similarity=0.9,
                    ),
                    match_type=MatchType.EXACT,
                    match_reason="Test",
                )
            ]

            mock_facade.return_value.match_file.return_value = mock_match_result

            with patch("codedocsync.main.analyze_multiple_pairs") as mock_analyze:
                mock_analyze.return_value = [
                    AnalysisResult(
                        matched_pair=mock_match_result.matched_pairs[0],
                        issues=[
                            create_test_issue(
                                issue_type="parameter_name_mismatch",
                                description="Parameter name mismatch",
                            )
                        ],
                        used_llm=False,
                        analysis_time_ms=10.0,
                    )
                ]

                result = runner.invoke(
                    app, ["suggest", str(test_file), "--format", "json"]
                )

                # Check command execution
                assert result.exit_code == 0
                output = json.loads(result.stdout)
                assert "summary" in output
                assert "suggestions" in output

    def test_suggest_command_dry_run(self, tmp_path: Path) -> None:
        """Test suggest command with dry-run option."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        with patch("codedocsync.main.UnifiedMatchingFacade"):
            runner.invoke(app, ["suggest", str(test_file), "--apply", "--dry-run"])

            # Should not actually modify files
            assert test_file.read_text() == "def test(): pass"


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""

    def test_performance_tracking(self) -> None:
        """Test that performance is tracked during suggestion generation."""
        monitor = get_performance_monitor()
        monitor.reset()

        # Generate suggestions with monitoring
        function = create_test_function()
        issue = create_test_issue()
        context: Mock = Mock()
        context.issue = issue
        context.function = function
        context.docstring = create_parsed_docstring()
        context.project_style = "google"

        from codedocsync.suggestions.generators import ParameterSuggestionGenerator

        generator = ParameterSuggestionGenerator(SuggestionConfig())

        # Generate with performance tracking
        with monitor.measure("test_suggestion"):
            generator.generate(context)

        # Check metrics
        stats = monitor.get_stats("test_suggestion")
        assert "test_suggestion" in stats
        assert stats["test_suggestion"].count == 1
        assert stats["test_suggestion"].avg_time > 0

    def test_performance_recommendations(self) -> None:
        """Test performance optimization recommendations."""
        monitor = get_performance_monitor()
        monitor.reset()

        # Simulate slow operations
        import time

        for _i in range(5):
            with monitor.measure("slow_operation"):
                time.sleep(0.1)  # Simulate slow operation

        recommendations = monitor.get_recommendations()
        assert len(recommendations) > 0
        assert any("slow" in rec.lower() for rec in recommendations)


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_graceful_degradation(self) -> None:
        """Test graceful degradation when suggestion generation fails."""
        handler = get_error_handler()

        # Create context
        context: Mock = Mock()
        context.issue = create_test_issue()
        context.function = create_test_function()
        context.docstring = create_parsed_docstring()
        context.project_style = "google"

        # Test various error scenarios
        errors = [
            SuggestionError("General error", context=context),
            ValueError("Unexpected error"),
            RuntimeError("Critical error"),
        ]

        for error in errors:
            result = handler.handle_error(error, context)
            # Should return fallback or None
            assert result is None or hasattr(result, "confidence")

    def test_error_recovery_in_batch(self) -> None:
        """Test error recovery during batch processing."""
        # Create batch with one failing item
        results = []

        # Good result
        good_result = AnalysisResult(
            matched_pair=Mock(),
            issues=[create_test_issue()],
            used_llm=False,
            analysis_time_ms=10.0,
        )
        results.append(good_result)

        # Bad result that will cause error
        bad_result = AnalysisResult(
            matched_pair=None,  # type: ignore[arg-type]  # Intentionally invalid for error testing
            issues=[create_test_issue()],
            used_llm=False,
            analysis_time_ms=10.0,
        )
        results.append(bad_result)

        # Process batch with error handling

        enhanced_results = enhance_multiple_with_suggestions(
            results, SuggestionConfig()
        )

        # Should process what it can
        assert len(enhanced_results) >= 1
        # Verify first result was processed successfully
        assert enhanced_results[0].matched_pair is not None


class TestProductionScenarios:
    """Test production-ready scenarios."""

    def test_large_codebase_simulation(self) -> None:
        """Test handling of large codebase with many functions."""
        # Simulate 100 functions with various issues
        num_functions = 100
        results = []

        issue_types = [
            "parameter_name_mismatch",
            "parameter_missing",
            "return_type_mismatch",
            "missing_raises",
            "description_vague",
        ]

        for i in range(num_functions):
            function = create_test_function(
                name=f"function_{i}",
                params=[f"param_{j}" for j in range(i % 5 + 1)],
            )

            issue = create_test_issue(
                issue_type=issue_types[i % len(issue_types)],
                severity=["critical", "high", "medium", "low"][i % 4],
            )

            pair = MatchedPair(
                function=function,
                docstring=None,
                confidence=MatchConfidence(
                    overall=0.9,
                    name_similarity=1.0,
                    location_score=1.0,
                    signature_similarity=0.9,
                ),
                match_type=MatchType.EXACT,
                match_reason="Test",
            )

            result = AnalysisResult(
                matched_pair=pair,
                issues=[issue],
                used_llm=False,
                analysis_time_ms=5.0,
            )

            results.append(result)

        # Process large batch
        config = SuggestionConfig()
        import time

        start_time = time.time()
        enhanced_results = enhance_multiple_with_suggestions(results, config)
        end_time = time.time()

        # Verify performance
        assert len(enhanced_results) == num_functions
        assert end_time - start_time < 5.0  # Should complete in reasonable time

        # Check suggestion quality
        high_confidence_count = sum(
            1
            for result in enhanced_results
            for issue in result.issues
            if issue.rich_suggestion and issue.rich_suggestion.confidence >= 0.8
        )
        assert (
            high_confidence_count > num_functions * 0.7
        )  # Most should be high quality

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency with large suggestions."""
        import gc
        import sys

        # Get initial memory
        gc.collect()
        initial_size = sys.getsizeof(gc.get_objects())

        # Generate many large suggestions
        for _ in range(100):
            function = create_test_function(
                params=[f"param_{i}" for i in range(50)]  # Many parameters
            )
            issue = create_test_issue()

            context: Mock = Mock()
            context.issue = issue
            context.function = function
            context.docstring = create_parsed_docstring(
                params={f"param_{i}": f"Description {i}" for i in range(50)}
            )
            context.project_style = "google"

            from codedocsync.suggestions.generators import ParameterSuggestionGenerator

            generator = ParameterSuggestionGenerator(SuggestionConfig())
            generator.generate(context)

        # Check memory didn't explode
        gc.collect()
        final_size = sys.getsizeof(gc.get_objects())
        memory_growth = final_size - initial_size

        # Memory growth should be reasonable
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth

    def test_concurrent_suggestion_generation(self) -> None:
        """Test concurrent suggestion generation."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        async def generate_suggestion_async(index: int) -> Any:
            """Generate suggestion asynchronously."""
            function = create_test_function(name=f"func_{index}")
            issue = create_test_issue()

            context: Mock = Mock()
            context.issue = issue
            context.function = function
            context.docstring = create_parsed_docstring()
            context.project_style = "google"

            from codedocsync.suggestions.generators import ParameterSuggestionGenerator

            generator = ParameterSuggestionGenerator(SuggestionConfig())

            # Run in thread pool to simulate concurrent access
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, generator.generate, context)

        async def run_concurrent_test() -> list[Any]:
            """Run concurrent generation test."""
            tasks = [generate_suggestion_async(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent generation
        suggestions = asyncio.run(run_concurrent_test())

        # All should complete successfully
        assert len(suggestions) == 10
        assert all(s is not None for s in suggestions)


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_config_precedence(self, tmp_path: Path) -> None:
        """Test configuration precedence (CLI > project > user > default)."""
        # Create project config
        project_config = tmp_path / ".codedocsync.yml"
        project_config.write_text(
            """
suggestion:
  default_style: numpy
  confidence_threshold: 0.8
"""
        )

        # Create user config
        user_config = tmp_path / "user_config.yml"
        user_config.write_text(
            """
suggestion:
  default_style: sphinx
  confidence_threshold: 0.6
"""
        )

        from codedocsync.suggestions.config_manager import SuggestionConfigManager

        manager = SuggestionConfigManager()

        # Test precedence
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.cwd", return_value=tmp_path):
                config = manager.load_config()

                # Project config should override user config
                assert config.suggestion.default_style == "numpy"
                assert config.suggestion.confidence_threshold == 0.8

    # TODO: Re-enable when SUGGESTION_PROFILES and get_profile_config are implemented
    # def test_profile_selection(self) -> None:
    #     """Test configuration profile selection."""
    #     from codedocsync.suggestions.config_manager import (
    #         SUGGESTION_PROFILES,
    #         get_profile_config,
    #     )

    #     # Test each profile
    #     for profile_name in SUGGESTION_PROFILES:
    #         config = get_profile_config(profile_name)
    #         assert config is not None
    #         assert config.suggestion is not None

    #     # Test invalid profile
    #     config = get_profile_config("invalid_profile")
    #     assert config is not None  # Should return default
