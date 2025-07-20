"""Performance validation tests for unified matching facade."""

import pytest
import time
import tempfile
import psutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from codedocsync.matcher.unified_facade import UnifiedMatchingFacade
from codedocsync.matcher.models import (
    MatchResult,
    MatchedPair,
    MatchType,
    MatchConfidence,
)
from codedocsync.parser import ParsedFunction, FunctionSignature, RawDocstring


class TestUnifiedPerformanceValidation:
    """Validate unified matching performance meets production requirements."""

    @pytest.fixture
    def sample_functions(self):
        """Generate sample functions for performance testing."""
        functions = []
        for i in range(1000):  # Large dataset for performance testing
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"test_function_{i}", parameters=[], return_type="None"
                ),
                docstring=RawDocstring(
                    raw_text=f"Test function {i} docstring", line_number=i * 5 + 2
                ),
                file_path=f"test_module_{i % 50}.py",  # 50 different files
                line_number=i * 5,
                end_line_number=i * 5 + 4,
                source_code=f"def test_function_{i}(): pass",
            )
            functions.append(func)
        return functions

    @pytest.fixture
    def large_project_structure(self):
        """Create a large temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a realistic project structure with many files
            for i in range(100):  # 100 Python files
                module_file = temp_path / f"module_{i}.py"
                content = f'''"""Module {i} documentation."""

def function_{i}_1():
    """Function {i}_1 documentation."""
    pass

def function_{i}_2():
    """Function {i}_2 documentation."""
    pass

class Class{i}:
    """Class {i} documentation."""

    def method_{i}(self):
        """Method {i} documentation."""
        pass
'''
                module_file.write_text(content)

            # Create subdirectories
            for subdir in ["utils", "core", "tests"]:
                (temp_path / subdir).mkdir()
                for i in range(20):
                    subfile = temp_path / subdir / f"{subdir}_module_{i}.py"
                    subfile.write_text(f"def {subdir}_function_{i}(): pass\n")

            yield temp_path

    @pytest.mark.asyncio
    async def test_unified_matching_performance_targets(self, sample_functions):
        """Test that unified matching meets all performance targets."""
        # Performance targets from architecture:
        # - Direct matching: <1ms per function
        # - Contextual matching: <20ms per function
        # - Semantic matching: <200ms per function
        # - Memory usage: <500MB for 10k functions
        # - Total time: <30s for 1000 functions

        with (
            patch("codedocsync.matcher.unified_facade.IntegratedParser") as mock_parser,
            patch("codedocsync.matcher.unified_facade.MatchingFacade") as mock_direct,
            patch(
                "codedocsync.matcher.unified_facade.ContextualMatchingFacade"
            ) as mock_contextual,
            patch(
                "codedocsync.matcher.unified_facade.SemanticMatcher"
            ) as mock_semantic,
        ):
            # Setup mocks for performance testing
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance
            mock_parser_instance.parse_file.return_value = sample_functions[
                :10
            ]  # 10 functions per file

            # Mock results with realistic match distributions
            mock_direct_result = MatchResult(
                total_functions=1000,
                matched_pairs=[
                    MatchedPair(
                        function=func,
                        docstring=func.docstring,
                        match_type=MatchType.EXACT,
                        confidence=MatchConfidence(
                            overall=0.95,
                            name_similarity=1.0,
                            location_score=1.0,
                            signature_similarity=1.0,
                        ),
                        match_reason="Exact match",
                    )
                    for func in sample_functions[:900]  # 90% direct matches
                ],
                unmatched_functions=sample_functions[900:],
            )

            mock_contextual_result = MatchResult(
                total_functions=1000,
                matched_pairs=mock_direct_result.matched_pairs
                + [
                    MatchedPair(
                        function=func,
                        docstring=func.docstring,
                        match_type=MatchType.CONTEXTUAL,
                        confidence=MatchConfidence(
                            overall=0.8,
                            name_similarity=0.7,
                            location_score=0.8,
                            signature_similarity=0.9,
                        ),
                        match_reason="Contextual import match",
                    )
                    for func in sample_functions[900:980]  # 8% contextual matches
                ],
                unmatched_functions=sample_functions[980:],
            )

            mock_semantic_result = MatchResult(
                total_functions=1000,
                matched_pairs=mock_contextual_result.matched_pairs
                + [
                    MatchedPair(
                        function=func,
                        docstring=func.docstring,
                        match_type=MatchType.SEMANTIC,
                        confidence=MatchConfidence(
                            overall=0.7,
                            name_similarity=0.6,
                            location_score=0.5,
                            signature_similarity=0.8,
                        ),
                        match_reason="Semantic similarity match",
                    )
                    for func in sample_functions[980:1000]  # 2% semantic matches
                ],
                unmatched_functions=[],
            )

            # Configure mocks
            mock_direct_instance = Mock()
            mock_direct.return_value = mock_direct_instance
            mock_direct_instance.match_project.return_value = mock_direct_result

            mock_contextual_instance = Mock()
            mock_contextual.return_value = mock_contextual_instance
            mock_contextual_instance.match_project.return_value = mock_contextual_result

            mock_semantic_instance = Mock()
            mock_semantic.return_value = mock_semantic_instance
            mock_semantic_instance.prepare_semantic_index = AsyncMock()
            mock_semantic_instance.match_with_embeddings = AsyncMock(
                return_value=mock_semantic_result
            )
            mock_semantic_instance.get_stats.return_value = {
                "functions_processed": 1000,
                "semantic_matches_found": 20,
                "average_time_per_function_ms": 150.0,
            }

            # Test performance
            facade = UnifiedMatchingFacade()

            # Monitor memory before test
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            _result = await facade.match_project("/test/path", enable_semantic=True)
            total_time = time.time() - start_time

            # Monitor memory after test
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            # Validate performance targets
            assert total_time < 30.0, f"Total time {total_time:.2f}s exceeds 30s target"
            assert (
                memory_growth < 500
            ), f"Memory growth {memory_growth:.1f}MB exceeds 500MB target"

            # Validate match distribution (should follow 90/8/2 pattern)
            stats = facade.get_comprehensive_stats()
            assert (
                stats["matches_by_type"]["direct"] >= 850
            ), "Direct matches below expected range"
            assert (
                50 <= stats["matches_by_type"]["contextual"] <= 150
            ), "Contextual matches outside expected range"
            assert (
                stats["matches_by_type"]["semantic"] <= 50
            ), "Too many semantic matches"

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_project(self, large_project_structure):
        """Test memory efficiency with large project (architecture requirement)."""
        with patch(
            "codedocsync.matcher.unified_facade.IntegratedParser"
        ) as mock_parser:
            # Mock parser to simulate large project
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance

            # Simulate 10 functions per file * 140 files = 1400 functions
            mock_functions = [
                ParsedFunction(
                    signature=FunctionSignature(
                        name=f"func_{i}", parameters=[], return_type="None"
                    ),
                    docstring=RawDocstring(
                        raw_text=f"Function {i}", line_number=i * 2 + 1
                    ),
                    file_path=f"file_{i//10}.py",
                    line_number=i * 2,
                    end_line_number=i * 2 + 1,
                    source_code=f"def func_{i}(): pass",
                )
                for i in range(1400)
            ]
            mock_parser_instance.parse_file.return_value = mock_functions[:10]

            facade = UnifiedMatchingFacade()

            # Monitor memory during execution
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            with (
                patch(
                    "codedocsync.matcher.unified_facade.MatchingFacade"
                ) as mock_direct,
                patch(
                    "codedocsync.matcher.unified_facade.ContextualMatchingFacade"
                ) as mock_contextual,
            ):
                # Configure mocks for memory efficiency test
                mock_direct.return_value.match_project.return_value = MatchResult(
                    total_functions=1400,
                    matched_pairs=[],
                    unmatched_functions=mock_functions,
                )
                mock_contextual.return_value.match_project.return_value = MatchResult(
                    total_functions=1400,
                    matched_pairs=[],
                    unmatched_functions=mock_functions,
                )

                _result = await facade.match_project(
                    str(large_project_structure), enable_semantic=False
                )

                peak_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = peak_memory - initial_memory

                # Architecture requirement: <100MB for 10k embeddings, scale down for 1.4k functions
                expected_max_memory = 50  # MB for 1400 functions
                assert (
                    memory_growth < expected_max_memory
                ), f"Memory growth {memory_growth:.1f}MB exceeds {expected_max_memory}MB for large project"

    def test_throughput_performance_metrics(self):
        """Test throughput meets performance requirements."""
        facade = UnifiedMatchingFacade()

        # Simulate processing statistics
        facade.stats.update(
            {
                "total_time": 5.0,  # 5 seconds
                "functions_processed": 1000,
                "files_processed": 100,
                "matches_by_type": {"direct": 900, "contextual": 80, "semantic": 20},
            }
        )

        # Finalize stats
        facade._finalize_stats(time.time() - 5.0, 1000, 100)

        stats = facade.get_comprehensive_stats()
        throughput = stats["throughput"]

        # Validate throughput targets
        assert (
            throughput["functions_per_second"] >= 100
        ), "Functions per second below target (100/s)"
        assert (
            throughput["files_per_second"] >= 10
        ), "Files per second below target (10/s)"

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self):
        """Test that error recovery doesn't significantly impact performance."""
        with patch(
            "codedocsync.matcher.unified_facade.IntegratedParser"
        ) as mock_parser:
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance

            # Simulate parsing errors for some files
            def side_effect_with_errors(file_path):
                if "error_file" in file_path:
                    raise Exception("Simulated parsing error")
                return [
                    ParsedFunction(
                        signature=FunctionSignature(
                            name="test_func", parameters=[], return_type="None"
                        ),
                        docstring=RawDocstring(raw_text="Test", line_number=2),
                        file_path=file_path,
                        line_number=1,
                        end_line_number=2,
                        source_code="def test_func(): pass",
                    )
                ]

            mock_parser_instance.parse_file.side_effect = side_effect_with_errors

            facade = UnifiedMatchingFacade()

            with (
                patch(
                    "codedocsync.matcher.unified_facade.MatchingFacade"
                ) as mock_direct,
                patch(
                    "codedocsync.matcher.unified_facade.ContextualMatchingFacade"
                ) as mock_contextual,
            ):
                mock_direct.return_value.match_project.return_value = MatchResult(
                    total_functions=0, matched_pairs=[], unmatched_functions=[]
                )
                mock_contextual.return_value.match_project.return_value = MatchResult(
                    total_functions=0, matched_pairs=[], unmatched_functions=[]
                )

                # Create test files including error files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    (temp_path / "normal_file.py").write_text("def test(): pass")
                    (temp_path / "error_file.py").write_text("def test(): pass")

                    start_time = time.time()
                    _result = await facade.match_project(
                        str(temp_path), enable_semantic=False
                    )
                    error_recovery_time = time.time() - start_time

                    # Error recovery should complete quickly
                    assert error_recovery_time < 5.0, "Error recovery took too long"

                    # Should track parsing errors
                    stats = facade.get_comprehensive_stats()
                    assert (
                        stats["error_summary"]["parsing_errors"] > 0
                    ), "Parsing errors not tracked"
                    assert (
                        stats["error_summary"]["total_errors"] > 0
                    ), "Total errors not calculated"

    def test_performance_recommendations_accuracy(self):
        """Test that performance recommendations are accurate and helpful."""
        facade = UnifiedMatchingFacade()

        # Test case 1: Slow parsing
        facade.stats.update(
            {
                "total_time": 10.0,
                "parsing_time": 6.0,  # 60% of total time
                "direct_matching_time": 2.0,
                "contextual_matching_time": 1.0,
                "semantic_matching_time": 1.0,
                "memory_usage": {"initial_mb": 100, "peak_mb": 200, "final_mb": 150},
                "errors": {
                    "parsing_errors": 0,
                    "matching_errors": 0,
                    "total_errors": 0,
                },
            }
        )

        recommendations = facade.get_performance_recommendations()
        parsing_rec = any("cache" in rec.lower() for rec in recommendations)
        assert parsing_rec, "Should recommend caching for slow parsing"

        # Test case 2: High memory usage
        facade.stats["memory_usage"] = {
            "initial_mb": 100,
            "peak_mb": 700,
            "final_mb": 200,
        }
        recommendations = facade.get_performance_recommendations()
        memory_rec = any("memory" in rec.lower() for rec in recommendations)
        assert memory_rec, "Should warn about high memory usage"

        # Test case 3: Good performance
        facade.stats.update(
            {
                "total_time": 2.0,
                "parsing_time": 0.5,
                "direct_matching_time": 0.5,
                "contextual_matching_time": 0.5,
                "semantic_matching_time": 0.5,
                "memory_usage": {"initial_mb": 100, "peak_mb": 150, "final_mb": 120},
            }
        )

        recommendations = facade.get_performance_recommendations()
        good_perf = any("good" in rec.lower() for rec in recommendations)
        assert good_perf, "Should acknowledge good performance"

    def test_efficiency_metrics_calculation(self):
        """Test efficiency metrics are calculated correctly."""
        facade = UnifiedMatchingFacade()

        # Set up test data
        facade.stats.update(
            {
                "total_time": 5.0,
                "functions_processed": 1000,
                "memory_usage": {"initial_mb": 100, "peak_mb": 200, "final_mb": 150},
                "matches_by_type": {"direct": 900, "contextual": 80, "semantic": 20},
                "errors": {"total_errors": 5},
            }
        )

        stats = facade.get_comprehensive_stats()
        efficiency = stats["efficiency_metrics"]

        # Validate calculations
        assert (
            efficiency["functions_per_mb"] == 1000 / 200
        ), "Functions per MB calculation incorrect"
        assert (
            efficiency["matches_per_second"] == 1000 / 5.0
        ), "Matches per second calculation incorrect"
        assert efficiency["error_rate"] == 5 / 1000, "Error rate calculation incorrect"
        assert (
            efficiency["memory_efficiency"] == 1000 / 100
        ), "Memory efficiency calculation incorrect"

    @pytest.mark.asyncio
    async def test_scalability_with_large_datasets(self):
        """Test system scales appropriately with large datasets."""
        # Test different dataset sizes and ensure performance scales linearly or better
        dataset_sizes = [100, 500, 1000]
        performance_results = []

        for size in dataset_sizes:
            with (
                patch(
                    "codedocsync.matcher.unified_facade.IntegratedParser"
                ) as mock_parser,
                patch(
                    "codedocsync.matcher.unified_facade.MatchingFacade"
                ) as mock_direct,
                patch(
                    "codedocsync.matcher.unified_facade.ContextualMatchingFacade"
                ) as mock_contextual,
            ):
                # Setup mocks for different sizes
                functions = [
                    ParsedFunction(
                        signature=FunctionSignature(
                            name=f"func_{i}", parameters=[], return_type="None"
                        ),
                        docstring=RawDocstring(
                            raw_text=f"Function {i}", line_number=i * 2 + 1
                        ),
                        file_path=f"file_{i//10}.py",
                        line_number=i * 2,
                        end_line_number=i * 2 + 1,
                        source_code=f"def func_{i}(): pass",
                    )
                    for i in range(size)
                ]

                mock_parser.return_value.parse_file.return_value = functions[:10]
                mock_direct.return_value.match_project.return_value = MatchResult(
                    total_functions=size,
                    matched_pairs=[],
                    unmatched_functions=functions,
                )
                mock_contextual.return_value.match_project.return_value = MatchResult(
                    total_functions=size,
                    matched_pairs=[],
                    unmatched_functions=functions,
                )

                facade = UnifiedMatchingFacade()

                start_time = time.time()
                await facade.match_project("/test/path", enable_semantic=False)
                processing_time = time.time() - start_time

                performance_results.append((size, processing_time))

        # Check that performance scales reasonably (not exponentially)
        # Time per function should not increase dramatically with size
        for i in range(1, len(performance_results)):
            prev_size, prev_time = performance_results[i - 1]
            curr_size, curr_time = performance_results[i]

            time_per_func_prev = prev_time / prev_size
            time_per_func_curr = curr_time / curr_size

            # Performance per function should not degrade by more than 2x
            degradation_factor = time_per_func_curr / time_per_func_prev
            assert (
                degradation_factor < 2.0
            ), f"Performance degraded by {degradation_factor:.2f}x from {prev_size} to {curr_size} functions"

    def test_concurrent_performance_safety(self):
        """Test that the system handles concurrent operations safely."""
        # While the unified facade isn't explicitly designed for concurrency,
        # it should not crash or corrupt data under concurrent access
        import threading

        facade = UnifiedMatchingFacade()
        results = []
        errors = []

        def concurrent_stats_access():
            try:
                for _ in range(100):
                    stats = facade.get_comprehensive_stats()
                    results.append(stats)
            except Exception as e:
                errors.append(e)

        # Run multiple threads accessing stats concurrently
        threads = [threading.Thread(target=concurrent_stats_access) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        assert len(results) == 500, "Not all concurrent operations completed"

    @pytest.mark.asyncio
    async def test_cleanup_performance(self):
        """Test that cleanup operations complete quickly."""
        facade = UnifiedMatchingFacade()

        # Simulate some processing to create cleanup work
        facade.stats.update(
            {
                "total_time": 5.0,
                "functions_processed": 1000,
                "memory_usage": {"initial_mb": 100, "peak_mb": 200, "final_mb": 200},
            }
        )

        # Test cleanup performance
        start_time = time.time()
        await facade.cleanup()
        cleanup_time = time.time() - start_time

        # Cleanup should be very fast
        assert cleanup_time < 1.0, f"Cleanup took {cleanup_time:.2f}s, should be <1s"
