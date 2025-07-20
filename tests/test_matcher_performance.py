"""Performance tests for the matcher."""

import time

from codedocsync.matcher import DirectMatcher
from codedocsync.parser import FunctionSignature, ParsedFunction, RawDocstring


class TestMatcherPerformance:
    """Test matcher performance requirements."""

    def test_exact_match_performance(self):
        """Test exact matching meets <1ms per function requirement."""
        # Create 1000 functions
        functions = []
        for i in range(1000):
            func = ParsedFunction(
                signature=FunctionSignature(name=f"function_{i}"),
                docstring=RawDocstring(raw_text=f"Docstring {i}", line_number=i + 1),
                file_path="test.py",
                line_number=i * 3,
                end_line_number=i * 3 + 2,
                source_code=f"def function_{i}(): pass",
            )
            functions.append(func)

        matcher = DirectMatcher()

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Check performance
        ms_per_function = (duration * 1000) / len(functions)
        assert ms_per_function < 1.0, f"Too slow: {ms_per_function:.2f}ms per function"

        # Verify correctness
        assert len(result.matched_pairs) == 1000
        assert all(p.match_type.value == "exact" for p in result.matched_pairs)

    def test_basic_match_performance(self):
        """Test basic matching performance."""
        # Create functions with docstrings
        functions = []
        for i in range(100):
            name = f"function_{i}"

            func = ParsedFunction(
                signature=FunctionSignature(name=name),
                docstring=RawDocstring(
                    raw_text=f"Function {i} description", line_number=i + 1
                ),
                file_path="test.py",
                line_number=i * 3,
                end_line_number=i * 3 + 2,
                source_code=f"def {name}(): pass",
            )
            functions.append(func)

        matcher = DirectMatcher()

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Check performance
        ms_per_function = (duration * 1000) / len(functions)
        assert ms_per_function < 5.0, f"Too slow: {ms_per_function:.2f}ms per function"

        # Verify that all functions matched (they all have docstrings)
        assert len(result.matched_pairs) == 100

    def test_memory_usage(self):
        """Test memory usage stays reasonable for large numbers of functions."""
        # Create 10,000 functions
        functions = []
        for i in range(10000):
            func = ParsedFunction(
                signature=FunctionSignature(
                    name=f"func_{i}",
                    parameters=[],  # Keep it simple
                ),
                docstring=(
                    RawDocstring(raw_text=f"Doc {i}", line_number=i + 1)
                    if i % 2 == 0
                    else None
                ),
                file_path=f"file_{i % 100}.py",  # 100 different files
                line_number=i,
                end_line_number=i + 1,
                source_code=f"def func_{i}(): pass",
            )
            functions.append(func)

        # Run matching
        matcher = DirectMatcher()
        result = matcher.match_functions(functions)

        # Verify results
        assert result.total_functions == 10000
        assert len(result.matched_pairs) == 5000  # Half have docstrings

    def test_large_file_matching(self):
        """Test performance with large individual files."""
        # Create functions representing a large file
        functions = []
        for i in range(500):  # 500 functions in one file
            func = ParsedFunction(
                signature=FunctionSignature(name=f"large_file_func_{i}"),
                docstring=RawDocstring(
                    raw_text=f"Function {i} in large file", line_number=i * 10 + 2
                ),
                file_path="large_file.py",
                line_number=i * 10,
                end_line_number=i * 10 + 8,
                source_code=f"def large_file_func_{i}():\n    pass",
            )
            functions.append(func)

        matcher = DirectMatcher()

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Should complete quickly even for large files
        assert duration < 1.0, f"Large file matching too slow: {duration:.2f}s"
        assert len(result.matched_pairs) == 500

    def test_multiple_files_performance(self):
        """Test performance across multiple files."""
        # Create functions across 100 different files
        functions = []
        for file_idx in range(100):
            for func_idx in range(10):  # 10 functions per file
                func = ParsedFunction(
                    signature=FunctionSignature(name=f"func_{func_idx}"),
                    docstring=RawDocstring(
                        raw_text=f"Function {func_idx}", line_number=func_idx * 5 + 2
                    ),
                    file_path=f"file_{file_idx}.py",
                    line_number=func_idx * 5,
                    end_line_number=func_idx * 5 + 4,
                    source_code=f"def func_{func_idx}(): pass",
                )
                functions.append(func)

        matcher = DirectMatcher()

        # Time the matching
        start_time = time.time()
        result = matcher.match_functions(functions)
        duration = time.time() - start_time

        # Should handle file grouping efficiently
        ms_per_function = (duration * 1000) / len(functions)
        assert (
            ms_per_function < 1.0
        ), f"Multi-file matching too slow: {ms_per_function:.2f}ms per function"
        assert len(result.matched_pairs) == 1000  # All functions have docstrings
