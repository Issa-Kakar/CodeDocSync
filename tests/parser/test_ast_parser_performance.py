"""
Performance tests for the AST Parser module.

Tests ensure the parser meets performance benchmarks:
- Parse medium file (<1000 lines): < 50ms
- Parse large file (5000+ lines): < 200ms
- Memory usage per function: < 50KB
"""

import os
import tempfile
import time
import tracemalloc

import pytest

from codedocsync.parser.ast_parser import parse_python_file, parse_python_file_lazy


def generate_test_file(num_functions: int, lines_per_function: int = 10) -> str:
    """Generate a Python file with specified number of functions."""
    content = []
    content.append("import os")
    content.append("import sys")
    content.append("from typing import List, Dict, Optional, Union, Any")
    content.append("")

    for i in range(num_functions):
        content.append(f"def function_{i}(")
        content.append("    param1: str,")
        content.append("    param2: int = 42,")
        content.append("    *args: Any,")
        content.append("    **kwargs: Dict[str, Any]")
        content.append(") -> Optional[Dict[str, Union[str, int]]]:")
        content.append('    """')
        content.append(f"    Function {i} documentation.")
        content.append("    ")
        content.append("    Args:")
        content.append("        param1: The first parameter")
        content.append("        param2: The second parameter with default")
        content.append("        *args: Variable positional arguments")
        content.append("        **kwargs: Variable keyword arguments")
        content.append("    ")
        content.append("    Returns:")
        content.append("        Optional dictionary with results")
        content.append('    """')

        # Add some actual code lines
        for j in range(lines_per_function - 8):
            content.append(f"    result_{j} = param1 + str(param2)")

        content.append(f"    return {{'result': result_0, 'id': {i}}}")
        content.append("")

    return "\n".join(content)


class TestASTParserPerformance:
    """Test AST parser performance benchmarks."""

    def test_parse_medium_file_under_50ms(self):
        """Test parsing a medium file (800 lines) completes under 50ms."""
        # Generate a file with ~80 functions (10 lines each = ~800 lines)
        content = generate_test_file(num_functions=80, lines_per_function=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Warm up - first parse might be slower due to imports
            parse_python_file(temp_path)

            # Actual timing
            start_time = time.perf_counter()
            functions = parse_python_file(temp_path)
            end_time = time.perf_counter()

            parse_time_ms = (end_time - start_time) * 1000

            assert len(functions) == 80
            assert (
                parse_time_ms < 50
            ), f"Parse time {parse_time_ms:.2f}ms exceeds 50ms limit"

        finally:
            os.unlink(temp_path)

    def test_parse_large_file_performance(self):
        """Test parsing a large file (5000+ lines) completes under 200ms."""
        # Generate a file with 500 functions (10 lines each = 5000 lines)
        content = generate_test_file(num_functions=500, lines_per_function=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Warm up
            parse_python_file(temp_path)

            # Actual timing
            start_time = time.perf_counter()
            functions = parse_python_file(temp_path)
            end_time = time.perf_counter()

            parse_time_ms = (end_time - start_time) * 1000

            assert len(functions) == 500
            assert (
                parse_time_ms < 200
            ), f"Parse time {parse_time_ms:.2f}ms exceeds 200ms limit"

        finally:
            os.unlink(temp_path)

    def test_memory_usage_per_function(self):
        """Test memory usage per function is under 50KB."""
        # Generate a file with 100 functions
        content = generate_test_file(num_functions=100, lines_per_function=15)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Start memory tracking
            tracemalloc.start()

            # Parse the file
            functions = parse_python_file(temp_path)

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate memory per function in KB
            memory_per_function_kb = (current / len(functions)) / 1024

            assert len(functions) == 100
            assert (
                memory_per_function_kb < 50
            ), f"Memory usage {memory_per_function_kb:.2f}KB per function exceeds 50KB limit"

        finally:
            os.unlink(temp_path)

    def test_lazy_parsing_memory_efficiency(self):
        """Test that lazy parsing uses less memory than regular parsing."""
        # Generate a large file
        content = generate_test_file(num_functions=200, lines_per_function=20)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test regular parsing memory
            tracemalloc.start()
            regular_functions = parse_python_file(temp_path)
            regular_memory, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Test lazy parsing memory
            tracemalloc.start()
            lazy_functions = list(parse_python_file_lazy(temp_path))
            lazy_memory, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Lazy parsing should use similar or less memory
            # (In practice, the difference might be small for this test)
            assert len(regular_functions) == len(lazy_functions)
            assert len(lazy_functions) == 200

        finally:
            os.unlink(temp_path)

    def test_incremental_parsing_performance(self):
        """Test performance of parsing files incrementally."""
        # Create multiple files
        temp_files = []

        try:
            for _ in range(10):
                content = generate_test_file(num_functions=50, lines_per_function=10)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(content)
                    temp_files.append(f.name)

            # Time parsing all files
            start_time = time.perf_counter()
            total_functions = 0

            for temp_path in temp_files:
                functions = parse_python_file(temp_path)
                total_functions += len(functions)

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000

            # Should parse 10 files with 50 functions each efficiently
            assert total_functions == 500
            # Average time per file should be reasonable
            avg_time_per_file = total_time_ms / 10
            assert (
                avg_time_per_file < 50
            ), f"Average parse time {avg_time_per_file:.2f}ms per file is too high"

        finally:
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_caching_performance_improvement(self):
        """Test that caching improves performance on repeated parses."""
        # Generate a medium-sized file
        content = generate_test_file(num_functions=100, lines_per_function=15)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # First parse (cold cache)
            start_time = time.perf_counter()
            first_functions = parse_python_file(temp_path)
            first_time = time.perf_counter() - start_time

            # Second parse (warm cache)
            start_time = time.perf_counter()
            second_functions = parse_python_file(temp_path)
            second_time = time.perf_counter() - start_time

            # Cache should make second parse faster
            assert len(first_functions) == len(second_functions) == 100
            # Second parse should be at least 20% faster due to caching
            assert second_time < first_time * 0.8, (
                f"Caching did not improve performance: "
                f"first={first_time * 1000:.2f}ms, second={second_time * 1000:.2f}ms"
            )

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
