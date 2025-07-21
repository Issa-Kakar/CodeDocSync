"""Integration tests for the contextual matching system."""

import tempfile
from pathlib import Path

import pytest

from codedocsync.matcher import ContextualMatchingFacade
from codedocsync.utils.config import CodeDocSyncConfig


class TestContextualIntegration:
    """Test complete contextual matching workflows."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create project structure
            (project_root / "src").mkdir()
            (project_root / "src" / "utils").mkdir()
            (project_root / "tests").mkdir()

            # Create main module
            main_module = project_root / "src" / "main.py"
            main_module.write_text(
                '''
"""Main module with documented functions."""

def process_data(data: list) -> dict:
    """Process input data and return results.

    Args:
        data: List of data items to process

    Returns:
        Dictionary with processed results
    """
    return {"processed": len(data)}

def calculate_metrics(values):
    """Calculate metrics from values."""
    return sum(values) / len(values)

def helper_function():
    # No docstring
    pass

from .utils.helpers import format_output
'''
            )

            # Create utils module
            utils_module = project_root / "src" / "utils" / "__init__.py"
            utils_module.write_text("")

            helpers_module = project_root / "src" / "utils" / "helpers.py"
            helpers_module.write_text(
                '''
"""Utility helper functions."""

def format_output(data: dict) -> str:
    """Format output data for display.

    Args:
        data: Dictionary containing data to format

    Returns:
        Formatted string representation
    """
    return str(data)

def validate_input(input_data):
    """Validate input data structure."""
    return isinstance(input_data, (list, dict))

def moved_function():
    """This function was moved from main module."""
    return "moved"
'''
            )

            # Create test file
            test_file = project_root / "tests" / "test_main.py"
            test_file.write_text(
                '''
"""Test file for main module."""

from src.main import process_data, calculate_metrics

def test_process_data():
    """Test process_data function."""
    result = process_data([1, 2, 3])
    assert result == {"processed": 3}

def test_calculate_metrics():
    """Test calculate_metrics function."""
    result = calculate_metrics([1, 2, 3])
    assert result == 2.0
'''
            )

            yield project_root

    def test_project_contextual_matching(self, temp_project):
        """Test contextual matching on a complete project."""
        # Create facade with default config
        facade = ContextualMatchingFacade()

        # Match entire project
        result = facade.match_project(str(temp_project))

        # Verify results
        assert result.total_functions > 0
        assert result.matched_pairs is not None
        assert result.metadata is not None

        # Check performance metrics
        assert "performance" in result.metadata
        assert "matcher_stats" in result.metadata
        assert result.metadata["performance"]["total_time"] > 0

        # Check that both direct and contextual stats are present
        assert "direct" in result.metadata["matcher_stats"]
        assert "contextual" in result.metadata["matcher_stats"]

    def test_single_file_contextual_matching(self, temp_project):
        """Test contextual matching on a single file."""
        main_file = temp_project / "src" / "main.py"

        # Create facade
        facade = ContextualMatchingFacade()

        # Match single file with project context
        result = facade.match_file(str(main_file), str(temp_project))

        # Verify results
        assert result.total_functions > 0
        assert result.matched_pairs is not None

        # Should have processed at least the main file functions
        functions = [pair.function for pair in result.matched_pairs]
        function_names = [f.signature.name for f in functions]

        # Check that main module functions are present
        assert any("process_data" in name for name in function_names)

    def test_contextual_matching_with_config(self, temp_project):
        """Test contextual matching with custom configuration."""
        # Create custom config
        config = CodeDocSyncConfig()
        config.matcher.fuzzy_threshold = 0.9
        config.matcher.enable_fuzzy = True

        # Create facade with config
        facade = ContextualMatchingFacade(config)

        # Match project
        result = facade.match_project(str(temp_project))

        # Verify results
        assert result.total_functions > 0
        assert result.matched_pairs is not None

    def test_performance_metrics_tracking(self, temp_project):
        """Test that performance metrics are properly tracked."""
        facade = ContextualMatchingFacade()

        # Match project
        result = facade.match_project(str(temp_project))

        # Verify all performance metrics are present
        assert "performance" in result.metadata
        perf = result.metadata["performance"]

        assert "total_time" in perf
        assert "parsing_time" in perf
        assert "direct_matching_time" in perf
        assert "contextual_matching_time" in perf
        assert "files_processed" in perf

        # All times should be non-negative
        assert perf["total_time"] >= 0
        assert perf["parsing_time"] >= 0
        assert perf["direct_matching_time"] >= 0
        assert perf["contextual_matching_time"] >= 0
        assert perf["files_processed"] > 0

    def test_print_summary(self, temp_project, capsys):
        """Test performance summary printing."""
        facade = ContextualMatchingFacade()

        # Match project
        facade.match_project(str(temp_project))

        # Print summary
        facade.print_summary()

        # Capture output
        captured = capsys.readouterr()

        # Verify summary contains expected information
        assert "Performance Summary" in captured.out
        assert "Total time:" in captured.out
        assert "Files processed:" in captured.out
        assert "Parsing:" in captured.out
        assert "Direct matching:" in captured.out
        assert "Contextual matching:" in captured.out

    def test_empty_project_handling(self, tmp_path):
        """Test handling of empty projects."""
        # Create empty project
        empty_project = tmp_path / "empty"
        empty_project.mkdir()

        facade = ContextualMatchingFacade()

        # Should handle empty project gracefully
        result = facade.match_project(str(empty_project))

        assert result.total_functions == 0
        assert len(result.matched_pairs) == 0
        assert result.metadata is not None

    def test_cache_usage(self, temp_project):
        """Test that caching works correctly."""
        facade = ContextualMatchingFacade()

        # First run with cache
        result1 = facade.match_project(str(temp_project), use_cache=True)

        # Second run with cache should be faster
        result2 = facade.match_project(str(temp_project), use_cache=True)

        # Results should be consistent
        assert result1.total_functions == result2.total_functions
        assert len(result1.matched_pairs) == len(result2.matched_pairs)

    def test_file_discovery_with_exclusions(self, temp_project):
        """Test file discovery respects exclusions."""
        # Create files that should be excluded
        pycache_dir = temp_project / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "test.pyc").write_text("compiled")

        git_dir = temp_project / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")

        # Create facade
        facade = ContextualMatchingFacade()

        # Discover files
        files = facade._discover_python_files(temp_project)

        # Should not include excluded files
        file_paths = [str(f) for f in files]
        assert not any("__pycache__" in path for path in file_paths)
        assert not any(".git" in path for path in file_paths)

        # Should include Python files
        assert any("main.py" in path for path in file_paths)
        assert any("helpers.py" in path for path in file_paths)

    def test_error_handling_in_file_parsing(self, temp_project):
        """Test error handling when file parsing fails."""
        # Create a file with syntax error
        bad_file = temp_project / "bad_syntax.py"
        bad_file.write_text(
            """
def broken_function(
    # Missing closing parenthesis and body
"""
        )

        facade = ContextualMatchingFacade()

        # Should handle parsing error gracefully
        result = facade.match_project(str(temp_project))

        # Should still process other files
        assert result.total_functions > 0
        assert result.metadata is not None

    def test_integration_with_import_resolution(self, temp_project):
        """Test integration with import resolution system."""
        facade = ContextualMatchingFacade()

        # Match project with imports
        result = facade.match_project(str(temp_project))

        # Should have analyzed imports and built context
        assert result.metadata is not None
        assert "contextual" in result.metadata["matcher_stats"]

        # Should have processed imports
        contextual_stats = result.metadata["matcher_stats"]["contextual"]
        assert contextual_stats["files_analyzed"] > 0
        assert contextual_stats["imports_resolved"] >= 0

    def test_nonexistent_file_handling(self, tmp_path):
        """Test handling of nonexistent files."""
        nonexistent_file = tmp_path / "nonexistent.py"

        facade = ContextualMatchingFacade()

        # Should handle nonexistent file gracefully
        from codedocsync.utils.errors import FileAccessError

        with pytest.raises(FileAccessError):
            facade.match_file(str(nonexistent_file))

    def test_mixed_content_project(self, temp_project):
        """Test project with mixed content types."""
        # Add non-Python files
        (temp_project / "README.md").write_text("# Project")
        (temp_project / "config.json").write_text('{"key": "value"}')
        (temp_project / "src" / "data.txt").write_text("data")

        facade = ContextualMatchingFacade()

        # Should only process Python files
        result = facade.match_project(str(temp_project))

        assert result.total_functions > 0
        assert result.metadata is not None

    def test_deep_directory_structure(self, temp_project):
        """Test handling of deep directory structures."""
        # Create deep structure
        deep_dir = temp_project / "src" / "deep" / "nested" / "modules"
        deep_dir.mkdir(parents=True)

        deep_file = deep_dir / "deep_module.py"
        deep_file.write_text(
            '''
def deep_function():
    """A function in a deeply nested module."""
    return "deep"
'''
        )

        facade = ContextualMatchingFacade()

        # Should handle deep structures
        result = facade.match_project(str(temp_project))

        assert result.total_functions > 0

        # Should have found the deep function
        functions = [pair.function for pair in result.matched_pairs]
        function_names = [f.signature.name for f in functions]
        assert "deep_function" in function_names
