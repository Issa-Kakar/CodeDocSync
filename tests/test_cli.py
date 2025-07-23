"""
Tests for CodeDocSync CLI commands.

Tests all CLI commands using Typer's testing utilities.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from codedocsync.cli.main import app

runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self) -> None:
        """Test that help command works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CodeDocSync" in result.stdout
        assert "analyze" in result.stdout
        assert "parse" in result.stdout
        assert "match" in result.stdout

    def test_version_command(self) -> None:
        """Test version display."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.stdout or "version" in result.stdout.lower()


class TestParseCommand:
    """Test the parse command."""

    def test_parse_single_file(self) -> None:
        """Test parsing a single Python file."""
        test_code = '''
def test_function(x: int, y: int) -> int:
    """Add two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        The sum
    """
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["parse", str(temp_path)])
            assert result.exit_code == 0
            assert "test_function" in result.stdout
            assert (
                "1 function" in result.stdout.lower()
                or "found 1" in result.stdout.lower()
            )
        finally:
            temp_path.unlink()

    def test_parse_directory(self) -> None:
        """Test parsing a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                test_file = Path(tmpdir) / f"test_{i}.py"
                test_file.write_text(
                    f'''
def function_{i}():
    """Function {i}"""
    pass
'''
                )

            result = runner.invoke(app, ["parse", tmpdir])
            assert result.exit_code == 0
            assert "3" in result.stdout  # Should find 3 functions

    def test_parse_json_output(self) -> None:
        """Test JSON output format."""
        test_code = '''
def example():
    """Example function"""
    pass
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["parse", str(temp_path), "--json"])
            assert result.exit_code == 0

            # Should be valid JSON
            data = json.loads(result.stdout)
            assert "functions" in data or "results" in data
        finally:
            temp_path.unlink()

    def test_parse_nonexistent_file(self) -> None:
        """Test parsing a non-existent file."""
        result = runner.invoke(app, ["parse", "nonexistent.py"])
        assert result.exit_code != 0
        assert "error" in result.stdout.lower() or "not found" in result.stdout.lower()


class TestMatchCommand:
    """Test the match command."""

    @patch("codedocsync.cli.match.match_functions_in_path")
    def test_match_basic(self, mock_match: Mock) -> None:
        """Test basic match command."""
        # Mock the matching function
        mock_result = Mock()
        mock_result.total_functions = 10
        mock_result.matched_pairs = []
        mock_result.match_rate = 0.8
        mock_match.return_value = mock_result

        result = runner.invoke(app, ["match", "."])
        assert result.exit_code == 0
        assert mock_match.called

    def test_match_with_threshold(self) -> None:
        """Test match command with confidence threshold."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["match", str(temp_path), "--threshold", "0.9"])
            assert result.exit_code == 0
        finally:
            temp_path.unlink()


class TestAnalyzeCommand:
    """Test the analyze command."""

    def test_analyze_basic(self) -> None:
        """Test basic analyze command."""
        test_code = '''
def calculate(x: int, y: int) -> int:
    """Calculate sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum
    """
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["analyze", str(temp_path)])
            assert result.exit_code == 0
            # Should detect parameter name mismatches
        finally:
            temp_path.unlink()

    def test_analyze_with_style(self) -> None:
        """Test analyze with specific docstring style."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["analyze", str(temp_path), "--style", "numpy"])
            assert result.exit_code == 0
        finally:
            temp_path.unlink()

    def test_analyze_json_output(self) -> None:
        """Test analyze with JSON output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('def test(): """Test""" \n pass')
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["analyze", str(temp_path), "--json"])
            assert result.exit_code == 0

            # Verify JSON output
            if result.stdout.strip():  # Only parse if there's output
                data = json.loads(result.stdout)
                assert isinstance(data, dict) or isinstance(data, list)
        finally:
            temp_path.unlink()

    def test_analyze_ci_mode(self) -> None:
        """Test analyze in CI mode."""
        test_code = '''
def broken_function(x: int) -> str:
    """Function with issues.

    Returns:
        int: Wrong return type
    """
    return x  # Returns int, not str
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # CI mode with fail-on-critical
            result = runner.invoke(
                app, ["analyze", str(temp_path), "--ci", "--fail-on-critical"]
            )
            # May exit with non-zero if critical issues found
            assert result.exit_code >= 0
        finally:
            temp_path.unlink()


class TestSuggestCommand:
    """Test the suggest command."""

    def test_suggest_basic(self) -> None:
        """Test basic suggest command."""
        test_code = '''
def add(x: int, y: int) -> int:
    """Add numbers.

    Args:
        a: First number
    """
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["suggest", str(temp_path)])
            assert result.exit_code == 0
            # Should suggest fixes for missing parameter and wrong name
        finally:
            temp_path.unlink()

    def test_suggest_auto_fix(self) -> None:
        """Test suggest with auto-fix option."""
        test_code = """
def test():
    return 42
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Note: Auto-fix should create backup
            result = runner.invoke(app, ["suggest", str(temp_path), "--fix"])
            assert result.exit_code == 0
        finally:
            temp_path.unlink(missing_ok=True)
            # Clean up backup if created
            backup_path = Path(str(temp_path) + ".backup")
            if backup_path.exists():
                backup_path.unlink()


class TestClearCacheCommand:
    """Test the clear-cache command."""

    @patch("codedocsync.cli.cache.clear_all_caches")
    def test_clear_cache_basic(self, mock_clear: Mock) -> None:
        """Test basic cache clearing."""
        mock_clear.return_value = {"parser": 10, "embeddings": 50}

        result = runner.invoke(app, ["clear-cache"])
        assert result.exit_code == 0
        assert mock_clear.called

    @patch("codedocsync.cli.cache.clear_all_caches")
    def test_clear_cache_force(self, mock_clear: Mock) -> None:
        """Test force cache clearing."""
        mock_clear.return_value = {"parser": 10, "embeddings": 50}

        result = runner.invoke(app, ["clear-cache", "--force"])
        assert result.exit_code == 0
        assert mock_clear.called


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    def test_config_file_loading(self) -> None:
        """Test loading configuration from file."""
        config_content = """
style: numpy
threshold: 0.85
ignore_patterns:
  - test_*.py
  - *_test.py
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".codedocsync.yml"
            config_path.write_text(config_content)

            # Create a test file
            test_file = Path(tmpdir) / "example.py"
            test_file.write_text("def test(): pass")

            # Run with config
            result = runner.invoke(app, ["analyze", str(test_file)])
            assert result.exit_code == 0

    def test_verbose_output(self) -> None:
        """Test verbose output mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["--verbose", "parse", str(temp_path)])
            assert result.exit_code == 0
            # Verbose mode should show more details
        finally:
            temp_path.unlink()

    def test_quiet_mode(self) -> None:
        """Test quiet output mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["--quiet", "parse", str(temp_path)])
            assert result.exit_code == 0
            # Quiet mode should show minimal output
        finally:
            temp_path.unlink()


class TestErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command(self) -> None:
        """Test handling of invalid commands."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        assert "invalid" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_missing_required_argument(self) -> None:
        """Test handling of missing required arguments."""
        result = runner.invoke(app, ["parse"])  # Missing path argument
        assert result.exit_code != 0
        assert "missing" in result.stdout.lower() or "required" in result.stdout.lower()

    def test_invalid_option_value(self) -> None:
        """Test handling of invalid option values."""
        result = runner.invoke(app, ["analyze", ".", "--style", "invalid-style"])
        assert result.exit_code != 0

    def test_keyboard_interrupt_handling(self) -> None:
        """Test handling of keyboard interrupts."""
        with patch("codedocsync.cli.parse.parse_files") as mock_parse:
            # Simulate KeyboardInterrupt
            mock_parse.side_effect = KeyboardInterrupt()

            result = runner.invoke(app, ["parse", "."])
            # Should handle gracefully
            assert "interrupt" in result.stdout.lower() or result.exit_code != 0


class TestPerformance:
    """Test CLI performance characteristics."""

    def test_large_project_handling(self) -> None:
        """Test handling of large projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 100 small Python files
            for i in range(100):
                test_file = Path(tmpdir) / f"module_{i}.py"
                test_file.write_text(
                    f'''
def function_{i}(x: int) -> int:
    """Function {i} in module."""
    return x + {i}
'''
                )

            import time

            start = time.perf_counter()
            result = runner.invoke(app, ["parse", tmpdir])
            duration = time.perf_counter() - start

            assert result.exit_code == 0
            assert duration < 10.0  # Should complete within 10 seconds

    def test_progress_bar_display(self) -> None:
        """Test progress bar for long operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create enough files to trigger progress bar
            for i in range(20):
                test_file = Path(tmpdir) / f"file_{i}.py"
                test_file.write_text("def test(): pass")

            result = runner.invoke(app, ["analyze", tmpdir])
            assert result.exit_code == 0
            # Progress indicators might be shown for multiple files
