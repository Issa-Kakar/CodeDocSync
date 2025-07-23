"""
Comprehensive integration tests for CodeDocSync pipeline.

Tests the complete flow from parsing through analysis to suggestion generation.
"""

import tempfile
from pathlib import Path

import pytest

from codedocsync.analyzer.integration import analyze_matched_pair
from codedocsync.analyzer.models import InconsistencyIssue
from codedocsync.matcher.models import MatchConfidence, MatchedPair, MatchType
from codedocsync.parser import parse_python_file
from codedocsync.suggestions.integration import create_suggestion


class TestFullPipeline:
    """Test the complete CodeDocSync pipeline end-to-end."""

    def test_parse_match_analyze_suggest_flow(self) -> None:
        """Test complete flow from parsing to suggestion generation."""
        # Create a test file with documentation issues
        test_code = '''
def calculate_average(numbers: list[float], weights: list[float] | None = None) -> float:
    """Calculate the average of a list of numbers.

    Args:
        nums: List of numbers to average
        weight: Optional weights for weighted average

    Returns:
        The calculated average
    """
    if weights:
        return sum(n * w for n, w in zip(numbers, weights)) / sum(weights)
    return sum(numbers) / len(numbers)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Step 1: Parse the file
            functions = parse_python_file(str(temp_path))
            assert len(functions) == 1
            function = functions[0]

            # Verify parsing worked
            assert function.signature.name == "calculate_average"
            assert len(function.signature.parameters) == 2
            assert function.docstring is not None

            # Step 2: Create a matched pair (simulating matcher output)
            matched_pair = MatchedPair(
                function=function,
                docstring=function.docstring,
                match_type=MatchType.DIRECT,
                confidence=MatchConfidence(
                    overall=1.0,
                    name_similarity=1.0,
                    location_score=1.0,
                    signature_similarity=1.0,
                ),
                match_reason="Direct match in same file",
            )

            # Step 3: Analyze for issues
            issues = analyze_matched_pair(matched_pair)

            # Should find parameter name mismatches
            assert len(issues) > 0
            param_issues = [
                i for i in issues if i.issue_type == "parameter_name_mismatch"
            ]
            assert len(param_issues) >= 1

            # Step 4: Generate suggestions for each issue
            suggestions = []
            for issue in issues:
                suggestion = create_suggestion(issue, function, "google")
                if suggestion:
                    suggestions.append(suggestion)

            # Should have suggestions for fixing parameter names
            assert len(suggestions) > 0

            # Check that suggestions contain corrections
            all_suggestion_text = " ".join(s.suggested_text for s in suggestions)
            assert "numbers" in all_suggestion_text or "weights" in all_suggestion_text

        finally:
            temp_path.unlink()

    def test_multiple_issues_handling(self) -> None:
        """Test handling of multiple documentation issues in one function."""
        test_code = '''
def process_data(data: dict[str, Any], validate: bool = True) -> dict[str, Any]:
    """Process the input data.

    Args:
        input_data: The data to process

    Returns:
        str: Processed result

    Raises:
        ValueError: If validation fails
    """
    if validate and not data:
        raise ValueError("Empty data")

    if "error" in data:
        raise KeyError("Invalid key found")

    return {"processed": True, **data}
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Parse and analyze
            functions = parse_python_file(str(temp_path))
            function = functions[0]

            # Create matched pair
            matched_pair = MatchedPair(
                function=function,
                docstring=function.docstring,
                match_type=MatchType.DIRECT,
                confidence=MatchConfidence(overall=1.0),
                match_reason="Direct match",
            )

            # Analyze issues
            issues = analyze_matched_pair(matched_pair)

            # Should find multiple issues
            issue_types = {issue.issue_type for issue in issues}

            # Check for expected issues
            assert "parameter_name_mismatch" in issue_types  # input_data vs data
            assert "missing_parameter_doc" in issue_types  # validate not documented
            assert "return_type_mismatch" in issue_types  # str vs dict

            # Generate suggestions
            suggestions = []
            for issue in issues:
                suggestion = create_suggestion(issue, function, "google")
                if suggestion:
                    suggestions.append(suggestion)

            assert len(suggestions) >= 3

        finally:
            temp_path.unlink()

    def test_no_docstring_handling(self) -> None:
        """Test handling of functions without docstrings."""
        test_code = """
def important_function(x: int, y: int) -> int:
    result = x + y
    if result > 100:
        raise ValueError("Result too large")
    return result
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Parse
            functions = parse_python_file(str(temp_path))
            function = functions[0]

            # Should have no docstring
            assert function.docstring is None

            # Create issue for missing docstring
            issue = InconsistencyIssue(
                issue_type="missing_docstring",
                severity="high",
                description="Function has no docstring",
                suggestion="Add a comprehensive docstring",
                line_number=function.line_number,
            )

            # Generate suggestion
            suggestion = create_suggestion(issue, function, "google")

            assert suggestion is not None
            assert "Args:" in suggestion.suggested_text
            assert "x (int):" in suggestion.suggested_text
            assert "y (int):" in suggestion.suggested_text
            assert "Returns:" in suggestion.suggested_text
            assert "int:" in suggestion.suggested_text
            assert "Raises:" in suggestion.suggested_text
            assert "ValueError:" in suggestion.suggested_text

        finally:
            temp_path.unlink()

    def test_async_function_handling(self) -> None:
        """Test handling of async functions."""
        test_code = '''
async def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch data from URL.

    Args:
        url: The URL to fetch from

    Returns:
        The fetched data
    """
    # Async implementation
    return {"url": url, "data": "example"}
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Parse
            functions = parse_python_file(str(temp_path))
            function = functions[0]

            assert function.signature.is_async

            # Analyze
            matched_pair = MatchedPair(
                function=function,
                docstring=function.docstring,
                match_type=MatchType.DIRECT,
                confidence=MatchConfidence(overall=1.0),
                match_reason="Direct match",
            )

            issues = analyze_matched_pair(matched_pair)

            # Should find missing parameter doc for timeout
            param_issues = [i for i in issues if "timeout" in i.description.lower()]
            assert len(param_issues) > 0

            # Generate suggestion
            suggestion = create_suggestion(param_issues[0], function, "google")
            assert suggestion is not None
            assert "timeout" in suggestion.suggested_text

        finally:
            temp_path.unlink()

    def test_class_method_handling(self) -> None:
        """Test handling of class methods and properties."""
        test_code = '''
class DataProcessor:
    @property
    def status(self) -> str:
        """Get the processor status."""
        return self._status

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DataProcessor":
        """Create processor from configuration.

        Args:
            cfg: Configuration dictionary

        Returns:
            New processor instance
        """
        return cls(**config)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Parse
            functions = parse_python_file(str(temp_path))

            # Find the class method
            class_method = next(
                f for f in functions if f.signature.name == "from_config"
            )

            # Check decorators
            assert "classmethod" in class_method.signature.decorators

            # Analyze
            matched_pair = MatchedPair(
                function=class_method,
                docstring=class_method.docstring,
                match_type=MatchType.DIRECT,
                confidence=MatchConfidence(overall=1.0),
                match_reason="Direct match",
            )

            issues = analyze_matched_pair(matched_pair)

            # Should find parameter name mismatch (cfg vs config)
            param_issues = [
                i for i in issues if i.issue_type == "parameter_name_mismatch"
            ]
            assert len(param_issues) > 0

        finally:
            temp_path.unlink()

    @pytest.mark.parametrize("style", ["google", "numpy", "sphinx"])
    def test_different_docstring_styles(self, style: str) -> None:
        """Test pipeline with different docstring styles."""
        test_code = """
def example_function(x: int, y: int) -> int:
    return x * y
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Parse
            functions = parse_python_file(str(temp_path))
            function = functions[0]

            # Create missing docstring issue
            issue = InconsistencyIssue(
                issue_type="missing_docstring",
                severity="high",
                description="Function has no docstring",
                suggestion="Add docstring",
                line_number=function.line_number,
            )

            # Generate suggestion in specified style
            suggestion = create_suggestion(issue, function, style)

            assert suggestion is not None

            # Check style-specific markers
            if style == "google":
                assert "Args:" in suggestion.suggested_text
                assert "Returns:" in suggestion.suggested_text
            elif style == "numpy":
                assert "Parameters" in suggestion.suggested_text
                assert "-------" in suggestion.suggested_text
            elif style == "sphinx":
                assert ":param" in suggestion.suggested_text
                assert ":return:" in suggestion.suggested_text

        finally:
            temp_path.unlink()

    def test_error_recovery(self) -> None:
        """Test pipeline's ability to handle errors gracefully."""
        # Test with invalid Python syntax
        test_code = """
def broken_function(x, y)
    # Missing colon
    return x + y
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            # Should handle syntax errors gracefully
            functions = parse_python_file(str(temp_path))

            # Parser should return empty list or handle error
            assert isinstance(functions, list)

        finally:
            temp_path.unlink()

    def test_performance_characteristics(self) -> None:
        """Test pipeline performance with multiple functions."""
        # Generate a file with 50 functions
        test_code = ""
        for i in range(50):
            test_code += f'''
def function_{i}(x: int, y: int) -> int:
    """Function {i} documentation.

    Args:
        x: First parameter
        y: Second parameter

    Returns:
        The result
    """
    return x + y

'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            import time

            # Measure parsing time
            start = time.perf_counter()
            functions = parse_python_file(str(temp_path))
            parse_time = time.perf_counter() - start

            assert len(functions) == 50
            assert parse_time < 1.0  # Should parse 50 functions in under 1 second

            # Measure analysis time
            start = time.perf_counter()
            for function in functions[:10]:  # Analyze first 10
                matched_pair = MatchedPair(
                    function=function,
                    docstring=function.docstring,
                    match_type=MatchType.DIRECT,
                    confidence=MatchConfidence(overall=1.0),
                    match_reason="Direct match",
                )
                analyze_matched_pair(matched_pair)

            analyze_time = time.perf_counter() - start
            assert (
                analyze_time < 0.5
            )  # Should analyze 10 functions in under 0.5 seconds

        finally:
            temp_path.unlink()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_file(self) -> None:
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            functions = parse_python_file(str(temp_path))
            assert functions == []
        finally:
            temp_path.unlink()

    def test_comments_only_file(self) -> None:
        """Test handling of files with only comments."""
        test_code = '''
# This file contains only comments
# No functions to parse
"""
Module docstring but no functions
"""
# More comments
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            functions = parse_python_file(str(temp_path))
            assert functions == []
        finally:
            temp_path.unlink()

    def test_nested_functions(self) -> None:
        """Test handling of nested function definitions."""
        test_code = '''
def outer_function(x: int) -> int:
    """Outer function.

    Args:
        x: Input value

    Returns:
        Processed value
    """
    def inner_function(y: int) -> int:
        """Inner function."""
        return y * 2

    return inner_function(x)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)

        try:
            functions = parse_python_file(str(temp_path))

            # Should find both functions
            assert len(functions) >= 1
            function_names = {f.signature.name for f in functions}
            assert "outer_function" in function_names

        finally:
            temp_path.unlink()
