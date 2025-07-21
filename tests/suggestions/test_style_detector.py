"""
Comprehensive tests for docstring style detection system.

Tests style detection from files, individual docstrings, and validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.models import StyleDetectionError
from codedocsync.suggestions.style_detector import (
    DocstringStyleDetector,
    style_detector,
)


class TestDocstringStyleDetector:
    """Test DocstringStyleDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DocstringStyleDetector()

    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.config is not None
        assert self.detector._detection_cache == {}
        assert "google" in self.detector._style_indicators
        assert "numpy" in self.detector._style_indicators
        assert "sphinx" in self.detector._style_indicators
        assert "rest" in self.detector._style_indicators

    def test_detector_with_custom_config(self):
        """Test detector with custom configuration."""
        config = SuggestionConfig(default_style="numpy")
        detector = DocstringStyleDetector(config)

        assert detector.config.default_style == "numpy"

    def test_detect_google_style_docstring(self):
        """Test detecting Google-style docstring."""
        google_docstring = """
        Function description.

        Args:
            param1 (str): First parameter description.
            param2 (int): Second parameter description.

        Returns:
            bool: Return value description.

        Raises:
            ValueError: If something goes wrong.
        """

        detected_style = self.detector.detect_from_docstring(google_docstring)
        assert detected_style == "google"

    def test_detect_numpy_style_docstring(self):
        """Test detecting NumPy-style docstring."""
        numpy_docstring = """
        Function description.

        Parameters
        ----------
        param1 : str
            First parameter description.
        param2 : int
            Second parameter description.

        Returns
        -------
        bool
            Return value description.

        Raises
        ------
        ValueError
            If something goes wrong.
        """

        detected_style = self.detector.detect_from_docstring(numpy_docstring)
        assert detected_style == "numpy"

    def test_detect_sphinx_style_docstring(self):
        """Test detecting Sphinx-style docstring."""
        sphinx_docstring = """
        Function description.

        :param param1: First parameter description.
        :type param1: str
        :param param2: Second parameter description.
        :type param2: int
        :returns: Return value description.
        :rtype: bool
        :raises ValueError: If something goes wrong.
        """

        detected_style = self.detector.detect_from_docstring(sphinx_docstring)
        assert detected_style == "sphinx"

    def test_detect_rest_style_docstring(self):
        """Test detecting reStructuredText-style docstring."""
        rest_docstring = """
        Function description.

        .. note::
           This is a note.

        * First item
        * Second item

        1. Numbered item
        2. Another numbered item
        """

        detected_style = self.detector.detect_from_docstring(rest_docstring)
        assert detected_style == "rest"

    def test_detect_empty_docstring(self):
        """Test detecting style from empty docstring."""
        detected_style = self.detector.detect_from_docstring("")
        assert detected_style == self.detector.config.default_style

    def test_detect_plain_docstring(self):
        """Test detecting style from plain docstring without markers."""
        plain_docstring = "Simple function description without any style markers."

        detected_style = self.detector.detect_from_docstring(plain_docstring)
        assert detected_style == self.detector.config.default_style

    def test_detect_ambiguous_docstring(self):
        """Test detecting style from ambiguous docstring."""
        # Mix of styles that should result in low confidence
        ambiguous_docstring = """
        Function description.

        Args: some parameters
        :param test: sphinx style
        """

        detected_style = self.detector.detect_from_docstring(ambiguous_docstring)
        # Should fall back to default due to low confidence
        assert detected_style == self.detector.config.default_style

    def test_detect_from_file_with_python_code(self):
        """Test detecting style from Python file."""
        python_code = '''
def function1(param1, param2):
    """
    Google-style function.

    Args:
        param1 (str): First parameter.
        param2 (int): Second parameter.

    Returns:
        bool: Result.
    """
    return True

def function2(x, y):
    """
    Another Google-style function.

    Args:
        x: X value.
        y: Y value.

    Returns:
        Sum of x and y.
    """
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            temp_path = f.name

        try:
            detected_style = self.detector.detect_from_file(temp_path)
            assert detected_style == "google"
        finally:
            Path(temp_path).unlink()

    def test_detect_from_file_missing_file(self):
        """Test detecting style from missing file."""
        with pytest.raises(StyleDetectionError) as exc_info:
            self.detector.detect_from_file("/non/existent/file.py")

        assert exc_info.value.fallback_style == self.detector.config.default_style

    def test_detect_from_file_no_docstrings(self):
        """Test detecting style from file with no docstrings."""
        python_code = """
def function1():
    pass

def function2():
    return 42
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            temp_path = f.name

        try:
            detected_style = self.detector.detect_from_file(temp_path)
            assert detected_style == self.detector.config.default_style
        finally:
            Path(temp_path).unlink()

    def test_extract_docstrings_from_code(self):
        """Test extracting docstrings from Python code."""
        python_code = '''
"""Module docstring."""

def function1():
    """Function 1 docstring."""
    pass

class TestClass:
    """Class docstring."""

    def method1(self):
        """Method docstring."""
        pass

async def async_function():
    """Async function docstring."""
    pass
'''

        docstrings = self.detector._extract_docstrings_from_code(python_code)

        assert len(docstrings) == 5
        assert "Module docstring." in docstrings
        assert "Function 1 docstring." in docstrings
        assert "Class docstring." in docstrings
        assert "Method docstring." in docstrings
        assert "Async function docstring." in docstrings

    def test_extract_docstrings_invalid_syntax(self):
        """Test extracting docstrings from invalid Python syntax."""
        invalid_code = "def invalid_function( unclosed_paren:"

        docstrings = self.detector._extract_docstrings_from_code(invalid_code)
        assert docstrings == []

    def test_analyze_multiple_docstrings_consistent(self):
        """Test analyzing multiple consistent docstrings."""
        google_docstrings = [
            """
            Function 1.

            Args:
                param: Parameter.

            Returns:
                Result.
            """,
            """
            Function 2.

            Args:
                x: X value.
                y: Y value.

            Returns:
                Sum.
            """,
        ]

        detected_style = self.detector._analyze_multiple_docstrings(google_docstrings)
        assert detected_style == "google"

    def test_analyze_multiple_docstrings_mixed(self):
        """Test analyzing multiple mixed-style docstrings."""
        mixed_docstrings = [
            """
            Google-style function.

            Args:
                param: Parameter.
            """,
            """
            Sphinx-style function.

            :param param: Parameter.
            :returns: Result.
            """,
        ]

        # Should return the style with more votes or default
        detected_style = self.detector._analyze_multiple_docstrings(mixed_docstrings)
        # Result depends on scoring, but should be one of the styles or default
        assert detected_style in [
            "google",
            "sphinx",
            self.detector.config.default_style,
        ]

    def test_analyze_multiple_docstrings_empty(self):
        """Test analyzing empty list of docstrings."""
        detected_style = self.detector._analyze_multiple_docstrings([])
        assert detected_style == self.detector.config.default_style

    def test_calculate_style_score_google(self):
        """Test calculating style score for Google-style docstring."""
        google_docstring = """
        Function description.

        Args:
            param1: First parameter.
            param2: Second parameter.

        Returns:
            Result description.
        """

        indicators = self.detector._style_indicators["google"]
        score = self.detector._calculate_style_score(google_docstring, indicators)

        assert score > 0.0
        assert score <= 1.0

    def test_calculate_style_score_with_forbidden_patterns(self):
        """Test style score calculation with forbidden patterns."""
        # Google docstring with NumPy-style forbidden pattern
        mixed_docstring = """
        Function description.

        Args:
            param: Parameter.

        Parameters
        ----------
        param : str
            Parameter description.
        """

        indicators = self.detector._style_indicators["google"]
        score = self.detector._calculate_style_score(mixed_docstring, indicators)

        # Score should be reduced due to forbidden pattern
        assert score >= 0.0  # Should not be negative

    def test_validate_style_consistency_google(self):
        """Test validating Google-style consistency."""
        valid_google = """
        Function description.

        Args:
            param1 (str): First parameter.
            param2 (int): Second parameter.

        Returns:
            bool: Result.
        """

        is_valid, issues = self.detector.validate_style_consistency(
            valid_google, "google"
        )
        assert is_valid
        assert len(issues) == 0

    def test_validate_style_consistency_google_issues(self):
        """Test validating Google-style with issues."""
        invalid_google = """
        Function description.

        Args
            param1: Missing colon in section header.
        """

        is_valid, issues = self.detector.validate_style_consistency(
            invalid_google, "google"
        )
        assert not is_valid
        assert len(issues) > 0
        assert any("colon" in issue.lower() for issue in issues)

    def test_validate_style_consistency_numpy(self):
        """Test validating NumPy-style consistency."""
        valid_numpy = """
        Function description.

        Parameters
        ----------
        param1 : str
            First parameter.

        Returns
        -------
        bool
            Result.
        """

        is_valid, issues = self.detector.validate_style_consistency(
            valid_numpy, "numpy"
        )
        assert is_valid
        assert len(issues) == 0

    def test_validate_style_consistency_numpy_issues(self):
        """Test validating NumPy-style with issues."""
        invalid_numpy = """
        Function description.

        Parameters
        param1 : str
            Missing underline.
        """

        is_valid, issues = self.detector.validate_style_consistency(
            invalid_numpy, "numpy"
        )
        assert not is_valid
        assert len(issues) > 0
        assert any(
            "dashes" in issue.lower() or "underline" in issue.lower()
            for issue in issues
        )

    def test_validate_style_consistency_sphinx(self):
        """Test validating Sphinx-style consistency."""
        valid_sphinx = """
        Function description.

        :param param1: First parameter.
        :type param1: str
        :returns: Result.
        :rtype: bool
        """

        is_valid, issues = self.detector.validate_style_consistency(
            valid_sphinx, "sphinx"
        )
        assert is_valid
        assert len(issues) == 0

    def test_validate_style_consistency_unknown_style(self):
        """Test validating unknown style."""
        text = "Some docstring text."

        is_valid, issues = self.detector.validate_style_consistency(
            text, "unknown_style"
        )
        assert not is_valid
        assert any("Unknown style" in issue for issue in issues)

    def test_get_style_confidence(self):
        """Test getting confidence score for specific style."""
        google_docstring = """
        Function description.

        Args:
            param: Parameter.

        Returns:
            Result.
        """

        google_confidence = self.detector.get_style_confidence(
            google_docstring, "google"
        )
        numpy_confidence = self.detector.get_style_confidence(google_docstring, "numpy")

        assert google_confidence > numpy_confidence
        assert 0.0 <= google_confidence <= 1.0
        assert 0.0 <= numpy_confidence <= 1.0

    def test_get_all_style_scores(self):
        """Test getting confidence scores for all styles."""
        docstring = """
        Function description.

        Args:
            param: Parameter.
        """

        scores = self.detector.get_all_style_scores(docstring)

        assert "google" in scores
        assert "numpy" in scores
        assert "sphinx" in scores
        assert "rest" in scores

        for _style, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_caching_behavior(self):
        """Test that detection results are cached."""
        docstring = "Simple docstring for caching test."

        # First call
        result1 = self.detector.detect_from_docstring(docstring)
        cache_size_after_first = len(self.detector._detection_cache)

        # Second call with same docstring
        result2 = self.detector.detect_from_docstring(docstring)
        cache_size_after_second = len(self.detector._detection_cache)

        assert result1 == result2
        assert cache_size_after_first == cache_size_after_second  # No new cache entry

    def test_clear_cache(self):
        """Test clearing detection cache."""
        # Populate cache
        self.detector.detect_from_docstring("Test docstring")
        assert len(self.detector._detection_cache) > 0

        # Clear cache
        self.detector.clear_cache()
        assert len(self.detector._detection_cache) == 0

    def test_get_style_template_google(self):
        """Test getting Google-style templates."""
        templates = self.detector.get_style_template("google")

        assert "section_header" in templates
        assert "parameter" in templates
        assert "return" in templates
        assert "raises" in templates

        assert "{section}" in templates["section_header"]
        assert "{name}" in templates["parameter"]
        assert "{type}" in templates["parameter"]

    def test_get_style_template_numpy(self):
        """Test getting NumPy-style templates."""
        templates = self.detector.get_style_template("numpy")

        assert "section_header" in templates
        assert "{underline}" in templates["section_header"]
        assert ":" in templates["parameter"]

    def test_get_style_template_sphinx(self):
        """Test getting Sphinx-style templates."""
        templates = self.detector.get_style_template("sphinx")

        assert ":param" in templates["parameter"]
        assert ":returns" in templates["return"]
        assert ":raises" in templates["raises"]

    def test_get_style_template_unknown(self):
        """Test getting templates for unknown style defaults to Google."""
        templates = self.detector.get_style_template("unknown_style")
        google_templates = self.detector.get_style_template("google")

        assert templates == google_templates

    @patch("random.sample")
    def test_detect_from_project_sampling(self, mock_sample):
        """Test project detection with file sampling."""
        # Create temporary project structure
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create multiple Python files
            for i in range(60):  # More than default sample size
                file_path = project_path / f"file_{i}.py"
                file_path.write_text(
                    f'''
def function_{i}():
    """
    Google-style function {i}.

    Args:
        param: Parameter.

    Returns:
        Result.
    """
    pass
'''
                )

            # Mock random.sample to return first 50 files
            py_files = list(project_path.rglob("*.py"))
            mock_sample.return_value = py_files[:50]

            detected_style = self.detector.detect_from_project(
                str(project_path), sample_size=50
            )

            # Should have called random.sample since we have more than 50 files
            mock_sample.assert_called_once()
            assert detected_style == "google"

    def test_detect_from_project_no_python_files(self):
        """Test project detection with no Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory
            detected_style = self.detector.detect_from_project(temp_dir)
            assert detected_style == self.detector.config.default_style

    def test_detect_from_project_with_errors(self):
        """Test project detection handles file errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create one valid file
            valid_file = project_path / "valid.py"
            valid_file.write_text(
                '''
def valid_function():
    """
    Args:
        param: Parameter.
    """
    pass
'''
            )

            # Create one file that will cause errors
            invalid_file = project_path / "invalid.py"
            invalid_file.write_text("invalid python syntax {{{")

            # Should still work despite the invalid file
            detected_style = self.detector.detect_from_project(str(project_path))
            assert detected_style in ["google", self.detector.config.default_style]


class TestGlobalStyleDetector:
    """Test global style detector instance."""

    def test_global_style_detector_exists(self):
        """Test that global style detector exists."""
        assert style_detector is not None
        assert isinstance(style_detector, DocstringStyleDetector)

    def test_global_style_detector_usage(self):
        """Test using global style detector."""
        result = style_detector.detect_from_docstring("Simple docstring.")
        assert isinstance(result, str)


class TestStyleDetectionIntegration:
    """Test style detection integration scenarios."""

    def test_real_world_google_examples(self):
        """Test detection with real-world Google-style examples."""
        examples = [
            """
            Authenticates a user with the given credentials.

            Args:
                username (str): The username for authentication.
                password (str): The password for authentication.
                remember_me (bool, optional): Whether to remember the login.
                    Defaults to False.

            Returns:
                bool: True if authentication successful, False otherwise.

            Raises:
                ValueError: If username or password is empty.
                ConnectionError: If unable to connect to authentication server.
            """,
            """
            Calculates the moving average of a time series.

            Args:
                data (List[float]): Time series data points.
                window_size (int): Size of the moving window.

            Returns:
                List[float]: Moving averages for each window position.

            Example:
                >>> calculate_moving_average([1, 2, 3, 4, 5], 3)
                [2.0, 3.0, 4.0]
            """,
        ]

        detector = DocstringStyleDetector()

        for example in examples:
            detected = detector.detect_from_docstring(example)
            assert (
                detected == "google"
            ), f"Failed to detect Google style in: {example[:50]}..."

    def test_real_world_numpy_examples(self):
        """Test detection with real-world NumPy-style examples."""
        examples = [
            """
            Compute the discrete Fourier Transform.

            Parameters
            ----------
            a : array_like
                Input array, can be complex.
            n : int, optional
                Length of the transformed axis of the output.
            axis : int, optional
                Axis over which to compute the FFT.

            Returns
            -------
            out : complex ndarray
                The truncated or zero-padded input, transformed along the axis.

            See Also
            --------
            ifft : The inverse of `fft`.
            fft2 : 2-D FFT.

            Examples
            --------
            >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
            array([ -2.33146835e-16+1.14423775e-17j, ...])
            """,
        ]

        detector = DocstringStyleDetector()

        for example in examples:
            detected = detector.detect_from_docstring(example)
            assert (
                detected == "numpy"
            ), f"Failed to detect NumPy style in: {example[:50]}..."

    def test_style_detection_edge_cases(self):
        """Test style detection edge cases."""
        detector = DocstringStyleDetector()

        # Very short docstrings
        assert detector.detect_from_docstring("Short.") == detector.config.default_style

        # Only summary, no sections
        assert (
            detector.detect_from_docstring("Function summary only.")
            == detector.config.default_style
        )

        # Mixed case section headers
        mixed_case = "Args:\n    param: description"
        assert detector.detect_from_docstring(mixed_case) == "google"

        # Multiple blank lines
        spaced_docstring = """


        Function description.


        Args:
            param: Parameter.


        Returns:
            Result.


        """
        assert detector.detect_from_docstring(spaced_docstring) == "google"
