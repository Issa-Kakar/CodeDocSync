"""
Comprehensive tests for base suggestion generator functionality.

Tests the abstract base class, validation methods, and utility functions.
"""

from unittest.mock import Mock, patch

import pytest

from codedocsync.suggestions.base import (
    BaseSuggestionGenerator,
    with_suggestion_fallback,
)
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionContext,
    SuggestionDiff,
    SuggestionError,
    SuggestionMetadata,
    SuggestionType,
    SuggestionValidationError,
)


class TestBaseSuggestionGenerator:
    """Test BaseSuggestionGenerator abstract class."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class ConcreteGenerator(BaseSuggestionGenerator):
            def generate(self, context):
                return Suggestion(
                    suggestion_type=SuggestionType.PARAMETER_UPDATE,
                    original_text="def func(): pass",
                    suggested_text='def func():\n    """Docstring."""\n    pass',
                    confidence=0.8,
                    diff=SuggestionDiff(
                        original_lines=["def func(): pass"],
                        suggested_lines=[
                            "def func():",
                            '    """Docstring."""',
                            "    pass",
                        ],
                        start_line=1,
                        end_line=3,
                    ),
                    style="google",
                    metadata=SuggestionMetadata(generator_type="test_generator"),
                )

        self.generator = ConcreteGenerator()

    def test_generator_initialization_default_config(self):
        """Test generator initialization with default config."""
        generator = type(self.generator)()
        assert generator.config is not None
        assert isinstance(generator.config, SuggestionConfig)
        assert generator._validation_cache == {}

    def test_generator_initialization_custom_config(self):
        """Test generator initialization with custom config."""
        config = SuggestionConfig(default_style="numpy", max_line_length=100)
        generator = type(self.generator)(config)

        assert generator.config.default_style == "numpy"
        assert generator.config.max_line_length == 100

    def test_generate_with_timing(self):
        """Test generate_with_timing method."""
        mock_issue = Mock()
        mock_function = Mock()
        context = SuggestionContext(issue=mock_issue, function=mock_function)

        suggestion = self.generator.generate_with_timing(context)

        assert isinstance(suggestion, Suggestion)
        assert suggestion.metadata.generation_time_ms > 0

    def test_generate_with_timing_exception_handling(self):
        """Test generate_with_timing handles exceptions."""

        # Create a generator that raises an exception
        class FailingGenerator(BaseSuggestionGenerator):
            def generate(self, context):
                raise ValueError("Test error")

        generator = FailingGenerator()
        mock_issue = Mock()
        mock_function = Mock()
        context = SuggestionContext(issue=mock_issue, function=mock_function)

        with pytest.raises(SuggestionError) as exc_info:
            generator.generate_with_timing(context)

        assert "Failed to generate suggestion" in str(exc_info.value)
        assert "Test error" in str(exc_info.value)

    def test_validate_suggestion_valid(self):
        """Test validating a valid suggestion."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="def func(): pass",
            suggested_text='def func():\n    """Valid docstring."""\n    pass',
            confidence=0.8,
            diff=SuggestionDiff(
                original_lines=["def func(): pass"],
                suggested_lines=[
                    "def func():",
                    '    """Valid docstring."""',
                    "    pass",
                ],
                start_line=1,
                end_line=3,
            ),
            style="google",
            metadata=SuggestionMetadata(generator_type="test"),
        )

        is_valid = self.generator.validate_suggestion(suggestion)
        assert is_valid
        assert suggestion.validation_passed

    def test_validate_suggestion_invalid_python_string(self):
        """Test validation fails for invalid Python string."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="def func(): pass",
            suggested_text='def func():\n    """Unclosed string',  # Invalid
            confidence=0.8,
            diff=SuggestionDiff(
                original_lines=["def func(): pass"],
                suggested_lines=['def func():\n    """Unclosed string'],
                start_line=1,
                end_line=2,
            ),
            style="google",
            metadata=SuggestionMetadata(generator_type="test"),
        )

        is_valid = self.generator.validate_suggestion(suggestion)
        assert not is_valid
        assert not suggestion.validation_passed

    def test_validate_suggestion_inconsistent_indentation(self):
        """Test validation fails for inconsistent indentation."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="def func(): pass",
            suggested_text='def func():\n\t"""Mixed indentation."""\n    pass',  # Mixed tabs/spaces
            confidence=0.8,
            diff=SuggestionDiff(
                original_lines=["def func(): pass"],
                suggested_lines=['def func():\n\t"""Mixed indentation."""\n    pass'],
                start_line=1,
                end_line=3,
            ),
            style="google",
            metadata=SuggestionMetadata(generator_type="test"),
        )

        is_valid = self.generator.validate_suggestion(suggestion)
        assert not is_valid
        assert not suggestion.validation_passed

    def test_validate_suggestion_vague_content(self):
        """Test validation detects vague suggestions."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="def func(): pass",
            suggested_text='def func():\n    """TODO: fix this."""\n    pass',  # Vague
            confidence=0.8,
            diff=SuggestionDiff(
                original_lines=["def func(): pass"],
                suggested_lines=['def func():\n    """TODO: fix this."""\n    pass'],
                start_line=1,
                end_line=3,
            ),
            style="google",
            metadata=SuggestionMetadata(generator_type="test"),
        )

        self.generator.validate_suggestion(suggestion)
        # Should still pass validation but mark as not actionable
        assert suggestion.validation_passed  # Syntactically valid
        assert not suggestion.is_actionable  # But not actionable

    def test_is_valid_python_string_caching(self):
        """Test that Python string validation uses caching."""
        text = '"""Valid docstring."""'

        # First call
        result1 = self.generator._is_valid_python_string(text)
        cache_size_after_first = len(self.generator._validation_cache)

        # Second call with same text
        result2 = self.generator._is_valid_python_string(text)
        cache_size_after_second = len(self.generator._validation_cache)

        assert result1 == result2
        assert cache_size_after_first == cache_size_after_second

    def test_has_consistent_indentation_valid(self):
        """Test consistent indentation validation."""
        valid_text = '''def func():
    """Docstring."""
    if True:
        return True
    return False'''

        assert self.generator._has_consistent_indentation(valid_text)

    def test_has_consistent_indentation_mixed_tabs_spaces(self):
        """Test mixed tabs and spaces detection."""
        mixed_text = '''def func():
    """Docstring."""
\tif True:  # Tab here
        return True  # Spaces here'''

        assert not self.generator._has_consistent_indentation(mixed_text)

    def test_has_consistent_indentation_too_deep(self):
        """Test overly deep indentation detection."""
        deep_text = '''def func():
                        """Very deep indentation."""
                        pass'''

        assert not self.generator._has_consistent_indentation(deep_text)

    def test_has_proper_quote_escaping_valid(self):
        """Test proper quote escaping validation."""
        valid_text = '"""Valid docstring with "quotes" inside."""'
        assert self.generator._has_proper_quote_escaping(valid_text)

    def test_has_proper_quote_escaping_unmatched(self):
        """Test detection of unmatched quotes."""
        invalid_text = '"""Unmatched triple quotes'
        assert not self.generator._has_proper_quote_escaping(invalid_text)

    def test_matches_expected_style_google(self):
        """Test Google style validation."""
        google_text = """
        Function description.

        Args:
            param: Parameter description.

        Returns:
            Result description.
        """

        assert self.generator._matches_expected_style(google_text, "google")

    def test_matches_expected_style_numpy(self):
        """Test NumPy style validation."""
        numpy_text = """
        Function description.

        Parameters
        ----------
        param : str
            Parameter description.
        """

        assert self.generator._matches_expected_style(numpy_text, "numpy")

    def test_matches_expected_style_sphinx(self):
        """Test Sphinx style validation."""
        sphinx_text = """
        Function description.

        :param param: Parameter description.
        :returns: Result description.
        """

        assert self.generator._matches_expected_style(sphinx_text, "sphinx")

    def test_matches_expected_style_unknown(self):
        """Test unknown style defaults to basic validation."""
        text = "Simple description."
        assert self.generator._matches_expected_style(text, "unknown_style")

    def test_validate_google_style_valid(self):
        """Test Google style specific validation."""
        valid_google = """
        Args:
            param: Description.

        Returns:
            Result.
        """

        issues = self.generator._validate_google_style(valid_google)
        assert len(issues) == 0

    def test_validate_google_style_missing_colon(self):
        """Test Google style validation catches missing colons."""
        invalid_google = """
        Args
            param: Description.
        """

        issues = self.generator._validate_google_style(invalid_google)
        assert len(issues) > 0
        assert any("colon" in issue.lower() for issue in issues)

    def test_validate_google_style_bad_indentation(self):
        """Test Google style validation catches bad indentation."""
        invalid_google = """
        Args:
        param: Not indented properly.
        """

        issues = self.generator._validate_google_style(invalid_google)
        assert len(issues) > 0
        assert any("indented" in issue.lower() for issue in issues)

    def test_validate_numpy_style_valid(self):
        """Test NumPy style specific validation."""
        valid_numpy = """
        Parameters
        ----------
        param : str
            Description.
        """

        issues = self.generator._validate_numpy_style(valid_numpy)
        assert len(issues) == 0

    def test_validate_numpy_style_missing_underline(self):
        """Test NumPy style validation catches missing underlines."""
        invalid_numpy = """
        Parameters
        param : str
            Description.
        """

        issues = self.generator._validate_numpy_style(invalid_numpy)
        assert len(issues) > 0
        assert any("dashes" in issue.lower() for issue in issues)

    def test_validate_numpy_style_wrong_underline_length(self):
        """Test NumPy style validation catches wrong underline length."""
        invalid_numpy = """
        Parameters
        ---
        param : str
            Description.
        """

        issues = self.generator._validate_numpy_style(invalid_numpy)
        assert len(issues) > 0
        assert any("length" in issue.lower() for issue in issues)

    def test_validate_sphinx_style_valid(self):
        """Test Sphinx style specific validation."""
        valid_sphinx = """
        :param param: Description.
        :type param: str
        :returns: Result.
        """

        issues = self.generator._validate_sphinx_style(valid_sphinx)
        assert len(issues) == 0

    def test_validate_rest_style_valid(self):
        """Test reStructuredText style validation."""
        valid_rest = """
        .. note::
           This is a note.

        .. code-block:: python

           def example():
               pass
        """

        issues = self.generator._validate_rest_style(valid_rest)
        assert len(issues) == 0

    def test_validate_rest_style_bad_indentation(self):
        """Test rST style validation catches bad directive indentation."""
        invalid_rest = """
        .. note::
        Not properly indented.
        """

        issues = self.generator._validate_rest_style(invalid_rest)
        assert len(issues) > 0
        assert any("indented" in issue.lower() for issue in issues)

    def test_is_actionable_valid(self):
        """Test actionability validation for valid suggestions."""
        actionable_text = """
        Function authenticates a user with given credentials.

        Args:
            username (str): The user's login name.
            password (str): The user's password.

        Returns:
            bool: True if authentication successful.
        """

        assert self.generator._is_actionable(actionable_text)

    def test_is_actionable_vague_phrases(self):
        """Test actionability validation catches vague phrases."""
        vague_texts = [
            "Fix this function.",
            "Update documentation.",
            "TODO: improve description.",
            "Make it better.",
            "[placeholder]",
        ]

        for text in vague_texts:
            assert not self.generator._is_actionable(text)

    def test_is_actionable_too_short(self):
        """Test actionability validation catches too short suggestions."""
        short_text = "Fix."
        assert not self.generator._is_actionable(short_text)

    def test_is_actionable_no_identifiers(self):
        """Test actionability validation catches text with no identifiers."""
        no_identifiers = "!@#$%^&*()"
        assert not self.generator._is_actionable(no_identifiers)

    def test_create_metadata(self):
        """Test metadata creation."""
        metadata = self.generator.create_metadata(
            generator_type="test_generator",
            template_used="google_template",
            style_detected="google",
            rule_triggers=["parameter_missing", "return_missing"],
        )

        assert metadata.generator_type == "test_generator"
        assert metadata.template_used == "google_template"
        assert metadata.style_detected == "google"
        assert metadata.rule_triggers == ["parameter_missing", "return_missing"]
        assert not metadata.llm_used  # Default value

    def test_format_docstring_lines(self):
        """Test docstring line formatting."""
        content = "First line\nSecond line\n\nFourth line"
        formatted = self.generator.format_docstring_lines(content, indent=4)

        expected = [
            "    First line",
            "    Second line",
            "",
            "    Fourth line",
        ]

        assert formatted == expected

    def test_format_docstring_lines_custom_indent(self):
        """Test docstring line formatting with custom indent."""
        content = "Line one\nLine two"
        formatted = self.generator.format_docstring_lines(content, indent=8)

        expected = [
            "        Line one",
            "        Line two",
        ]

        assert formatted == expected

    def test_wrap_long_lines_no_wrapping_needed(self):
        """Test line wrapping when no wrapping is needed."""
        short_text = "This is a short line."
        wrapped = self.generator.wrap_long_lines(short_text, max_length=100)

        assert wrapped == short_text

    def test_wrap_long_lines_wrapping_needed(self):
        """Test line wrapping when wrapping is needed."""
        long_text = "This is a very long line that exceeds the maximum length and should be wrapped."
        wrapped = self.generator.wrap_long_lines(long_text, max_length=40)

        lines = wrapped.split("\n")
        assert len(lines) > 1
        # All lines should be within the limit (except possibly the last)
        for line in lines[:-1]:
            assert len(line) <= 40

    def test_wrap_long_lines_preserve_indentation(self):
        """Test line wrapping preserves indentation."""
        indented_text = (
            "    This is an indented line that is too long and needs wrapping."
        )
        wrapped = self.generator.wrap_long_lines(indented_text, max_length=30)

        lines = wrapped.split("\n")
        assert len(lines) > 1
        # First line should have original indentation
        assert lines[0].startswith("    ")
        # Continuation lines should have extra indentation
        for line in lines[1:]:
            if line.strip():  # Non-empty line
                assert lines[1].startswith("        ")  # 4 + 4 spaces

    def test_preserve_existing_content_empty_original(self):
        """Test content preservation with empty original."""
        result = self.generator.preserve_existing_content(
            original="",
            updated_section="New section content",
            section_name="parameters",
        )

        assert result == "New section content"

    def test_preserve_existing_content_with_original(self):
        """Test content preservation with existing content."""
        result = self.generator.preserve_existing_content(
            original="Original content",
            updated_section="Updated section",
            section_name="parameters",
        )

        # Currently returns updated section (simple implementation)
        assert result == "Updated section"

    def test_validation_exception_handling(self):
        """Test validation handles exceptions gracefully."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.8,
            diff=SuggestionDiff(
                original_lines=["test"],
                suggested_lines=["test"],
                start_line=1,
                end_line=1,
            ),
            style="google",
            metadata=SuggestionMetadata(generator_type="test"),
        )

        # Mock a method to raise an exception
        with patch.object(
            self.generator,
            "_is_valid_python_string",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(SuggestionValidationError) as exc_info:
                self.generator.validate_suggestion(suggestion)

            assert "Validation failed" in str(exc_info.value)
            assert "Test error" in str(exc_info.value)


class TestWithSuggestionFallback:
    """Test with_suggestion_fallback decorator."""

    def test_decorator_normal_execution(self):
        """Test decorator allows normal execution."""

        @with_suggestion_fallback
        def normal_function(arg1, arg2):
            return f"Result: {arg1} + {arg2}"

        result = normal_function("A", "B")
        assert result == "Result: A + B"

    def test_decorator_handles_suggestion_error_with_partial(self):
        """Test decorator handles SuggestionError with partial result."""
        from codedocsync.suggestions.models import SuggestionGenerationError

        @with_suggestion_fallback
        def failing_function():
            error = SuggestionGenerationError(
                "Generation failed", partial_result="Partial content"
            )
            raise error

        result = failing_function()
        assert result == "Partial content"

    def test_decorator_handles_suggestion_error_without_partial(self):
        """Test decorator handles SuggestionError without partial result."""
        from codedocsync.suggestions.models import SuggestionError

        @with_suggestion_fallback
        def failing_function():
            raise SuggestionError("Generation failed")

        # Should re-raise since no partial result and no fallback method
        with pytest.raises(SuggestionError):
            failing_function()

    def test_decorator_handles_unexpected_exception(self):
        """Test decorator converts unexpected exceptions."""

        @with_suggestion_fallback
        def failing_function():
            raise ValueError("Unexpected error")

        with pytest.raises(SuggestionError) as exc_info:
            failing_function()

        assert "Unexpected error in suggestion generation" in str(exc_info.value)
        assert "Unexpected error" in str(exc_info.value)

    def test_decorator_with_fallback_method(self):
        """Test decorator can use fallback method."""

        class MockGenerator:
            def _create_fallback_suggestion(self, context):
                return "Fallback suggestion"

            @with_suggestion_fallback
            def generate(self, context):
                from codedocsync.suggestions.models import SuggestionError

                raise SuggestionError("Generation failed")

        generator = MockGenerator()
        result = generator.generate("test_context")
        assert result == "Fallback suggestion"


class TestBaseSuggestionGeneratorIntegration:
    """Test integration scenarios for BaseSuggestionGenerator."""

    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""

        class TestGenerator(BaseSuggestionGenerator):
            def generate(self, context):
                return Suggestion(
                    suggestion_type=SuggestionType.PARAMETER_UPDATE,
                    original_text="def func(x): pass",
                    suggested_text='''def func(x):
    """
    Process input value.

    Args:
        x (int): Input value to process.

    Returns:
        int: Processed result.
    """
    pass''',
                    confidence=0.9,
                    diff=SuggestionDiff(
                        original_lines=["def func(x): pass"],
                        suggested_lines=[
                            "def func(x):",
                            '    """',
                            "    Process input value.",
                            "    ",
                            "    Args:",
                            "        x (int): Input value to process.",
                            "        ",
                            "    Returns:",
                            "        int: Processed result.",
                            '    """',
                            "    pass",
                        ],
                        start_line=1,
                        end_line=11,
                    ),
                    style="google",
                    metadata=SuggestionMetadata(
                        generator_type="test_generator",
                        template_used="google_function",
                    ),
                )

        generator = TestGenerator()
        mock_issue = Mock()
        mock_function = Mock()
        context = SuggestionContext(issue=mock_issue, function=mock_function)

        # Generate with timing
        suggestion = generator.generate_with_timing(context)

        # Validate
        is_valid = generator.validate_suggestion(suggestion)

        assert is_valid
        assert suggestion.validation_passed
        assert suggestion.is_actionable
        assert suggestion.is_high_confidence
        assert suggestion.metadata.generation_time_ms > 0

    def test_style_specific_validation_integration(self):
        """Test integration of style-specific validation."""

        class StyleTestGenerator(BaseSuggestionGenerator):
            def __init__(self, style):
                super().__init__()
                self._style = style

            def generate(self, context):
                if self._style == "google":
                    suggested_text = '''
    """
    Function description.

    Args:
        param: Parameter description.

    Returns:
        Result description.
    """'''
                elif self._style == "numpy":
                    suggested_text = '''
    """
    Function description.

    Parameters
    ----------
    param : str
        Parameter description.

    Returns
    -------
    str
        Result description.
    """'''
                else:
                    suggested_text = '"""Simple description."""'

                return Suggestion(
                    suggestion_type=SuggestionType.FULL_DOCSTRING,
                    original_text='"""Old description."""',
                    suggested_text=suggested_text,
                    confidence=0.8,
                    diff=SuggestionDiff(
                        original_lines=['"""Old description."""'],
                        suggested_lines=[suggested_text],
                        start_line=1,
                        end_line=2,
                    ),
                    style=self._style,
                    metadata=SuggestionMetadata(generator_type="style_test"),
                )

        # Test Google style
        google_generator = StyleTestGenerator("google")
        context = SuggestionContext(issue=Mock(), function=Mock())
        google_suggestion = google_generator.generate(context)
        assert google_generator.validate_suggestion(google_suggestion)

        # Test NumPy style
        numpy_generator = StyleTestGenerator("numpy")
        numpy_suggestion = numpy_generator.generate(context)
        assert numpy_generator.validate_suggestion(numpy_suggestion)
