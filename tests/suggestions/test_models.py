"""
Comprehensive tests for suggestion data models.

Tests all data models including validation, serialization, and business logic.
"""

import pytest

from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionBatch,
    SuggestionContext,
    SuggestionDiff,
    SuggestionMetadata,
    SuggestionType,
    DocstringStyle,
    SuggestionError,
    StyleDetectionError,
    SuggestionGenerationError,
    SuggestionValidationError,
)


class TestSuggestionDiff:
    """Test SuggestionDiff model."""

    def test_valid_diff_creation(self):
        """Test creating a valid diff."""
        diff = SuggestionDiff(
            original_lines=["line 1", "line 2"],
            suggested_lines=["line 1", "line 2 modified", "line 3"],
            start_line=1,
            end_line=3,
        )

        assert diff.start_line == 1
        assert diff.end_line == 3
        assert diff.lines_changed == 3
        assert diff.additions == 1
        assert diff.deletions == 0

    def test_diff_validation_negative_start_line(self):
        """Test validation fails for negative start line."""
        with pytest.raises(ValueError, match="start_line must be positive"):
            SuggestionDiff(
                original_lines=["line 1"],
                suggested_lines=["line 1"],
                start_line=-1,
                end_line=1,
            )

    def test_diff_validation_end_before_start(self):
        """Test validation fails when end_line < start_line."""
        with pytest.raises(ValueError, match="end_line.*must be >= start_line"):
            SuggestionDiff(
                original_lines=["line 1"],
                suggested_lines=["line 1"],
                start_line=5,
                end_line=2,
            )

    def test_unified_diff_generation(self):
        """Test unified diff generation."""
        diff = SuggestionDiff(
            original_lines=["def func():", "    pass"],
            suggested_lines=["def func():", '    """Docstring."""', "    pass"],
            start_line=1,
            end_line=3,
        )

        unified = diff.to_unified_diff("test.py")
        assert "--- a/test.py" in unified
        assert "+++ b/test.py" in unified
        assert '+    """Docstring."""' in unified

    def test_diff_statistics(self):
        """Test diff statistics calculation."""
        # Test additions
        diff = SuggestionDiff(
            original_lines=["line 1"],
            suggested_lines=["line 1", "line 2", "line 3"],
            start_line=1,
            end_line=3,
        )
        assert diff.additions == 2
        assert diff.deletions == 0
        assert diff.lines_changed == 3

        # Test deletions
        diff = SuggestionDiff(
            original_lines=["line 1", "line 2", "line 3"],
            suggested_lines=["line 1"],
            start_line=1,
            end_line=3,
        )
        assert diff.additions == 0
        assert diff.deletions == 2
        assert diff.lines_changed == 3


class TestSuggestionMetadata:
    """Test SuggestionMetadata model."""

    def test_valid_metadata_creation(self):
        """Test creating valid metadata."""
        metadata = SuggestionMetadata(
            generator_type="parameter_generator",
            generator_version="1.0.0",
            template_used="google_parameter",
            style_detected="google",
            rule_triggers=["parameter_missing"],
            llm_used=False,
            generation_time_ms=150.5,
            token_usage=None,
        )

        assert metadata.generator_type == "parameter_generator"
        assert metadata.generation_time_ms == 150.5
        assert not metadata.llm_used
        assert metadata.rule_triggers == ["parameter_missing"]

    def test_metadata_validation_negative_time(self):
        """Test validation fails for negative generation time."""
        with pytest.raises(ValueError, match="generation_time_ms must be non-negative"):
            SuggestionMetadata(
                generator_type="test",
                generation_time_ms=-10.0,
            )

    def test_metadata_validation_empty_generator_type(self):
        """Test validation fails for empty generator type."""
        with pytest.raises(ValueError, match="generator_type cannot be empty"):
            SuggestionMetadata(generator_type="")

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = SuggestionMetadata(generator_type="test")

        assert metadata.generator_version == "1.0.0"
        assert metadata.template_used is None
        assert metadata.style_detected is None
        assert metadata.rule_triggers == []
        assert not metadata.llm_used
        assert metadata.generation_time_ms == 0.0
        assert metadata.token_usage is None


class TestSuggestionContext:
    """Test SuggestionContext model."""

    def test_valid_context_creation(self):
        """Test creating valid context."""
        # Mock objects for testing
        mock_issue = type("MockIssue", (), {"issue_type": "parameter_missing"})()
        mock_function = type("MockFunction", (), {"name": "test_func"})()

        context = SuggestionContext(
            issue=mock_issue,
            function=mock_function,
            project_style="google",
            max_line_length=88,
            preserve_descriptions=True,
        )

        assert context.project_style == "google"
        assert context.max_line_length == 88
        assert context.preserve_descriptions
        assert context.issue == mock_issue
        assert context.function == mock_function

    def test_context_validation_line_length_too_small(self):
        """Test validation fails for too small line length."""
        mock_issue = type("MockIssue", (), {})()
        mock_function = type("MockFunction", (), {})()

        with pytest.raises(ValueError, match="max_line_length too small"):
            SuggestionContext(
                issue=mock_issue,
                function=mock_function,
                max_line_length=30,
            )

    def test_context_validation_invalid_style(self):
        """Test validation fails for invalid project style."""
        mock_issue = type("MockIssue", (), {})()
        mock_function = type("MockFunction", (), {})()

        with pytest.raises(ValueError, match="Invalid project_style"):
            SuggestionContext(
                issue=mock_issue,
                function=mock_function,
                project_style="invalid_style",
            )

    def test_context_defaults(self):
        """Test context default values."""
        mock_issue = type("MockIssue", (), {})()
        mock_function = type("MockFunction", (), {})()

        context = SuggestionContext(
            issue=mock_issue,
            function=mock_function,
        )

        assert context.project_style == "google"
        assert context.max_line_length == 88
        assert context.preserve_descriptions
        assert context.surrounding_code is None
        assert context.related_functions == []
        assert context.file_imports == []


class TestSuggestion:
    """Test Suggestion model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.valid_diff = SuggestionDiff(
            original_lines=["def func():"],
            suggested_lines=["def func():", '    """Docstring."""'],
            start_line=1,
            end_line=2,
        )

        self.valid_metadata = SuggestionMetadata(generator_type="test_generator")

    def test_valid_suggestion_creation(self):
        """Test creating a valid suggestion."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="def func():\n    pass",
            suggested_text="def func():\n    '''Updated docstring.'''\n    pass",
            confidence=0.85,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
            line_range=(1, 3),
        )

        assert suggestion.suggestion_type == SuggestionType.PARAMETER_UPDATE
        assert suggestion.confidence == 0.85
        assert suggestion.style == "google"
        assert suggestion.is_high_confidence
        assert suggestion.is_ready_to_apply

    def test_suggestion_validation_invalid_confidence(self):
        """Test validation fails for invalid confidence."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            Suggestion(
                suggestion_type=SuggestionType.PARAMETER_UPDATE,
                original_text="test",
                suggested_text="test",
                confidence=1.5,  # Invalid
                diff=self.valid_diff,
                style="google",
                metadata=self.valid_metadata,
            )

    def test_suggestion_validation_empty_text(self):
        """Test validation fails for empty text fields."""
        with pytest.raises(ValueError, match="original_text cannot be empty"):
            Suggestion(
                suggestion_type=SuggestionType.PARAMETER_UPDATE,
                original_text="",  # Invalid
                suggested_text="test",
                confidence=0.8,
                diff=self.valid_diff,
                style="google",
                metadata=self.valid_metadata,
            )

    def test_suggestion_validation_invalid_line_range(self):
        """Test validation fails for invalid line range."""
        with pytest.raises(ValueError, match="Invalid line_range"):
            Suggestion(
                suggestion_type=SuggestionType.PARAMETER_UPDATE,
                original_text="test",
                suggested_text="test",
                confidence=0.8,
                diff=self.valid_diff,
                style="google",
                metadata=self.valid_metadata,
                line_range=(0, 1),  # Invalid start line
            )

    def test_suggestion_validation_invalid_style(self):
        """Test validation fails for invalid style."""
        with pytest.raises(ValueError, match="style must be one of"):
            Suggestion(
                suggestion_type=SuggestionType.PARAMETER_UPDATE,
                original_text="test",
                suggested_text="test",
                confidence=0.8,
                diff=self.valid_diff,
                style="invalid_style",  # Invalid
                metadata=self.valid_metadata,
            )

    def test_suggestion_quality_score(self):
        """Test quality score calculation."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.8,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
            copy_paste_ready=True,
            is_actionable=True,
            validation_passed=True,
        )

        quality_score = suggestion.get_quality_score()
        assert 0.0 <= quality_score <= 1.0
        assert quality_score == 0.8  # All factors are positive

    def test_suggestion_quality_score_with_issues(self):
        """Test quality score with some issues."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.8,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
            copy_paste_ready=False,  # Issue
            is_actionable=False,  # Issue
            validation_passed=True,
        )

        quality_score = suggestion.get_quality_score()
        assert quality_score < 0.8  # Should be lower due to issues

    def test_suggestion_to_dict(self):
        """Test suggestion serialization to dictionary."""
        suggestion = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test updated",
            confidence=0.85,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
            affected_sections=["parameters"],
            line_range=(1, 5),
        )

        result = suggestion.to_dict()

        assert result["suggestion_type"] == "parameter_update"
        assert result["confidence"] == 0.85
        assert result["style"] == "google"
        assert result["affected_sections"] == ["parameters"]
        assert result["line_range"] == (1, 5)
        assert "diff" in result
        assert "metadata" in result
        assert result["quality_score"] == suggestion.get_quality_score()

    def test_suggestion_properties(self):
        """Test suggestion boolean properties."""
        # High confidence suggestion
        high_conf = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.9,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
        )
        assert high_conf.is_high_confidence

        # Low confidence suggestion
        low_conf = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.5,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
        )
        assert not low_conf.is_high_confidence

        # Ready to apply
        ready = Suggestion(
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
            original_text="test",
            suggested_text="test",
            confidence=0.8,
            diff=self.valid_diff,
            style="google",
            metadata=self.valid_metadata,
            copy_paste_ready=True,
            is_actionable=True,
            validation_passed=True,
        )
        assert ready.is_ready_to_apply


class TestSuggestionBatch:
    """Test SuggestionBatch model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.diff = SuggestionDiff(
            original_lines=["test"],
            suggested_lines=["test"],
            start_line=1,
            end_line=1,
        )

        self.metadata = SuggestionMetadata(generator_type="test")

        self.suggestions = [
            Suggestion(
                suggestion_type=SuggestionType.PARAMETER_UPDATE,
                original_text="test1",
                suggested_text="test1",
                confidence=0.9,
                diff=self.diff,
                style="google",
                metadata=self.metadata,
            ),
            Suggestion(
                suggestion_type=SuggestionType.RETURN_UPDATE,
                original_text="test2",
                suggested_text="test2",
                confidence=0.7,
                diff=self.diff,
                style="google",
                metadata=self.metadata,
            ),
            Suggestion(
                suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
                original_text="test3",
                suggested_text="test3",
                confidence=0.5,
                diff=self.diff,
                style="google",
                metadata=self.metadata,
            ),
        ]

    def test_valid_batch_creation(self):
        """Test creating a valid suggestion batch."""
        batch = SuggestionBatch(
            suggestions=self.suggestions,
            function_name="test_function",
            file_path="/path/to/file.py",
            total_generation_time_ms=250.0,
        )

        assert len(batch.suggestions) == 3
        assert batch.function_name == "test_function"
        assert batch.file_path == "/path/to/file.py"
        assert batch.total_generation_time_ms == 250.0

    def test_batch_validation_negative_time(self):
        """Test validation fails for negative generation time."""
        with pytest.raises(
            ValueError, match="total_generation_time_ms must be non-negative"
        ):
            SuggestionBatch(
                suggestions=self.suggestions,
                total_generation_time_ms=-10.0,
            )

    def test_batch_high_confidence_suggestions(self):
        """Test filtering high confidence suggestions."""
        batch = SuggestionBatch(suggestions=self.suggestions)
        high_conf = batch.high_confidence_suggestions

        assert len(high_conf) == 1  # Only the 0.9 confidence suggestion
        assert high_conf[0].confidence == 0.9

    def test_batch_ready_to_apply_suggestions(self):
        """Test filtering ready-to-apply suggestions."""
        # Make one suggestion ready to apply
        self.suggestions[0].copy_paste_ready = True
        self.suggestions[0].is_actionable = True
        self.suggestions[0].validation_passed = True

        batch = SuggestionBatch(suggestions=self.suggestions)
        ready = batch.ready_to_apply_suggestions

        assert len(ready) == 1
        assert ready[0].confidence == 0.9

    def test_batch_average_confidence(self):
        """Test average confidence calculation."""
        batch = SuggestionBatch(suggestions=self.suggestions)
        avg_conf = batch.average_confidence

        expected = (0.9 + 0.7 + 0.5) / 3
        assert abs(avg_conf - expected) < 0.001

    def test_batch_empty_suggestions(self):
        """Test batch with no suggestions."""
        batch = SuggestionBatch(suggestions=[])

        assert batch.average_confidence == 0.0
        assert batch.get_best_suggestion() is None
        assert len(batch.high_confidence_suggestions) == 0
        assert len(batch.ready_to_apply_suggestions) == 0

    def test_batch_best_suggestion(self):
        """Test getting the best suggestion."""
        batch = SuggestionBatch(suggestions=self.suggestions)
        best = batch.get_best_suggestion()

        # Should be the highest confidence one
        assert best.confidence == 0.9

    def test_batch_sort_by_quality(self):
        """Test sorting suggestions by quality."""
        batch = SuggestionBatch(suggestions=self.suggestions)
        sorted_suggestions = batch.sort_by_quality()

        # Should be sorted by quality score (highest first)
        assert len(sorted_suggestions) == 3
        # Quality scores should be in descending order
        for i in range(len(sorted_suggestions) - 1):
            assert (
                sorted_suggestions[i].get_quality_score()
                >= sorted_suggestions[i + 1].get_quality_score()
            )

    def test_batch_summary(self):
        """Test batch summary generation."""
        batch = SuggestionBatch(
            suggestions=self.suggestions,
            function_name="test_func",
            file_path="/test.py",
            total_generation_time_ms=100.0,
        )

        summary = batch.get_summary()

        assert summary["total_suggestions"] == 3
        assert summary["function_name"] == "test_func"
        assert summary["file_path"] == "/test.py"
        assert summary["total_generation_time_ms"] == 100.0
        assert "high_confidence" in summary
        assert "ready_to_apply" in summary
        assert "average_confidence" in summary
        assert "best_quality_score" in summary


class TestSuggestionExceptions:
    """Test suggestion exception classes."""

    def test_suggestion_error(self):
        """Test base SuggestionError."""
        error = SuggestionError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_style_detection_error(self):
        """Test StyleDetectionError with fallback."""
        error = StyleDetectionError("Detection failed", fallback_style="numpy")
        assert str(error) == "Detection failed"
        assert error.fallback_style == "numpy"

    def test_style_detection_error_default_fallback(self):
        """Test StyleDetectionError with default fallback."""
        error = StyleDetectionError("Detection failed")
        assert error.fallback_style == "google"

    def test_suggestion_generation_error(self):
        """Test SuggestionGenerationError with partial result."""
        error = SuggestionGenerationError(
            "Generation failed", partial_result="partial docstring"
        )
        assert str(error) == "Generation failed"
        assert error.partial_result == "partial docstring"

    def test_suggestion_generation_error_no_partial(self):
        """Test SuggestionGenerationError without partial result."""
        error = SuggestionGenerationError("Generation failed")
        assert error.partial_result is None

    def test_suggestion_validation_error(self):
        """Test SuggestionValidationError."""
        error = SuggestionValidationError(
            "Validation failed", suggestion_text="invalid suggestion"
        )
        assert str(error) == "Validation failed"
        assert error.suggestion_text == "invalid suggestion"


class TestEnums:
    """Test enum classes."""

    def test_suggestion_type_enum(self):
        """Test SuggestionType enum values."""
        assert SuggestionType.FULL_DOCSTRING.value == "full_docstring"
        assert SuggestionType.PARAMETER_UPDATE.value == "parameter_update"
        assert SuggestionType.RETURN_UPDATE.value == "return_update"
        assert SuggestionType.RAISES_UPDATE.value == "raises_update"
        assert SuggestionType.DESCRIPTION_UPDATE.value == "description"
        assert SuggestionType.EXAMPLE_UPDATE.value == "example"

    def test_docstring_style_enum(self):
        """Test DocstringStyle enum values."""
        assert DocstringStyle.GOOGLE.value == "google"
        assert DocstringStyle.NUMPY.value == "numpy"
        assert DocstringStyle.SPHINX.value == "sphinx"
        assert DocstringStyle.REST.value == "rest"
        assert DocstringStyle.AUTO_DETECT.value == "auto_detect"

    def test_enum_iteration(self):
        """Test that enums are iterable."""
        suggestion_types = list(SuggestionType)
        assert len(suggestion_types) == 6

        docstring_styles = list(DocstringStyle)
        assert len(docstring_styles) == 5
