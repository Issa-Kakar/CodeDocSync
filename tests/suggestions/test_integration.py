"""
Tests for suggestion integration layer.

Tests the integration between analyzer results and suggestion generation,
including enhancement of analysis results and batch processing.
"""

from unittest.mock import Mock

import pytest

from codedocsync.analyzer.models import AnalysisResult, InconsistencyIssue
from codedocsync.suggestions.config import SuggestionConfig
from codedocsync.suggestions.integration import (
    EnhancedAnalysisResult,
    EnhancedIssue,
    SuggestionBatchProcessor,
    SuggestionIntegration,
    enhance_multiple_with_suggestions,
    enhance_with_suggestions,
)
from codedocsync.suggestions.models import DocstringStyle, Suggestion, SuggestionType


# Test fixtures
@pytest.fixture
def mock_issue():
    """Create a mock InconsistencyIssue."""
    return InconsistencyIssue(
        issue_type="parameter_name_mismatch",
        severity="critical",
        description="Parameter 'email' doesn't match 'username' in code",
        suggestion="Update parameter name from 'email' to 'username'",
        line_number=45,
        confidence=0.95,
    )


@pytest.fixture
def mock_function():
    """Create a mock ParsedFunction."""
    function = Mock()
    function.signature = Mock()
    function.signature.name = "authenticate_user"
    function.file_path = "auth/user.py"
    function.line_number = 42
    return function


@pytest.fixture
def mock_matched_pair(mock_function):
    """Create a mock MatchedPair."""
    pair = Mock()
    pair.function = mock_function
    pair.documentation = None
    pair.confidence = Mock()
    pair.confidence.value = 0.8
    pair.match_type = Mock()
    pair.match_type.value = "direct"
    pair.match_reason = "exact name match"
    return pair


@pytest.fixture
def mock_analysis_result(mock_matched_pair, mock_issue):
    """Create a mock AnalysisResult."""
    return AnalysisResult(
        matched_pair=mock_matched_pair,
        issues=[mock_issue],
        used_llm=False,
        analysis_time_ms=25.5,
        cache_hit=True,
    )


@pytest.fixture
def mock_suggestion():
    """Create a mock Suggestion."""
    return Suggestion(
        suggestion_type=SuggestionType.PARAMETER_UPDATE,
        original_text='    """Original docstring."""',
        suggested_text='    """Updated docstring with correct parameter."""',
        confidence=0.9,
        style=DocstringStyle.GOOGLE,
        copy_paste_ready=True,
    )


@pytest.fixture
def config():
    """Create a test configuration."""
    return SuggestionConfig(
        default_style="google",
        confidence_threshold=0.5,
        preserve_descriptions=True,
    )


class TestEnhancedIssue:
    """Test EnhancedIssue data model."""

    def test_creation(self):
        """Test creating an enhanced issue."""
        issue = EnhancedIssue(
            issue_type="parameter_missing",
            severity="high",
            description="Missing parameter documentation",
            suggestion="Add parameter to docstring",
            line_number=10,
            confidence=0.8,
        )

        assert issue.issue_type == "parameter_missing"
        assert issue.severity == "high"
        assert issue.confidence == 0.8
        assert issue.rich_suggestion is None
        assert issue.ranking_score is None

    def test_validation(self):
        """Test enhanced issue validation."""
        # Invalid confidence
        with pytest.raises(ValueError):
            EnhancedIssue(
                issue_type="test",
                severity="medium",
                description="test",
                suggestion="test",
                line_number=1,
                confidence=1.5,  # Invalid
            )

        # Invalid line number
        with pytest.raises(ValueError):
            EnhancedIssue(
                issue_type="test",
                severity="medium",
                description="test",
                suggestion="test",
                line_number=0,  # Invalid
                confidence=0.8,
            )


class TestEnhancedAnalysisResult:
    """Test EnhancedAnalysisResult data model."""

    def test_creation(self, mock_matched_pair):
        """Test creating an enhanced analysis result."""
        result = EnhancedAnalysisResult(
            matched_pair=mock_matched_pair,
            analysis_time_ms=100.0,
            suggestion_generation_time_ms=50.0,
            suggestions_generated=2,
            suggestions_skipped=1,
        )

        assert result.total_time_ms == 150.0
        assert result.suggestions_generated == 2
        assert result.suggestions_skipped == 1
        assert not result.has_suggestions  # No issues with suggestions yet

    def test_has_suggestions(self, mock_matched_pair, mock_suggestion):
        """Test has_suggestions property."""
        issue = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="test",
            suggestion="test",
            line_number=1,
            rich_suggestion=mock_suggestion,
        )

        result = EnhancedAnalysisResult(
            matched_pair=mock_matched_pair,
            issues=[issue],
        )

        assert result.has_suggestions
        assert len(result.get_suggestions()) == 1


class TestSuggestionIntegration:
    """Test SuggestionIntegration class."""

    def test_initialization(self, config):
        """Test integration initialization."""
        integration = SuggestionIntegration(config)

        assert integration.config == config
        assert len(integration._generators) > 0
        assert "parameter_name_mismatch" in integration._generators
        assert "return_type_mismatch" in integration._generators

    def test_create_enhanced_issue(self, config, mock_issue):
        """Test creating enhanced issue from original."""
        integration = SuggestionIntegration(config)
        enhanced = integration._create_enhanced_issue(mock_issue)

        assert enhanced.issue_type == mock_issue.issue_type
        assert enhanced.severity == mock_issue.severity
        assert enhanced.description == mock_issue.description
        assert enhanced.confidence == mock_issue.confidence
        assert enhanced.rich_suggestion is None

    def test_create_context(self, config, mock_issue, mock_matched_pair):
        """Test creating suggestion context."""
        integration = SuggestionIntegration(config)
        context = integration._create_context(mock_issue, mock_matched_pair)

        assert context.issue == mock_issue
        assert context.function == mock_matched_pair.function
        assert context.project_style == config.default_style

    def test_enhance_analysis_result(self, config, mock_analysis_result):
        """Test enhancing analysis result with suggestions."""
        # Mock the generator to return a suggestion
        integration = SuggestionIntegration(config)
        mock_generator = Mock()
        mock_suggestion = Mock()
        mock_suggestion.confidence = 0.9
        mock_generator.generate.return_value = mock_suggestion

        # Replace generator for the test issue type
        integration._generators["parameter_name_mismatch"] = mock_generator

        result = integration.enhance_analysis_result(mock_analysis_result)

        assert isinstance(result, EnhancedAnalysisResult)
        assert result.matched_pair == mock_analysis_result.matched_pair
        assert len(result.issues) == 1
        assert result.suggestions_generated >= 0

    def test_enhance_with_low_confidence(self, config, mock_analysis_result):
        """Test enhancement skips low confidence issues."""
        # Set high confidence threshold
        config.confidence_threshold = 0.9

        # Make the issue low confidence
        mock_analysis_result.issues[0].confidence = 0.5

        integration = SuggestionIntegration(config)
        result = integration.enhance_analysis_result(mock_analysis_result)

        # Should skip the suggestion due to low confidence
        assert result.suggestions_skipped > 0
        if result.issues:
            assert result.issues[0].rich_suggestion is None

    def test_generator_failure_handling(self, config, mock_analysis_result):
        """Test handling when generator fails."""
        integration = SuggestionIntegration(config)

        # Mock generator that raises exception
        mock_generator = Mock()
        mock_generator.generate.side_effect = Exception("Generator failed")
        integration._generators["parameter_name_mismatch"] = mock_generator

        # Should not raise exception, should handle gracefully
        result = integration.enhance_analysis_result(mock_analysis_result)

        assert isinstance(result, EnhancedAnalysisResult)
        assert result.suggestions_skipped > 0


class TestSuggestionBatchProcessor:
    """Test SuggestionBatchProcessor class."""

    def test_initialization(self, config):
        """Test batch processor initialization."""
        processor = SuggestionBatchProcessor(config)

        assert processor.config == config
        assert isinstance(processor.integration, SuggestionIntegration)

    def test_process_batch(self, config, mock_analysis_result):
        """Test processing batch of analysis results."""
        processor = SuggestionBatchProcessor(config)
        results = [mock_analysis_result]

        enhanced_results = processor.process_batch(results)

        assert len(enhanced_results) == 1
        assert isinstance(enhanced_results[0], EnhancedAnalysisResult)

    def test_process_batch_with_failures(self, config):
        """Test batch processing handles individual failures."""
        processor = SuggestionBatchProcessor(config)

        # Create a result that will cause issues
        bad_result = Mock()
        bad_result.matched_pair = None  # This should cause issues

        enhanced_results = processor.process_batch([bad_result])

        # Should still return a result, even if enhancement failed
        assert len(enhanced_results) == 1
        assert isinstance(enhanced_results[0], EnhancedAnalysisResult)

    def test_create_suggestion_batch(
        self, config, mock_analysis_result, mock_suggestion
    ):
        """Test creating suggestion batch from results."""
        processor = SuggestionBatchProcessor(config)

        # Create enhanced result with suggestion
        enhanced_issue = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="test",
            suggestion="test",
            line_number=1,
            rich_suggestion=mock_suggestion,
        )

        enhanced_result = EnhancedAnalysisResult(
            matched_pair=mock_analysis_result.matched_pair,
            issues=[enhanced_issue],
            suggestions_generated=1,
        )

        batch = processor.create_suggestion_batch([enhanced_result])

        assert len(batch.suggestions) == 1
        assert batch.functions_processed == 1
        assert batch.total_issues == 1


class TestFactoryFunctions:
    """Test factory functions for integration."""

    def test_enhance_with_suggestions(self, mock_analysis_result, config):
        """Test enhance_with_suggestions factory function."""
        result = enhance_with_suggestions(mock_analysis_result, config)

        assert isinstance(result, EnhancedAnalysisResult)
        assert result.matched_pair == mock_analysis_result.matched_pair

    def test_enhance_multiple_with_suggestions(self, mock_analysis_result, config):
        """Test enhance_multiple_with_suggestions factory function."""
        results = [mock_analysis_result]
        enhanced_results = enhance_multiple_with_suggestions(results, config)

        assert len(enhanced_results) == 1
        assert isinstance(enhanced_results[0], EnhancedAnalysisResult)

    def test_enhance_with_default_config(self, mock_analysis_result):
        """Test enhancement with default configuration."""
        result = enhance_with_suggestions(mock_analysis_result)

        assert isinstance(result, EnhancedAnalysisResult)
        # Should use default config


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_analysis_result(self, config, mock_matched_pair):
        """Test enhancing analysis result with no issues."""
        empty_result = AnalysisResult(
            matched_pair=mock_matched_pair,
            issues=[],
            used_llm=False,
            analysis_time_ms=10.0,
        )

        integration = SuggestionIntegration(config)
        enhanced = integration.enhance_analysis_result(empty_result)

        assert len(enhanced.issues) == 0
        assert enhanced.suggestions_generated == 0
        assert enhanced.suggestions_skipped == 0

    def test_unknown_issue_type(self, config, mock_analysis_result):
        """Test handling unknown issue type."""
        # Change to unknown issue type
        mock_analysis_result.issues[0].issue_type = "unknown_issue_type"

        integration = SuggestionIntegration(config)
        result = integration.enhance_analysis_result(mock_analysis_result)

        # Should handle gracefully
        assert isinstance(result, EnhancedAnalysisResult)
        # Likely skipped due to no generator
        assert result.suggestions_skipped >= 0

    def test_is_edge_case_detection(self, config):
        """Test edge case detection for special functions."""
        integration = SuggestionIntegration(config)

        # Mock function with magic method name
        magic_function = Mock()
        magic_function.signature = Mock()
        magic_function.signature.name = "__init__"

        assert integration._is_edge_case(magic_function)

        # Normal function
        normal_function = Mock()
        normal_function.signature = Mock()
        normal_function.signature.name = "normal_function"

        assert not integration._is_edge_case(normal_function)


class TestPerformanceMetrics:
    """Test performance tracking in integration."""

    def test_timing_metrics(self, config, mock_analysis_result):
        """Test that timing metrics are tracked."""
        integration = SuggestionIntegration(config)
        result = integration.enhance_analysis_result(mock_analysis_result)

        assert result.suggestion_generation_time_ms >= 0
        assert result.total_time_ms >= result.analysis_time_ms

    def test_generation_counts(self, config, mock_analysis_result):
        """Test that generation counts are accurate."""
        integration = SuggestionIntegration(config)
        result = integration.enhance_analysis_result(mock_analysis_result)

        total_processed = result.suggestions_generated + result.suggestions_skipped
        assert total_processed == len(mock_analysis_result.issues)


if __name__ == "__main__":
    pytest.main([__file__])
