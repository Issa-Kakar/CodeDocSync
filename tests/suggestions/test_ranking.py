"""
Tests for suggestion ranking and filtering system.

Tests the ranking algorithms, filtering criteria, and priority boosting
functionality for organizing suggestions by importance and quality.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from codedocsync.suggestions.integration import EnhancedAnalysisResult, EnhancedIssue
from codedocsync.suggestions.models import (
    Suggestion,
    SuggestionDiff,
    SuggestionType,
)
from codedocsync.suggestions.ranking import (
    PriorityBooster,
    RankingConfig,
    RankingMetrics,
    RankingStrategy,
    SuggestionFilter,
    SuggestionRanker,
    create_balanced_ranker,
    create_permissive_ranker,
    create_strict_ranker,
)


# Test fixtures
@pytest.fixture
def critical_issue() -> Any:
    """Create a critical severity issue."""
    return EnhancedIssue(
        issue_type="parameter_name_mismatch",
        severity="critical",
        description="Critical parameter mismatch",
        suggestion="Fix critical parameter",
        line_number=10,
        confidence=0.95,
    )


@pytest.fixture
def high_issue() -> Any:
    """Create a high severity issue."""
    return EnhancedIssue(
        issue_type="return_type_mismatch",
        severity="high",
        description="Return type mismatch",
        suggestion="Fix return type",
        line_number=20,
        confidence=0.85,
    )


@pytest.fixture
def medium_issue() -> Any:
    """Create a medium severity issue."""
    return EnhancedIssue(
        issue_type="missing_raises",
        severity="medium",
        description="Missing exception documentation",
        suggestion="Add exception docs",
        line_number=30,
        confidence=0.75,
    )


@pytest.fixture
def low_issue() -> Any:
    """Create a low severity issue."""
    return EnhancedIssue(
        issue_type="example_invalid",
        severity="low",
        description="Invalid example",
        suggestion="Fix example",
        line_number=40,
        confidence=0.65,
    )


@pytest.fixture
def low_confidence_issue() -> Any:
    """Create a low confidence issue."""
    return EnhancedIssue(
        issue_type="description_outdated",
        severity="medium",
        description="Outdated description",
        suggestion="Update description",
        line_number=50,
        confidence=0.3,  # Low confidence
    )


@pytest.fixture
def high_quality_suggestion() -> Any:
    """Create a high quality suggestion."""
    return Suggestion(
        suggestion_type=SuggestionType.PARAMETER_UPDATE,
        original_text="old text",
        suggested_text="new text",
        confidence=0.9,
        diff=SuggestionDiff(
            original_lines=["old text"],
            suggested_lines=["new text"],
            start_line=1,
            end_line=1,
        ),
        style="google",
        copy_paste_ready=True,
    )


@pytest.fixture
def low_quality_suggestion() -> Any:
    """Create a low quality suggestion."""
    return Suggestion(
        suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        original_text="old text",
        suggested_text="very long text that spans many lines and is complex",
        confidence=0.6,
        diff=SuggestionDiff(
            original_lines=["old text"],
            suggested_lines=["very long text that spans many lines and is complex"],
            start_line=1,
            end_line=1,
        ),
        style="google",
        copy_paste_ready=False,
    )


class TestRankingConfig:
    """Test RankingConfig data model."""

    def test_default_config(self) -> None:
        """Test default ranking configuration."""
        config = RankingConfig()

        assert config.strategy == RankingStrategy.BALANCED
        assert config.min_confidence == 0.5
        assert config.max_suggestions_per_function is None
        assert config.severity_weights["critical"] > config.severity_weights["low"]
        assert "parameter_name_mismatch" in config.issue_type_priorities

    def test_custom_config(self) -> None:
        """Test custom ranking configuration."""
        config = RankingConfig(
            strategy=RankingStrategy.SEVERITY_FIRST,
            min_confidence=0.8,
            max_total_suggestions=10,
            copy_paste_ready_only=True,
        )

        assert config.strategy == RankingStrategy.SEVERITY_FIRST
        assert config.min_confidence == 0.8
        assert config.max_total_suggestions == 10
        assert config.copy_paste_ready_only


class TestRankingMetrics:
    """Test RankingMetrics calculations."""

    def test_total_score_calculation(self) -> None:
        """Test total score calculation."""
        metrics = RankingMetrics(
            severity_score=10.0,
            confidence_score=5.0,
            impact_score=3.0,
            actionability_score=2.0,
            complexity_penalty=1.0,
            user_priority_bonus=1.0,
        )

        expected_total = 10.0 + 5.0 + 3.0 + 2.0 + 1.0 - 1.0
        assert metrics.total_score == expected_total

    def test_zero_scores(self) -> None:
        """Test metrics with zero values."""
        metrics = RankingMetrics()
        assert metrics.total_score == 0.0


class TestSuggestionRanker:
    """Test SuggestionRanker class."""

    def test_initialization(self) -> None:
        """Test ranker initialization."""
        config = RankingConfig()
        ranker = SuggestionRanker(config)

        assert ranker.config == config

    def test_rank_by_severity(
        self, critical_issue: Any, high_issue: Any, medium_issue: Any, low_issue: Any
    ) -> None:
        """Test ranking by severity."""
        config = RankingConfig(strategy=RankingStrategy.SEVERITY_FIRST)
        ranker = SuggestionRanker(config)

        issues = [low_issue, medium_issue, critical_issue, high_issue]
        ranked = ranker.rank_suggestions(issues)

        # Should be ordered by severity (critical > high > medium > low)
        severities = [issue.severity for issue in ranked]

        # Check that critical comes before others
        critical_index = next(i for i, s in enumerate(severities) if s == "critical")
        low_index = next(i for i, s in enumerate(severities) if s == "low")
        assert critical_index < low_index

    def test_rank_by_confidence(
        self, critical_issue: Any, low_confidence_issue: Any
    ) -> None:
        """Test ranking considers confidence."""
        config = RankingConfig(strategy=RankingStrategy.CONFIDENCE_FIRST)
        ranker = SuggestionRanker(config)

        issues = [low_confidence_issue, critical_issue]
        ranked = ranker.rank_suggestions(issues)

        # Higher confidence should rank higher
        assert ranked[0].confidence > ranked[1].confidence

    def test_filter_by_confidence(
        self, critical_issue: Any, low_confidence_issue: Any
    ) -> None:
        """Test filtering by minimum confidence."""
        config = RankingConfig(min_confidence=0.7)
        ranker = SuggestionRanker(config)

        issues = [critical_issue, low_confidence_issue]  # 0.95, 0.3
        ranked = ranker.rank_suggestions(issues)

        # Should filter out low confidence issue
        assert len(ranked) == 1
        assert ranked[0].confidence >= 0.7

    def test_filter_by_severity(self, critical_issue: Any, low_issue: Any) -> None:
        """Test filtering by allowed severities."""
        config = RankingConfig(allowed_severities=["critical", "high"])
        ranker = SuggestionRanker(config)

        issues = [critical_issue, low_issue]
        ranked = ranker.rank_suggestions(issues)

        # Should filter out low severity issue
        assert len(ranked) == 1
        assert ranked[0].severity in ["critical", "high"]

    def test_exclude_issue_types(self, critical_issue: Any, high_issue: Any) -> None:
        """Test excluding specific issue types."""
        config = RankingConfig(excluded_issue_types=["parameter_name_mismatch"])
        ranker = SuggestionRanker(config)

        issues = [critical_issue, high_issue]
        ranked = ranker.rank_suggestions(issues)

        # Should exclude parameter_name_mismatch
        issue_types = [issue.issue_type for issue in ranked]
        assert "parameter_name_mismatch" not in issue_types

    def test_copy_paste_ready_filter(
        self,
        critical_issue: Any,
        high_quality_suggestion: Any,
        low_quality_suggestion: Any,
    ) -> None:
        """Test filtering for copy-paste ready suggestions."""
        config = RankingConfig(copy_paste_ready_only=True)
        ranker = SuggestionRanker(config)

        # Add suggestions to issues
        issue1 = critical_issue
        issue1.rich_suggestion = high_quality_suggestion  # copy_paste_ready=True

        issue2 = EnhancedIssue(
            issue_type="test",
            severity="medium",
            description="test",
            suggestion="test",
            line_number=1,
            rich_suggestion=low_quality_suggestion,  # copy_paste_ready=False
        )

        issues = [issue1, issue2]
        ranked = ranker.rank_suggestions(issues)

        # Should only include copy-paste ready suggestions
        for issue in ranked:
            if issue.rich_suggestion:
                assert issue.rich_suggestion.copy_paste_ready

    def test_max_suggestions_limit(
        self, critical_issue: Any, high_issue: Any, medium_issue: Any
    ) -> None:
        """Test limiting maximum suggestions."""
        config = RankingConfig(max_total_suggestions=2)
        ranker = SuggestionRanker(config)

        issues = [critical_issue, high_issue, medium_issue]
        ranked = ranker.rank_suggestions(issues)

        assert len(ranked) <= 2

    def test_empty_issues_list(self) -> None:
        """Test ranking empty issues list."""
        ranker = SuggestionRanker()
        ranked = ranker.rank_suggestions([])

        assert ranked == []

    def test_ranking_score_assignment(self, critical_issue: Any) -> None:
        """Test that ranking scores are assigned."""
        ranker = SuggestionRanker()
        ranked = ranker.rank_suggestions([critical_issue])

        assert ranked[0].ranking_score is not None
        assert ranked[0].ranking_score > 0

    def test_rank_analysis_results(self, critical_issue: Any) -> None:
        """Test ranking analysis results."""
        mock_pair: Mock = Mock()
        mock_pair.function = Mock()

        result = EnhancedAnalysisResult(
            matched_pair=mock_pair,
            issues=[critical_issue],
        )

        ranker = SuggestionRanker()
        ranked_results = ranker.rank_analysis_results([result])

        assert len(ranked_results) == 1
        assert len(ranked_results[0].issues) == 1
        assert ranked_results[0].issues[0].ranking_score is not None


class TestSuggestionFilter:
    """Test SuggestionFilter static methods."""

    def test_by_confidence(
        self, critical_issue: Any, low_confidence_issue: Any
    ) -> None:
        """Test filtering by confidence."""
        issues = [critical_issue, low_confidence_issue]
        filtered = SuggestionFilter.by_confidence(issues, 0.7)

        assert len(filtered) == 1
        assert filtered[0].confidence >= 0.7

    def test_by_severity(
        self, critical_issue: Any, medium_issue: Any, low_issue: Any
    ) -> None:
        """Test filtering by severity."""
        issues = [critical_issue, medium_issue, low_issue]
        filtered = SuggestionFilter.by_severity(issues, ["critical", "high"])

        assert len(filtered) == 1  # Only critical issue
        assert filtered[0].severity in ["critical", "high"]

    def test_by_issue_type(self, critical_issue: Any, high_issue: Any) -> None:
        """Test filtering by issue type."""
        issues = [critical_issue, high_issue]
        filtered = SuggestionFilter.by_issue_type(issues, ["parameter_name_mismatch"])

        assert len(filtered) == 1
        assert filtered[0].issue_type == "parameter_name_mismatch"

    def test_exclude_issue_types(self, critical_issue: Any, high_issue: Any) -> None:
        """Test excluding issue types."""
        issues = [critical_issue, high_issue]
        filtered = SuggestionFilter.exclude_issue_types(
            issues, ["parameter_name_mismatch"]
        )

        assert len(filtered) == 1
        assert filtered[0].issue_type != "parameter_name_mismatch"

    def test_copy_paste_ready_only(self, high_quality_suggestion: Any) -> None:
        """Test filtering for copy-paste ready only."""
        issue1 = EnhancedIssue(
            issue_type="test1",
            severity="high",
            description="test1",
            suggestion="test1",
            line_number=1,
            rich_suggestion=high_quality_suggestion,  # copy_paste_ready=True
        )

        issue2 = EnhancedIssue(
            issue_type="test2",
            severity="high",
            description="test2",
            suggestion="test2",
            line_number=2,
            # No rich_suggestion
        )

        issues = [issue1, issue2]
        filtered = SuggestionFilter.copy_paste_ready_only(issues)

        assert len(filtered) == 1
        assert filtered[0].rich_suggestion is not None
        assert filtered[0].rich_suggestion.copy_paste_ready

    def test_top_n(
        self, critical_issue: Any, high_issue: Any, medium_issue: Any
    ) -> None:
        """Test getting top N suggestions."""
        # Set ranking scores
        critical_issue.ranking_score = 10.0
        high_issue.ranking_score = 8.0
        medium_issue.ranking_score = 6.0

        issues = [medium_issue, critical_issue, high_issue]  # Unsorted
        top_2 = SuggestionFilter.top_n(issues, 2)

        assert len(top_2) == 2
        assert top_2[0].ranking_score == 10.0  # Critical
        assert top_2[1].ranking_score == 8.0  # High


class TestPriorityBooster:
    """Test PriorityBooster class."""

    def test_initialization(self) -> None:
        """Test booster initialization."""
        booster = PriorityBooster()

        assert len(booster.boost_rules) > 0

    def test_calculate_boost(self, critical_issue: Any) -> None:
        """Test boost calculation."""
        booster = PriorityBooster()
        boost = booster.calculate_boost(critical_issue)

        # Should get boost for critical parameter issue
        assert boost >= 0

    def test_add_custom_rule(self, critical_issue: Any) -> None:
        """Test adding custom boost rule."""
        booster = PriorityBooster()

        def custom_rule(issue: Any) -> float:
            return 5.0 if issue.severity == "critical" else 0.0

        booster.add_boost_rule(custom_rule)
        boost = booster.calculate_boost(critical_issue)

        # Should include boost from custom rule
        assert boost >= 5.0

    def test_rule_failure_handling(self, critical_issue: Any) -> None:
        """Test handling when boost rule fails."""
        booster = PriorityBooster()

        def failing_rule(issue: Any) -> float:
            raise Exception("Rule failed")

        booster.add_boost_rule(failing_rule)

        # Should not raise exception
        boost = booster.calculate_boost(critical_issue)
        assert boost >= 0


class TestRankingStrategies:
    """Test different ranking strategies."""

    def test_severity_first_strategy(
        self, critical_issue: Any, medium_issue: Any
    ) -> None:
        """Test severity-first ranking strategy."""
        config = RankingConfig(strategy=RankingStrategy.SEVERITY_FIRST)
        ranker = SuggestionRanker(config)

        # Set same confidence for both
        critical_issue.confidence = 0.7
        medium_issue.confidence = 0.7

        issues = [medium_issue, critical_issue]
        ranked = ranker.rank_suggestions(issues)

        # Critical should rank higher due to severity boost
        assert ranked[0].severity == "critical"

    def test_confidence_first_strategy(
        self, critical_issue: Any, medium_issue: Any
    ) -> None:
        """Test confidence-first ranking strategy."""
        config = RankingConfig(strategy=RankingStrategy.CONFIDENCE_FIRST)
        ranker = SuggestionRanker(config)

        # Set different confidences
        critical_issue.confidence = 0.6
        medium_issue.confidence = 0.9

        issues = [critical_issue, medium_issue]
        ranked = ranker.rank_suggestions(issues)

        # Medium issue should rank higher due to confidence boost
        assert ranked[0].confidence > ranked[1].confidence

    def test_balanced_strategy(self, critical_issue: Any, medium_issue: Any) -> None:
        """Test balanced ranking strategy."""
        config = RankingConfig(strategy=RankingStrategy.BALANCED)
        ranker = SuggestionRanker(config)

        issues = [critical_issue, medium_issue]
        ranked = ranker.rank_suggestions(issues)

        # Should consider both severity and confidence
        assert len(ranked) == 2


class TestFactoryFunctions:
    """Test factory functions for creating rankers."""

    def test_create_strict_ranker(self) -> None:
        """Test creating strict ranker."""
        ranker = create_strict_ranker()

        assert ranker.config.min_confidence == 0.8
        assert ranker.config.copy_paste_ready_only
        assert ranker.config.allowed_severities == ["critical", "high"]

    def test_create_permissive_ranker(self) -> None:
        """Test creating permissive ranker."""
        ranker = create_permissive_ranker()

        assert ranker.config.min_confidence == 0.3
        assert not ranker.config.copy_paste_ready_only

    def test_create_balanced_ranker(self) -> None:
        """Test creating balanced ranker."""
        ranker = create_balanced_ranker()

        assert ranker.config.strategy == RankingStrategy.BALANCED
        assert ranker.config.min_confidence == 0.6


class TestActionabilityScoring:
    """Test actionability scoring for suggestions."""

    def test_high_actionability_suggestion(
        self, critical_issue: Any, high_quality_suggestion: Any
    ) -> None:
        """Test scoring high actionability suggestion."""
        critical_issue.rich_suggestion = high_quality_suggestion

        ranker = SuggestionRanker()
        score = ranker._calculate_actionability_score(high_quality_suggestion)

        # Should get high score for copy-paste ready + high confidence
        assert (
            score > 3.0
        )  # 2.0 for copy-paste + 1.5 for high confidence + 1.0 for short

    def test_low_actionability_suggestion(self, low_quality_suggestion: Any) -> None:
        """Test scoring low actionability suggestion."""
        ranker = SuggestionRanker()
        score = ranker._calculate_actionability_score(low_quality_suggestion)

        # Should get lower score
        assert score < 2.0


class TestComplexityPenalty:
    """Test complexity penalty calculations."""

    def test_low_confidence_penalty(self, low_confidence_issue: Any) -> None:
        """Test penalty for low confidence issues."""
        config = RankingConfig(penalize_low_confidence=True)
        ranker = SuggestionRanker(config)

        penalty = ranker._calculate_complexity_penalty(low_confidence_issue)

        # Should get penalty for low confidence
        assert penalty > 0

    def test_complex_issue_type_penalty(self) -> None:
        """Test penalty for complex issue types."""
        complex_issue = EnhancedIssue(
            issue_type="description_outdated",  # Complex type
            severity="medium",
            description="Complex issue",
            suggestion="Complex fix",
            line_number=1,
            confidence=0.8,
        )

        ranker = SuggestionRanker()
        penalty = ranker._calculate_complexity_penalty(complex_issue)

        # Should get small penalty for complex type
        assert penalty >= 0.5


if __name__ == "__main__":
    pytest.main([__file__])
