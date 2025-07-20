"""
Suggestion ranking and filtering system.

Provides intelligent ranking and filtering of suggestions based on
multiple criteria including severity, confidence, impact, and user preferences.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import logging

from .integration import EnhancedIssue, EnhancedAnalysisResult
from .models import Suggestion

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Different ranking strategies."""

    SEVERITY_FIRST = "severity_first"  # Prioritize by severity
    CONFIDENCE_FIRST = "confidence_first"  # Prioritize by confidence
    IMPACT_FIRST = "impact_first"  # Prioritize by estimated impact
    BALANCED = "balanced"  # Balanced approach
    USER_PRIORITY = "user_priority"  # User-defined priorities


class FilterCriteria(Enum):
    """Filter criteria for suggestions."""

    MIN_CONFIDENCE = "min_confidence"
    MAX_SUGGESTIONS = "max_suggestions"
    SEVERITY_LEVELS = "severity_levels"
    ISSUE_TYPES = "issue_types"
    EXCLUDE_TYPES = "exclude_types"
    COPY_PASTE_READY_ONLY = "copy_paste_ready_only"


@dataclass
class RankingConfig:
    """Configuration for suggestion ranking and filtering."""

    strategy: RankingStrategy = RankingStrategy.BALANCED
    min_confidence: float = 0.5
    max_suggestions_per_function: Optional[int] = None
    max_total_suggestions: Optional[int] = None

    # Severity weights (higher = more important)
    severity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "critical": 10.0,
            "high": 7.0,
            "medium": 4.0,
            "low": 1.0,
        }
    )

    # Issue type priorities (higher = more important)
    issue_type_priorities: Dict[str, float] = field(
        default_factory=lambda: {
            "parameter_name_mismatch": 10.0,
            "parameter_missing": 9.0,
            "parameter_type_mismatch": 8.0,
            "return_type_mismatch": 7.0,
            "missing_raises": 6.0,
            "parameter_order_different": 5.0,
            "description_outdated": 4.0,
            "example_invalid": 3.0,
        }
    )

    # Filters
    allowed_severities: Optional[List[str]] = None  # If None, allow all
    allowed_issue_types: Optional[List[str]] = None  # If None, allow all
    excluded_issue_types: Optional[List[str]] = None  # Types to exclude
    copy_paste_ready_only: bool = False

    # Advanced options
    prefer_actionable: bool = True  # Prefer suggestions that are actionable
    boost_quick_fixes: bool = True  # Boost suggestions that are quick to apply
    penalize_low_confidence: bool = True  # Penalize very low confidence suggestions


@dataclass
class RankingMetrics:
    """Metrics for ranking decisions."""

    severity_score: float = 0.0
    confidence_score: float = 0.0
    impact_score: float = 0.0
    actionability_score: float = 0.0
    complexity_penalty: float = 0.0
    user_priority_bonus: float = 0.0

    @property
    def total_score(self) -> float:
        """Calculate total ranking score."""
        return (
            self.severity_score
            + self.confidence_score
            + self.impact_score
            + self.actionability_score
            + self.user_priority_bonus
            - self.complexity_penalty
        )


class SuggestionRanker:
    """Rank and filter suggestions by quality and importance."""

    def __init__(self, config: Optional[RankingConfig] = None):
        """Initialize ranker with configuration."""
        self.config = config or RankingConfig()

    def rank_suggestions(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Rank suggestions by importance and confidence."""
        if not issues:
            return []

        # Filter issues first
        filtered_issues = self._apply_filters(issues)

        # Calculate ranking scores
        for issue in filtered_issues:
            metrics = self._calculate_ranking_metrics(issue)
            issue.ranking_score = metrics.total_score

        # Sort by ranking score (higher = better)
        ranked_issues = sorted(
            filtered_issues, key=lambda x: x.ranking_score or 0, reverse=True
        )

        # Apply limits
        ranked_issues = self._apply_limits(ranked_issues)

        logger.debug(
            f"Ranked {len(ranked_issues)} suggestions from {len(issues)} total"
        )

        return ranked_issues

    def rank_analysis_results(
        self, results: List[EnhancedAnalysisResult]
    ) -> List[EnhancedAnalysisResult]:
        """Rank analysis results and their internal suggestions."""
        enhanced_results = []

        for result in results:
            # Rank suggestions within each result
            ranked_issues = self.rank_suggestions(result.issues)

            # Create new result with ranked issues
            enhanced_result = EnhancedAnalysisResult(
                matched_pair=result.matched_pair,
                issues=ranked_issues,
                used_llm=result.used_llm,
                analysis_time_ms=result.analysis_time_ms,
                cache_hit=result.cache_hit,
                suggestion_generation_time_ms=result.suggestion_generation_time_ms,
                suggestions_generated=result.suggestions_generated,
                suggestions_skipped=result.suggestions_skipped,
            )

            enhanced_results.append(enhanced_result)

        # Sort results by their best suggestion score
        enhanced_results.sort(
            key=lambda r: max([i.ranking_score or 0 for i in r.issues], default=0),
            reverse=True,
        )

        return enhanced_results

    def _calculate_ranking_metrics(self, issue: EnhancedIssue) -> RankingMetrics:
        """Calculate ranking metrics for an issue."""
        metrics = RankingMetrics()

        # Severity score
        metrics.severity_score = self.config.severity_weights.get(issue.severity, 1.0)

        # Confidence score
        metrics.confidence_score = issue.confidence * 5.0  # Scale to 0-5

        # Impact score based on issue type
        metrics.impact_score = self.config.issue_type_priorities.get(
            issue.issue_type, 1.0
        )

        # Actionability score
        if issue.rich_suggestion:
            metrics.actionability_score = self._calculate_actionability_score(
                issue.rich_suggestion
            )

        # Complexity penalty
        metrics.complexity_penalty = self._calculate_complexity_penalty(issue)

        # Apply strategy-specific adjustments
        metrics = self._apply_strategy_adjustments(metrics, issue)

        return metrics

    def _calculate_actionability_score(self, suggestion: Suggestion) -> float:
        """Calculate how actionable a suggestion is."""
        score = 0.0

        # Copy-paste ready suggestions get bonus
        if suggestion.copy_paste_ready:
            score += 2.0

        # High confidence suggestions get bonus
        if suggestion.confidence >= 0.9:
            score += 1.5
        elif suggestion.confidence >= 0.8:
            score += 1.0
        elif suggestion.confidence >= 0.7:
            score += 0.5

        # Shorter suggestions are often easier to apply
        if suggestion.suggested_text:
            line_count = len(suggestion.suggested_text.split("\n"))
            if line_count <= 5:
                score += 1.0
            elif line_count <= 10:
                score += 0.5

        return score

    def _calculate_complexity_penalty(self, issue: EnhancedIssue) -> float:
        """Calculate complexity penalty for an issue."""
        penalty = 0.0

        # Very low confidence gets penalty
        if self.config.penalize_low_confidence and issue.confidence < 0.5:
            penalty += 2.0

        # Complex issue types get small penalty
        complex_types = [
            "description_outdated",
            "example_invalid",
        ]
        if issue.issue_type in complex_types:
            penalty += 0.5

        return penalty

    def _apply_strategy_adjustments(
        self, metrics: RankingMetrics, issue: EnhancedIssue
    ) -> RankingMetrics:
        """Apply strategy-specific adjustments to metrics."""
        if self.config.strategy == RankingStrategy.SEVERITY_FIRST:
            metrics.severity_score *= 2.0
        elif self.config.strategy == RankingStrategy.CONFIDENCE_FIRST:
            metrics.confidence_score *= 2.0
        elif self.config.strategy == RankingStrategy.IMPACT_FIRST:
            metrics.impact_score *= 2.0
        elif self.config.strategy == RankingStrategy.BALANCED:
            # Default balanced approach - no adjustments needed
            pass

        return metrics

    def _apply_filters(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Apply filtering criteria to issues."""
        filtered = []

        for issue in issues:
            # Confidence filter
            if issue.confidence < self.config.min_confidence:
                continue

            # Severity filter
            if (
                self.config.allowed_severities
                and issue.severity not in self.config.allowed_severities
            ):
                continue

            # Issue type filters
            if (
                self.config.allowed_issue_types
                and issue.issue_type not in self.config.allowed_issue_types
            ):
                continue

            if (
                self.config.excluded_issue_types
                and issue.issue_type in self.config.excluded_issue_types
            ):
                continue

            # Copy-paste ready filter
            if (
                self.config.copy_paste_ready_only
                and issue.rich_suggestion
                and not issue.rich_suggestion.copy_paste_ready
            ):
                continue

            filtered.append(issue)

        return filtered

    def _apply_limits(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Apply suggestion limits."""
        if (
            self.config.max_total_suggestions
            and len(issues) > self.config.max_total_suggestions
        ):
            return issues[: self.config.max_total_suggestions]

        return issues


class SuggestionFilter:
    """Filter suggestions based on various criteria."""

    @staticmethod
    def by_confidence(
        issues: List[EnhancedIssue], min_confidence: float
    ) -> List[EnhancedIssue]:
        """Filter by minimum confidence."""
        return [issue for issue in issues if issue.confidence >= min_confidence]

    @staticmethod
    def by_severity(
        issues: List[EnhancedIssue], severities: List[str]
    ) -> List[EnhancedIssue]:
        """Filter by severity levels."""
        return [issue for issue in issues if issue.severity in severities]

    @staticmethod
    def by_issue_type(
        issues: List[EnhancedIssue], issue_types: List[str]
    ) -> List[EnhancedIssue]:
        """Filter by issue types."""
        return [issue for issue in issues if issue.issue_type in issue_types]

    @staticmethod
    def exclude_issue_types(
        issues: List[EnhancedIssue], exclude_types: List[str]
    ) -> List[EnhancedIssue]:
        """Exclude specific issue types."""
        return [issue for issue in issues if issue.issue_type not in exclude_types]

    @staticmethod
    def copy_paste_ready_only(issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Filter to only copy-paste ready suggestions."""
        return [
            issue
            for issue in issues
            if issue.rich_suggestion and issue.rich_suggestion.copy_paste_ready
        ]

    @staticmethod
    def top_n(issues: List[EnhancedIssue], n: int) -> List[EnhancedIssue]:
        """Get top N issues by ranking score."""
        sorted_issues = sorted(issues, key=lambda x: x.ranking_score or 0, reverse=True)
        return sorted_issues[:n]


class PriorityBooster:
    """Boost priority of certain suggestions based on context."""

    def __init__(self):
        """Initialize priority booster."""
        self.boost_rules: List[Callable[[EnhancedIssue], float]] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default priority boost rules."""

        def boost_critical_parameters(issue: EnhancedIssue) -> float:
            """Boost issues with critical parameter problems."""
            if issue.issue_type in ["parameter_name_mismatch", "parameter_missing"]:
                return 2.0
            return 0.0

        def boost_public_functions(issue: EnhancedIssue) -> float:
            """Boost issues in public functions (no leading underscore)."""
            # This would need access to function information
            # Simplified for now
            return 0.0

        def boost_frequently_used(issue: EnhancedIssue) -> float:
            """Boost issues in frequently used functions."""
            # This would need usage statistics
            # Simplified for now
            return 0.0

        self.boost_rules.extend(
            [boost_critical_parameters, boost_public_functions, boost_frequently_used,]
        )

    def add_boost_rule(self, rule: Callable[[EnhancedIssue], float]):
        """Add a custom boost rule."""
        self.boost_rules.append(rule)

    def calculate_boost(self, issue: EnhancedIssue) -> float:
        """Calculate total boost for an issue."""
        total_boost = 0.0
        for rule in self.boost_rules:
            try:
                boost = rule(issue)
                total_boost += boost
            except Exception as e:
                logger.warning(f"Boost rule failed: {e}")

        return total_boost


# Factory functions for common configurations
def create_strict_ranker() -> SuggestionRanker:
    """Create ranker with strict filtering."""
    config = RankingConfig(
        min_confidence=0.8,
        copy_paste_ready_only=True,
        allowed_severities=["critical", "high"],
    )
    return SuggestionRanker(config)


def create_permissive_ranker() -> SuggestionRanker:
    """Create ranker with permissive filtering."""
    config = RankingConfig(min_confidence=0.3, copy_paste_ready_only=False,)
    return SuggestionRanker(config)


def create_balanced_ranker() -> SuggestionRanker:
    """Create ranker with balanced settings."""
    config = RankingConfig(strategy=RankingStrategy.BALANCED, min_confidence=0.6,)
    return SuggestionRanker(config)
