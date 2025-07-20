"""
Integration layer connecting suggestion generation with analyzer module.

This module provides the bridge between the analyzer's issue detection
and the suggestion system's fix generation, creating enhanced analysis
results with actionable suggestions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import logging

from ..analyzer.models import AnalysisResult, InconsistencyIssue
from ..matcher import MatchedPair
from .models import Suggestion, SuggestionContext, SuggestionBatch
from .config import SuggestionConfig
from .base import BaseSuggestionGenerator
from .generators.parameter_generator import ParameterSuggestionGenerator
from .generators.return_generator import ReturnSuggestionGenerator
from .generators.raises_generator import RaisesSuggestionGenerator
from .generators.behavior_generator import BehaviorSuggestionGenerator
from .generators.example_generator import ExampleSuggestionGenerator
from .generators.edge_case_handlers import EdgeCaseSuggestionGenerator

logger = logging.getLogger(__name__)


@dataclass
class EnhancedIssue:
    """Issue enhanced with rich suggestion."""

    # Original issue fields
    issue_type: str
    severity: str
    description: str
    suggestion: str  # Basic suggestion from analyzer
    line_number: int
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    # Enhanced suggestion fields
    rich_suggestion: Optional[Suggestion] = None
    formatted_output: Optional[str] = None
    ranking_score: Optional[float] = None

    def __post_init__(self):
        """Validate enhanced issue."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if self.line_number < 1:
            raise ValueError(f"line_number must be positive, got {self.line_number}")


@dataclass
class EnhancedAnalysisResult:
    """Analysis result enhanced with rich suggestions."""

    # Original fields from AnalysisResult
    matched_pair: MatchedPair
    issues: List[EnhancedIssue] = field(default_factory=list)
    used_llm: bool = False
    analysis_time_ms: float = 0.0
    cache_hit: bool = False

    # Enhancement fields
    suggestion_generation_time_ms: float = 0.0
    suggestions_generated: int = 0
    suggestions_skipped: int = 0

    @property
    def total_time_ms(self) -> float:
        """Total time including suggestion generation."""
        return self.analysis_time_ms + self.suggestion_generation_time_ms

    @property
    def has_suggestions(self) -> bool:
        """Check if any issues have rich suggestions."""
        return any(issue.rich_suggestion for issue in self.issues)

    def get_suggestions(self) -> List[Suggestion]:
        """Get all rich suggestions."""
        return [
            issue.rich_suggestion
            for issue in self.issues
            if issue.rich_suggestion is not None
        ]


class SuggestionIntegration:
    """Integrate suggestions with analyzer output."""

    def __init__(self, config: Optional[SuggestionConfig] = None):
        """Initialize integration with configuration."""
        self.config = config or SuggestionConfig()
        self._generators: Dict[str, BaseSuggestionGenerator] = {}
        self._setup_generators()

    def _setup_generators(self):
        """Setup specialized generators for different issue types."""
        self._generators.update(
            {
                # Parameter-related issues
                "parameter_name_mismatch": ParameterSuggestionGenerator(self.config),
                "parameter_missing": ParameterSuggestionGenerator(self.config),
                "parameter_type_mismatch": ParameterSuggestionGenerator(self.config),
                "parameter_count_mismatch": ParameterSuggestionGenerator(self.config),
                "parameter_order_different": ParameterSuggestionGenerator(self.config),
                "missing_params": ParameterSuggestionGenerator(self.config),
                "undocumented_kwargs": ParameterSuggestionGenerator(self.config),
                "default_mismatches": ParameterSuggestionGenerator(self.config),
                # Return-related issues
                "return_type_mismatch": ReturnSuggestionGenerator(self.config),
                "missing_returns": ReturnSuggestionGenerator(self.config),
                # Exception-related issues
                "missing_raises": RaisesSuggestionGenerator(self.config),
                # Behavioral issues
                "description_outdated": BehaviorSuggestionGenerator(self.config),
                # Example issues
                "example_invalid": ExampleSuggestionGenerator(self.config),
            }
        )

        # Edge case handler for special constructs
        edge_case_generator = EdgeCaseSuggestionGenerator(self.config)
        self._edge_case_generator = edge_case_generator

    def enhance_analysis_result(self, result: AnalysisResult) -> EnhancedAnalysisResult:
        """Add rich suggestions to analysis results."""
        start_time = time.perf_counter()

        enhanced_issues = []
        suggestions_generated = 0
        suggestions_skipped = 0

        for issue in result.issues:
            try:
                # Create enhanced issue from original
                enhanced_issue = self._create_enhanced_issue(issue)

                # Generate suggestion if confidence meets threshold
                if issue.confidence >= self.config.confidence_threshold:
                    context = self._create_context(issue, result.matched_pair)

                    # Check for edge cases first
                    if self._is_edge_case(result.matched_pair.function):
                        suggestion = self._edge_case_generator.generate(context)
                    else:
                        suggestion = self._generate_suggestion(issue, context)

                    if suggestion:
                        enhanced_issue.rich_suggestion = suggestion
                        suggestions_generated += 1
                    else:
                        suggestions_skipped += 1
                        logger.debug(f"Skipped suggestion for {issue.issue_type}")
                else:
                    suggestions_skipped += 1
                    logger.debug(
                        f"Skipped low-confidence issue: {issue.issue_type} "
                        f"(confidence: {issue.confidence:.2f})"
                    )

                enhanced_issues.append(enhanced_issue)

            except Exception as e:
                logger.warning(f"Failed to enhance issue {issue.issue_type}: {e}")
                # Add unenhanced issue
                enhanced_issue = self._create_enhanced_issue(issue)
                enhanced_issues.append(enhanced_issue)
                suggestions_skipped += 1

        suggestion_time = (time.perf_counter() - start_time) * 1000

        return EnhancedAnalysisResult(
            matched_pair=result.matched_pair,
            issues=enhanced_issues,
            used_llm=result.used_llm,
            analysis_time_ms=result.analysis_time_ms,
            cache_hit=result.cache_hit,
            suggestion_generation_time_ms=suggestion_time,
            suggestions_generated=suggestions_generated,
            suggestions_skipped=suggestions_skipped,
        )

    def _create_enhanced_issue(self, issue: InconsistencyIssue) -> EnhancedIssue:
        """Create enhanced issue from original."""
        return EnhancedIssue(
            issue_type=issue.issue_type,
            severity=issue.severity,
            description=issue.description,
            suggestion=issue.suggestion,
            line_number=issue.line_number,
            confidence=issue.confidence,
            details=issue.details.copy(),
        )

    def _create_context(
        self, issue: InconsistencyIssue, pair: MatchedPair
    ) -> SuggestionContext:
        """Create context for suggestion generation."""
        from .models import SuggestionContext

        return SuggestionContext(
            issue=issue,
            function=pair.function,
            docstring=pair.documentation,
            project_style=self.config.default_style,
            surrounding_code=None,  # Could be enhanced later
        )

    def _generate_suggestion(
        self, issue: InconsistencyIssue, context: SuggestionContext
    ) -> Optional[Suggestion]:
        """Generate suggestion for the given issue."""
        generator = self._generators.get(issue.issue_type)
        if not generator:
            logger.debug(f"No generator for issue type: {issue.issue_type}")
            return None

        try:
            return generator.generate(context)
        except Exception as e:
            logger.warning(f"Generator failed for {issue.issue_type}: {e}")
            return None

    def _is_edge_case(self, function) -> bool:
        """Check if function requires edge case handling."""
        # Check for special decorators or constructs
        if hasattr(function, "signature") and hasattr(function.signature, "name"):
            name = function.signature.name
            # Properties, magic methods, etc.
            if name.startswith("__") and name.endswith("__"):
                return True

        # Could check for decorators in source code
        # This is a simplified version
        return False


class SuggestionBatchProcessor:
    """Process multiple analysis results with suggestions."""

    def __init__(self, config: Optional[SuggestionConfig] = None):
        """Initialize batch processor."""
        self.config = config or SuggestionConfig()
        self.integration = SuggestionIntegration(config)

    def process_batch(
        self, results: List[AnalysisResult]
    ) -> List[EnhancedAnalysisResult]:
        """Process multiple analysis results."""
        enhanced_results = []

        for result in results:
            try:
                enhanced = self.integration.enhance_analysis_result(result)
                enhanced_results.append(enhanced)
            except Exception as e:
                logger.error(f"Failed to process analysis result: {e}")
                # Create minimal enhanced result
                enhanced = EnhancedAnalysisResult(
                    matched_pair=result.matched_pair,
                    issues=[],
                    used_llm=result.used_llm,
                    analysis_time_ms=result.analysis_time_ms,
                    cache_hit=result.cache_hit,
                )
                enhanced_results.append(enhanced)

        return enhanced_results

    def create_suggestion_batch(
        self, results: List[EnhancedAnalysisResult]
    ) -> SuggestionBatch:
        """Create a suggestion batch from enhanced results."""
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.get_suggestions())

        return SuggestionBatch(
            suggestions=all_suggestions,
            total_issues=sum(len(r.issues) for r in results),
            functions_processed=len(results),
            generation_time_ms=sum(r.suggestion_generation_time_ms for r in results),
        )


# Factory functions for easy integration
def enhance_with_suggestions(
    result: AnalysisResult, config: Optional[SuggestionConfig] = None
) -> EnhancedAnalysisResult:
    """Enhance a single analysis result with suggestions."""
    integration = SuggestionIntegration(config)
    return integration.enhance_analysis_result(result)


def enhance_multiple_with_suggestions(
    results: List[AnalysisResult], config: Optional[SuggestionConfig] = None
) -> List[EnhancedAnalysisResult]:
    """Enhance multiple analysis results with suggestions."""
    processor = SuggestionBatchProcessor(config)
    return processor.process_batch(results)
