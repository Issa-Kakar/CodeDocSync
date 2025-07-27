"""
Integration layer connecting suggestion generation with analyzer module.

This module provides the bridge between the analyzer's issue detection
and the suggestion system's fix generation, creating enhanced analysis
results with actionable suggestions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from codedocsync.parser import ParsedFunction

from ..analyzer.models import AnalysisResult, InconsistencyIssue
from ..matcher import MatchedPair
from .base import BaseSuggestionGenerator
from .config import SuggestionConfig
from .generators.behavior_generator import BehaviorSuggestionGenerator
from .generators.edge_case_handlers import EdgeCaseSuggestionGenerator
from .generators.example_generator import ExampleSuggestionGenerator
from .generators.parameter_generator import ParameterSuggestionGenerator
from .generators.raises_generator import RaisesSuggestionGenerator
from .generators.return_generator import ReturnSuggestionGenerator
from .models import Suggestion, SuggestionBatch, SuggestionContext
from .rag_corpus import RAGCorpusManager

logger = logging.getLogger(__name__)

# Global RAG manager instance (singleton pattern)
_suggestion_rag_manager = None


def _get_rag_manager() -> RAGCorpusManager:
    """Get singleton RAG manager instance."""
    global _suggestion_rag_manager
    if _suggestion_rag_manager is None:
        _suggestion_rag_manager = RAGCorpusManager()
    return _suggestion_rag_manager


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
    details: dict[str, Any] = field(default_factory=dict)

    # Enhanced suggestion fields
    rich_suggestion: Suggestion | None = None
    formatted_output: str | None = None
    ranking_score: float | None = None

    def __post_init__(self) -> None:
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
    issues: list[EnhancedIssue] = field(default_factory=list)
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

    def get_suggestions(self) -> list[Suggestion]:
        """Get all rich suggestions."""
        return [
            issue.rich_suggestion
            for issue in self.issues
            if issue.rich_suggestion is not None
        ]


class SuggestionIntegration:
    """Integrate suggestions with analyzer output."""

    def __init__(self, config: SuggestionConfig | None = None) -> None:
        """Initialize integration with configuration."""
        self.config = config or SuggestionConfig()
        self._generators: dict[str, BaseSuggestionGenerator] = {}
        self._setup_generators()

    def _setup_generators(self) -> None:
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
                # Behavioral issues - expand the existing single mapping
                "description_outdated": BehaviorSuggestionGenerator(self.config),
                "description_vague": BehaviorSuggestionGenerator(self.config),
                "missing_behavior_description": BehaviorSuggestionGenerator(
                    self.config
                ),
                "side_effects_undocumented": BehaviorSuggestionGenerator(self.config),
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
                    suggestion: Suggestion | None
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

        # Retrieve RAG examples if enabled
        related_functions = []
        if self.config.use_rag:
            try:
                import time

                start_time = time.time()

                # Get RAG corpus manager instance
                rag_manager = _get_rag_manager()

                # Retrieve similar examples
                examples = rag_manager.retrieve_examples(
                    function=pair.function,
                    n_results=3,
                    min_similarity=self.config.rag_min_similarity,
                )

                retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

                # Convert to format expected by generators
                for example in examples:
                    related_functions.append(
                        {
                            "signature": example.function_signature,
                            "docstring": example.docstring_content,
                            "similarity": getattr(example, "similarity_score", 0.8),
                        }
                    )

                if examples:
                    similarities = [
                        f"{getattr(e, 'similarity_score', 0):.2f}" for e in examples[:3]
                    ]
                    logger.info(
                        f"RAG: Retrieved {len(examples)} examples with similarities: {similarities}"
                    )
                    # Enhanced success logging with function name and best similarity
                    logger.info(
                        f"RAG: Using {len(examples)} examples for {pair.function.signature.name} "
                        f"(best similarity: {getattr(examples[0], 'similarity_score', 0):.3f})"
                    )
                    logger.info(
                        f"Retrieved {len(related_functions)} RAG examples for "
                        f"{issue.issue_type} in {pair.function.signature.name} "
                        f"(took {retrieval_time:.1f}ms)"
                    )
                else:
                    logger.info(
                        f"RAG: No suitable examples found for {pair.function.signature.name}"
                    )
            except Exception as e:
                logger.debug(f"RAG retrieval failed (non-critical): {e}")

        return SuggestionContext(
            issue=issue,
            function=pair.function,
            docstring=pair.docstring,
            project_style=self.config.default_style,
            surrounding_code=None,  # Could be enhanced later
            related_functions=related_functions,  # Now populated with RAG examples!
        )

    def _generate_suggestion(
        self, issue: InconsistencyIssue, context: SuggestionContext
    ) -> Suggestion | None:
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

    def _is_edge_case(self, function: ParsedFunction) -> bool:
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

    def __init__(self, config: SuggestionConfig | None = None) -> None:
        """Initialize batch processor."""
        self.config = config or SuggestionConfig()
        self.integration = SuggestionIntegration(config)

    def process_batch(
        self, results: list[AnalysisResult]
    ) -> list[EnhancedAnalysisResult]:
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
        self, results: list[EnhancedAnalysisResult]
    ) -> SuggestionBatch:
        """Create a suggestion batch from enhanced results."""
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.get_suggestions())

        # Get function name and file path from first result if available
        function_name = ""
        file_path = ""
        if results and results[0].matched_pair:
            # Safely access function name and file path
            if hasattr(results[0].matched_pair, "function"):
                func = results[0].matched_pair.function
                if hasattr(func, "signature") and hasattr(func.signature, "name"):
                    function_name = func.signature.name
                if hasattr(func, "file_path"):
                    file_path = func.file_path

        return SuggestionBatch(
            suggestions=all_suggestions,
            function_name=function_name,
            file_path=file_path,
            total_generation_time_ms=sum(
                r.suggestion_generation_time_ms for r in results
            ),
        )


# Factory functions for easy integration
def enhance_with_suggestions(
    result: AnalysisResult, config: SuggestionConfig | None = None
) -> EnhancedAnalysisResult:
    """Enhance a single analysis result with suggestions."""
    integration = SuggestionIntegration(config)
    return integration.enhance_analysis_result(result)


def enhance_multiple_with_suggestions(
    results: list[AnalysisResult], config: SuggestionConfig | None = None
) -> list[EnhancedAnalysisResult]:
    """Enhance multiple analysis results with suggestions."""
    processor = SuggestionBatchProcessor(config)
    return processor.process_batch(results)
