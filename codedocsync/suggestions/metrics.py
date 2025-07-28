"""
Metrics collection and analysis for suggestion improvement tracking.

This module provides comprehensive tracking of suggestion quality and acceptance
rates to measure RAG enhancement effectiveness.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Type aliases following codebase patterns
ExperimentVariant = Literal["control", "treatment"]
SuggestionEvent = Literal["generated", "displayed", "accepted", "rejected", "modified"]


@dataclass
class SuggestionMetrics:
    """Track suggestion quality and acceptance metrics."""

    # Unique identifier
    suggestion_id: str
    timestamp: float

    # Context
    function_signature: str
    file_path: str
    issue_type: str
    generator_used: str

    # RAG tracking
    rag_enhanced: bool
    examples_used: int

    # Quality indicators
    suggestion_length: int
    completeness_score: float  # 0-1, based on sections present
    confidence_score: float

    # Experiment tracking
    ab_group: ExperimentVariant

    # Fields with defaults must come after fields without defaults
    similarity_scores: list[float] = field(default_factory=list)

    # Lifecycle tracking
    events: list[dict[str, Any]] = field(default_factory=list)
    shown_to_user: bool = False
    accepted: bool = False
    modified_before_accept: bool = False
    time_to_decision: float | None = None

    def add_event(
        self, event_type: SuggestionEvent, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a lifecycle event."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.events.append(event)

        # Update flags based on event
        if event_type == "displayed":
            self.shown_to_user = True
        elif event_type == "accepted":
            self.accepted = True
            self.time_to_decision = time.time() - self.timestamp
        elif event_type == "modified":
            self.modified_before_accept = True


class MetricsCollector:
    """Collect and analyze suggestion improvement metrics."""

    def __init__(self, metrics_dir: Path | None = None):
        """Initialize metrics collector with storage directory."""
        self.metrics_dir = metrics_dir or Path("data/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: list[SuggestionMetrics] = []
        self.session_start = time.time()

        # Load historical metrics
        self._load_historical_metrics()

    def _load_historical_metrics(self) -> None:
        """Load metrics from previous sessions."""
        metrics_file = self.metrics_dir / "suggestion_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    data = json.load(f)
                    # Only load recent metrics (last 7 days)
                    cutoff = time.time() - (7 * 86400)
                    self.historical_metrics = [
                        SuggestionMetrics(**m)
                        for m in data.get("metrics", [])
                        if m["timestamp"] > cutoff
                    ]
            except Exception:
                # Start fresh if loading fails
                self.historical_metrics = []
        else:
            self.historical_metrics = []

    def track_suggestion(
        self,
        function_signature: str,
        file_path: str,
        issue_type: str,
        generator_name: str,
        suggestion_text: str,
        confidence: float,
        rag_examples: list[dict[str, Any]] | None = None,
        ab_group: ExperimentVariant | None = None,
    ) -> str:
        """Track a generated suggestion and return its ID."""
        suggestion_id = str(uuid.uuid4())

        # Calculate completeness based on suggestion content
        completeness = self._calculate_completeness(suggestion_text, issue_type)

        # Determine A/B group if not specified
        if ab_group is None:
            ab_group = "treatment" if rag_examples else "control"

        metric = SuggestionMetrics(
            suggestion_id=suggestion_id,
            timestamp=time.time(),
            function_signature=function_signature,
            file_path=file_path,
            issue_type=issue_type,
            generator_used=generator_name,
            rag_enhanced=bool(rag_examples),
            examples_used=len(rag_examples) if rag_examples else 0,
            similarity_scores=[
                ex.get("similarity", 0.0) for ex in (rag_examples or [])
            ],
            suggestion_length=len(suggestion_text),
            completeness_score=completeness,
            confidence_score=confidence,
            ab_group=ab_group,
        )

        # Add generation event
        metric.add_event(
            "generated",
            {"generator": generator_name, "rag_enhanced": bool(rag_examples)},
        )

        self.current_session.append(metric)
        return suggestion_id

    def mark_displayed(self, suggestion_id: str) -> None:
        """Mark suggestion as displayed to user."""
        metric = self._find_metric(suggestion_id)
        if metric:
            metric.add_event("displayed")

    def mark_accepted(self, suggestion_id: str, modified: bool = False) -> None:
        """Mark suggestion as accepted by user."""
        metric = self._find_metric(suggestion_id)
        if metric:
            if modified:
                metric.add_event("modified")
            metric.add_event("accepted")

    def mark_rejected(self, suggestion_id: str) -> None:
        """Mark suggestion as rejected by user."""
        metric = self._find_metric(suggestion_id)
        if metric:
            metric.add_event("rejected")

    def _find_metric(self, suggestion_id: str) -> SuggestionMetrics | None:
        """Find metric by suggestion ID."""
        for metric in self.current_session:
            if metric.suggestion_id == suggestion_id:
                return metric
        return None

    def _calculate_completeness(self, suggestion_text: str, issue_type: str) -> float:
        """Calculate how complete a suggestion is based on content."""
        score = 0.0

        # Base score for having content
        if suggestion_text.strip():
            score += 0.2

        # Check for key sections based on issue type
        text_lower = suggestion_text.lower()

        if "args:" in text_lower or "arguments:" in text_lower:
            score += 0.2
        if "returns:" in text_lower or "return:" in text_lower:
            score += 0.2
        if "raises:" in text_lower or "raise:" in text_lower:
            score += 0.1
        if "example:" in text_lower or ">>>" in suggestion_text:
            score += 0.15
        if len(suggestion_text) > 100:  # Reasonable length
            score += 0.15

        return min(score, 1.0)

    def save_session_metrics(self) -> None:
        """Persist current session metrics to disk."""
        if not self.current_session:
            return

        metrics_file = self.metrics_dir / "suggestion_metrics.json"

        # Prepare data for serialization
        all_metrics = self.historical_metrics + self.current_session
        data = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "total_suggestions": len(all_metrics),
            "metrics": [
                {
                    "suggestion_id": m.suggestion_id,
                    "timestamp": m.timestamp,
                    "function_signature": m.function_signature,
                    "file_path": m.file_path,
                    "issue_type": m.issue_type,
                    "generator_used": m.generator_used,
                    "rag_enhanced": m.rag_enhanced,
                    "examples_used": m.examples_used,
                    "similarity_scores": m.similarity_scores,
                    "suggestion_length": m.suggestion_length,
                    "completeness_score": m.completeness_score,
                    "confidence_score": m.confidence_score,
                    "ab_group": m.ab_group,
                    "events": m.events,
                    "shown_to_user": m.shown_to_user,
                    "accepted": m.accepted,
                    "modified_before_accept": m.modified_before_accept,
                    "time_to_decision": m.time_to_decision,
                }
                for m in all_metrics
            ],
        }

        # Atomic write
        temp_file = metrics_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(metrics_file)


class ABTestController:
    """Control A/B testing for RAG enhancement experiments."""

    def __init__(self, treatment_percentage: float = 0.5, seed: int | None = None):
        """Initialize A/B test controller.

        Args:
            treatment_percentage: Percentage of items to assign to treatment group (0-1)
            seed: Random seed for reproducible assignment
        """
        self.treatment_percentage = treatment_percentage
        self.assignments: dict[str, ExperimentVariant] = {}

        # Use deterministic assignment based on function signature hash
        # This ensures the same function always gets the same assignment
        self._seed = seed

    def get_assignment(self, function_signature: str) -> ExperimentVariant:
        """Get A/B test assignment for a function."""
        if function_signature not in self.assignments:
            # Use hash for deterministic assignment
            import hashlib

            hash_val = int(hashlib.md5(function_signature.encode()).hexdigest(), 16)
            if self._seed:
                hash_val += self._seed

            # Assign based on hash modulo
            use_treatment = (hash_val % 100) < (self.treatment_percentage * 100)
            self.assignments[function_signature] = (
                "treatment" if use_treatment else "control"
            )

        return self.assignments[function_signature]

    def should_use_rag(self, function_signature: str) -> bool:
        """Determine if RAG should be used for this function."""
        return self.get_assignment(function_signature) == "treatment"


class ImprovementCalculator:
    """Calculate improvement metrics for resume claims."""

    def __init__(self, metrics: list[SuggestionMetrics]):
        """Initialize with metrics to analyze."""
        self.metrics = metrics
        self._calculate_groups()

    def _calculate_groups(self) -> None:
        """Split metrics into control and treatment groups."""
        self.control_group = [m for m in self.metrics if m.ab_group == "control"]
        self.treatment_group = [m for m in self.metrics if m.ab_group == "treatment"]

    def calculate_acceptance_rates(self) -> dict[str, float]:
        """Calculate acceptance rates for each group."""

        def acceptance_rate(group: list[SuggestionMetrics]) -> float:
            if not group:
                return 0.0
            shown = [m for m in group if m.shown_to_user]
            if not shown:
                return 0.0
            accepted = [m for m in shown if m.accepted]
            return len(accepted) / len(shown)

        return {
            "control": acceptance_rate(self.control_group),
            "treatment": acceptance_rate(self.treatment_group),
            "overall": acceptance_rate(self.metrics),
        }

    def calculate_improvement_percentage(self) -> float:
        """Calculate percentage improvement of treatment over control."""
        rates = self.calculate_acceptance_rates()

        if rates["control"] == 0:
            return 0.0

        improvement = ((rates["treatment"] - rates["control"]) / rates["control"]) * 100
        return improvement

    def calculate_quality_metrics(self) -> dict[str, Any]:
        """Calculate quality metrics for each group."""

        def avg_metric(group: list[SuggestionMetrics], attr: str) -> float:
            if not group:
                return 0.0
            values = [getattr(m, attr) for m in group]
            return sum(values) / len(values) if values else 0.0

        return {
            "control": {
                "avg_completeness": avg_metric(
                    self.control_group, "completeness_score"
                ),
                "avg_confidence": avg_metric(self.control_group, "confidence_score"),
                "avg_length": avg_metric(self.control_group, "suggestion_length"),
                "avg_time_to_accept": avg_metric(
                    [m for m in self.control_group if m.time_to_decision],
                    "time_to_decision",
                ),
            },
            "treatment": {
                "avg_completeness": avg_metric(
                    self.treatment_group, "completeness_score"
                ),
                "avg_confidence": avg_metric(self.treatment_group, "confidence_score"),
                "avg_length": avg_metric(self.treatment_group, "suggestion_length"),
                "avg_time_to_accept": avg_metric(
                    [m for m in self.treatment_group if m.time_to_decision],
                    "time_to_decision",
                ),
            },
        }

    def generate_report(self) -> str:
        """Generate a comprehensive improvement report."""
        acceptance_rates = self.calculate_acceptance_rates()
        improvement_pct = self.calculate_improvement_percentage()
        quality_metrics = self.calculate_quality_metrics()

        report_lines = [
            "# RAG Enhancement Improvement Report",
            "",
            "## Executive Summary",
            f"- Total suggestions analyzed: {len(self.metrics)}",
            f"- Control group size: {len(self.control_group)}",
            f"- Treatment group size: {len(self.treatment_group)}",
            "",
        ]

        if improvement_pct > 0:
            report_lines.append(
                f"**✅ Achieved {improvement_pct:.1f}% improvement in documentation "
                f"suggestion acceptance rate through RAG enhancement**"
            )

        report_lines.extend(
            [
                "",
                "## Acceptance Rates",
                f"- Control group: {acceptance_rates['control']:.1%}",
                f"- Treatment group: {acceptance_rates['treatment']:.1%}",
                f"- Overall: {acceptance_rates['overall']:.1%}",
                "",
                "## Quality Metrics",
                "",
                "### Control Group",
                f"- Average completeness: {quality_metrics['control']['avg_completeness']:.2f}",
                f"- Average confidence: {quality_metrics['control']['avg_confidence']:.2f}",
                f"- Average length: {quality_metrics['control']['avg_length']:.0f} chars",
                f"- Average time to accept: {quality_metrics['control']['avg_time_to_accept']:.1f}s",
                "",
                "### Treatment Group (RAG-Enhanced)",
                f"- Average completeness: {quality_metrics['treatment']['avg_completeness']:.2f}",
                f"- Average confidence: {quality_metrics['treatment']['avg_confidence']:.2f}",
                f"- Average length: {quality_metrics['treatment']['avg_length']:.0f} chars",
                f"- Average time to accept: {quality_metrics['treatment']['avg_time_to_accept']:.1f}s",
            ]
        )

        # Calculate quality improvements
        completeness_imp = (
            (
                (
                    quality_metrics["treatment"]["avg_completeness"]
                    - quality_metrics["control"]["avg_completeness"]
                )
                / quality_metrics["control"]["avg_completeness"]
                * 100
            )
            if quality_metrics["control"]["avg_completeness"] > 0
            else 0
        )

        if completeness_imp > 0:
            report_lines.append(
                f"\n**✅ Increased documentation completeness by {completeness_imp:.1f}%**"
            )

        return "\n".join(report_lines)


# Module-level singleton instances
_metrics_collector: MetricsCollector | None = None
_ab_controller: ABTestController | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_ab_controller() -> ABTestController:
    """Get the singleton A/B test controller instance."""
    global _ab_controller
    if _ab_controller is None:
        _ab_controller = ABTestController()
    return _ab_controller
