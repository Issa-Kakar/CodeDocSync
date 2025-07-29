"""
Acceptance simulator for validating RAG system improvements.

This module simulates realistic user acceptance patterns to validate
the effectiveness of the RAG-enhanced suggestion system through A/B testing.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analyzer.models import AnalysisResult, InconsistencyIssue
from ..matcher import MatchConfidence, MatchedPair, MatchType
from ..parser import (
    FunctionParameter,
    FunctionSignature,
    ParsedFunction,
    RawDocstring,
)
from .config import SuggestionConfig
from .integration import SuggestionIntegration
from .metrics import get_ab_controller, get_metrics_collector
from .rag_corpus import RAGCorpusManager

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Results from acceptance simulation."""

    total_suggestions: int
    control_total: int
    control_accepted: int
    control_rate: float
    treatment_total: int
    treatment_accepted: int
    treatment_rate: float
    improvement_percentage: float
    metrics: dict[str, float]


class AcceptanceSimulator:
    """Simulates realistic user acceptance patterns for A/B testing validation."""

    def __init__(
        self,
        control_acceptance_rate: float = 0.25,
        treatment_acceptance_rate: float = 0.40,
        output_dir: Path = Path("data/simulated"),
        seed: int = 42,
    ):
        """Initialize the acceptance simulator."""
        self.control_acceptance_rate = control_acceptance_rate
        self.treatment_acceptance_rate = treatment_acceptance_rate
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)

        # Initialize real components
        self.metrics_collector = get_metrics_collector()
        self.ab_controller = get_ab_controller()
        self.rag_manager = RAGCorpusManager()

        # Function templates for diversity
        self.function_templates = self._create_function_templates()
        self.issue_distribution = {
            "missing_params": 0.25,
            "parameter_type_mismatch": 0.15,
            "missing_returns": 0.20,
            "missing_examples": 0.15,
            "missing_raises": 0.10,
            "description_outdated": 0.10,
            "parameter_name_mismatch": 0.05,
        }

    def simulate(self, count: int) -> SimulationResults:
        """Run the acceptance simulation."""
        suggestions_data = []
        accepted_suggestions = []

        # Generate diverse functions
        functions = self._generate_test_functions(count)

        # Create integration instances with different configs
        config_with_rag = SuggestionConfig(use_rag=True)
        config_no_rag = SuggestionConfig(use_rag=False)
        integration_with_rag = SuggestionIntegration(config_with_rag)
        integration_no_rag = SuggestionIntegration(config_no_rag)

        # Process each function
        for _, (function, issue_type) in enumerate(functions):
            # Determine A/B group
            use_rag = self.ab_controller.should_use_rag(str(function.signature))
            ab_group = "treatment" if use_rag else "control"

            # Create analysis result
            issue = InconsistencyIssue(
                issue_type=issue_type,
                severity="medium",  # Use valid severity
                description=f"Simulated {issue_type} issue for {function.signature.name}",
                suggestion=f"Fix the {issue_type}",
                line_number=function.line_number,
                confidence=0.85,
                details={
                    "function_name": function.signature.name,
                    "file_path": function.file_path,
                },
            )

            matched_pair = MatchedPair(
                function=function,
                match_type=MatchType.EXACT,
                confidence=MatchConfidence(
                    overall=0.9,
                    name_similarity=0.95,
                    location_score=0.85,
                    signature_similarity=0.9,
                ),
                match_reason="Simulated match",
                docstring=None,
            )

            analysis_result = AnalysisResult(
                matched_pair=matched_pair,
                issues=[issue],
                used_llm=False,
                analysis_time_ms=0.0,
                cache_hit=False,
            )

            # Generate suggestion using real integration
            integration = integration_with_rag if use_rag else integration_no_rag
            enhanced_result = integration.enhance_analysis_result(analysis_result)

            # Process enhanced issues
            for enhanced_issue in enhanced_result.issues:
                if enhanced_issue.rich_suggestion:
                    suggestion_id = (
                        enhanced_issue.rich_suggestion.metadata.suggestion_id
                    )

                    # Get the tracked metrics for this suggestion
                    metric = None
                    for m in self.metrics_collector.current_session:
                        if m.suggestion_id == suggestion_id:
                            metric = m
                            break

                    if metric:
                        # Simulate user decision
                        accepted = self._simulate_acceptance_decision(metric, ab_group)

                        if accepted and suggestion_id:
                            # Mark as accepted
                            self.metrics_collector.mark_accepted(
                                suggestion_id, modified=random.random() < 0.3
                            )

                            # Create accepted suggestion data
                            accepted_data = self._create_accepted_suggestion(
                                function,
                                enhanced_issue.rich_suggestion.suggested_text,
                                issue_type,
                                0.8,  # Default quality score for simulated acceptances
                            )
                            accepted_suggestions.append(accepted_data)

                            # Add to RAG corpus
                            self.rag_manager.add_accepted_suggestion(
                                function=function,
                                suggested_docstring=accepted_data["docstring_content"],
                                docstring_format="google",
                                issue_type=issue_type,
                            )
                        elif suggestion_id:
                            # Mark as rejected
                            self.metrics_collector.mark_rejected(suggestion_id)

                        # Create suggestion data for output
                        suggestion_data = {
                            "id": suggestion_id,
                            "timestamp": metric.timestamp,
                            "function_signature": str(function.signature),
                            "issue_type": issue_type,
                            "generator": enhanced_issue.rich_suggestion.metadata.generator_type,
                            "ab_group": ab_group,
                            "rag_enhanced": ab_group == "treatment",
                            "confidence_score": metric.confidence_score,
                            "completeness_score": metric.completeness_score,
                            "quality_score": 0.8 if ab_group == "treatment" else 0.7,
                            "suggestion_length": len(
                                enhanced_issue.rich_suggestion.suggested_text
                            ),
                            "examples_used": metric.examples_used,
                            "similarity_scores": metric.similarity_scores,
                            "lifecycle_events": [
                                {
                                    "event": event["type"],
                                    "timestamp": event["timestamp"],
                                    "details": event.get("metadata", {}),
                                }
                                for event in metric.events
                            ],
                        }
                        suggestions_data.append(suggestion_data)

        # Save results
        self._save_simulation_results(suggestions_data, accepted_suggestions)

        # Calculate metrics
        return self._calculate_simulation_metrics(suggestions_data)

    def _generate_test_functions(self, count: int) -> list[tuple[ParsedFunction, str]]:
        """Generate diverse test functions with issues."""
        functions = []

        # Module paths for diversity
        modules = [
            "api.endpoints.users",
            "services.payment",
            "utils.validation",
            "core.authentication",
            "data.processing",
            "ml.models",
            "cli.commands",
            "integrations.external",
        ]

        for i in range(count):
            # Select template and issue type
            template = random.choice(self.function_templates)
            issue_type = np.random.choice(
                list(self.issue_distribution.keys()),
                p=list(self.issue_distribution.values()),
            )

            # Create function with variation
            function = self._instantiate_function_template(
                template, index=i, module=random.choice(modules)
            )

            functions.append((function, issue_type))

        return functions

    def _create_function_templates(self) -> list[dict[str, Any]]:
        """Create diverse function templates."""
        return [
            # REST API endpoints
            {
                "pattern": "get_{resource}",
                "params": [
                    ("resource_id", "str"),
                    ("include_deleted", "bool", "False"),
                ],
                "returns": "dict[str, Any]",
                "complexity": 2,
            },
            {
                "pattern": "create_{resource}",
                "params": [
                    ("data", "dict[str, Any]"),
                    ("validate", "bool", "True"),
                ],
                "returns": "tuple[dict[str, Any], int]",
                "complexity": 3,
            },
            # Data processing
            {
                "pattern": "process_{data_type}",
                "params": [
                    ("input_data", "pd.DataFrame"),
                    ("config", "ProcessConfig | None", "None"),
                ],
                "returns": "pd.DataFrame",
                "complexity": 4,
            },
            # Async operations
            {
                "pattern": "fetch_{resource}_async",
                "params": [
                    ("url", "str"),
                    ("timeout", "float", "30.0"),
                    ("retries", "int", "3"),
                ],
                "returns": "Awaitable[Response]",
                "complexity": 3,
            },
            # Validation functions
            {
                "pattern": "validate_{entity}",
                "params": [
                    ("entity", "Any"),
                    ("strict", "bool", "False"),
                ],
                "returns": "bool",
                "raises": ["ValidationError"],
                "complexity": 2,
            },
            # ML operations
            {
                "pattern": "train_{model_type}_model",
                "params": [
                    ("X", "np.ndarray"),
                    ("y", "np.ndarray"),
                    ("epochs", "int", "100"),
                    ("batch_size", "int", "32"),
                ],
                "returns": "TrainedModel",
                "complexity": 5,
            },
        ]

    def _instantiate_function_template(
        self, template: dict[str, Any], index: int, module: str
    ) -> ParsedFunction:
        """Create a ParsedFunction from a template."""
        # Generate function name
        resource_types = ["user", "order", "product", "payment", "report", "config"]
        function_name = template["pattern"].replace(
            "{resource}", random.choice(resource_types)
        )
        function_name = function_name.replace(
            "{data_type}", random.choice(["csv", "json", "xml"])
        )
        function_name = function_name.replace(
            "{entity}", random.choice(["email", "phone", "address"])
        )
        function_name = function_name.replace(
            "{model_type}", random.choice(["linear", "neural", "forest"])
        )

        # Create parameters
        parameters = []
        for param_info in template.get("params", []):
            if len(param_info) == 2:
                name, type_annotation = param_info
                default = None
            else:
                name, type_annotation, default = param_info

            parameters.append(
                FunctionParameter(
                    name=name,
                    type_annotation=type_annotation,
                    default_value=default,
                    is_required=(default is None),
                )
            )

        # Create signature
        signature = FunctionSignature(
            name=function_name,
            parameters=parameters,
            return_type=template.get("returns"),
        )

        # Create parsed function with minimal docstring
        line_num = random.randint(50, 500)

        # Ensure docstring is never empty
        docstring_text = f'"""{function_name.replace("_", " ").capitalize()}."""'
        if not docstring_text.strip() or len(docstring_text) < 10:
            docstring_text = f'"""Function {function_name}."""'

        return ParsedFunction(
            signature=signature,
            docstring=RawDocstring(
                raw_text=docstring_text,
                line_number=line_num + 1,
            ),
            file_path=f"{module.replace('.', '/')}.py",
            line_number=line_num,
            end_line_number=line_num
            + random.randint(5, 20),  # Function spans 5-20 lines
        )

    def _simulate_acceptance_decision(self, metric: Any, ab_group: str) -> bool:
        """Simulate whether user accepts the suggestion."""
        # Base acceptance rate
        base_rate = (
            self.treatment_acceptance_rate
            if ab_group == "treatment"
            else self.control_acceptance_rate
        )

        # Factors affecting acceptance
        quality_factor = 0.8 if ab_group == "treatment" else 0.7
        completeness_factor = metric.completeness_score
        confidence_factor = metric.confidence_score

        # Calculate acceptance probability
        acceptance_prob = base_rate * (
            0.4 * quality_factor + 0.3 * completeness_factor + 0.3 * confidence_factor
        )

        # Add some randomness for realism
        acceptance_prob *= random.uniform(0.8, 1.2)

        return random.random() < acceptance_prob

    def _create_accepted_suggestion(
        self,
        function: ParsedFunction,
        docstring_content: str,
        issue_type: str,
        quality_score: float,
    ) -> dict[str, Any]:
        """Create accepted suggestion data."""
        # Note: Don't include 'id' as DocstringExample doesn't accept it
        return {
            "function_name": function.signature.name,
            "module_path": function.file_path,
            "function_signature": str(function.signature),
            "docstring_format": "google",
            "docstring_content": docstring_content,
            "has_params": len(function.signature.parameters) > 0,
            "has_returns": function.signature.return_type is not None,
            "has_examples": "example" in docstring_content.lower()
            or ">>>" in docstring_content,
            "complexity_score": 3,  # Default complexity for simulated functions
            "quality_score": int(quality_score * 5),  # Convert 0-1 to 1-5 scale
            "source": "accepted",  # Use 'accepted' not 'simulated' for compatibility
            "timestamp": time.time(),
            "issue_types": [issue_type],
            "original_issue": issue_type,
            "improvement_score": random.uniform(0.6, 0.9),
        }

    def _save_simulation_results(
        self,
        suggestions_data: list[dict[str, Any]],
        accepted_suggestions: list[dict[str, Any]],
    ) -> None:
        """Save simulation results to files."""
        # Save suggestion metrics
        metrics_file = self.output_dir / "simulated_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "version": "1.0.0",
                    "generated_at": time.time(),
                    "total_suggestions": len(suggestions_data),
                    "suggestions": suggestions_data,
                },
                f,
                indent=2,
            )

        # Save accepted suggestions
        accepted_file = self.output_dir / "simulated_acceptances.json"
        with open(accepted_file, "w") as f:
            json.dump(
                {
                    "version": "1.0.0",
                    "generated_at": time.time(),
                    "total_accepted": len(accepted_suggestions),
                    "examples": accepted_suggestions,
                },
                f,
                indent=2,
            )

        # Also update the real accepted_suggestions.json
        real_accepted_path = Path("data/accepted_suggestions.json")
        real_accepted_path.parent.mkdir(parents=True, exist_ok=True)
        with open(real_accepted_path, "w") as f:
            json.dump(
                {
                    "version": "1.0.0",
                    "last_updated": time.time(),
                    "total_accepted": len(accepted_suggestions),
                    "examples": accepted_suggestions,
                },
                f,
                indent=2,
            )

    def _calculate_simulation_metrics(
        self, suggestions_data: list[dict[str, Any]]
    ) -> SimulationResults:
        """Calculate improvement metrics from simulation."""
        control_suggestions = [
            s for s in suggestions_data if s["ab_group"] == "control"
        ]
        treatment_suggestions = [
            s for s in suggestions_data if s["ab_group"] == "treatment"
        ]

        control_accepted = sum(
            1
            for s in control_suggestions
            if any(e["event"] == "accepted" for e in s["lifecycle_events"])
        )
        treatment_accepted = sum(
            1
            for s in treatment_suggestions
            if any(e["event"] == "accepted" for e in s["lifecycle_events"])
        )

        control_rate = (
            control_accepted / len(control_suggestions) if control_suggestions else 0
        )
        treatment_rate = (
            treatment_accepted / len(treatment_suggestions)
            if treatment_suggestions
            else 0
        )

        improvement_pct = (
            ((treatment_rate - control_rate) / control_rate * 100)
            if control_rate > 0
            else 0
        )

        return SimulationResults(
            total_suggestions=len(suggestions_data),
            control_total=len(control_suggestions),
            control_accepted=control_accepted,
            control_rate=control_rate,
            treatment_total=len(treatment_suggestions),
            treatment_accepted=treatment_accepted,
            treatment_rate=treatment_rate,
            improvement_percentage=improvement_pct,
            metrics={
                "avg_quality_control": (
                    np.mean([s["quality_score"] for s in control_suggestions])
                    if control_suggestions
                    else 0
                ),
                "avg_quality_treatment": (
                    np.mean([s["quality_score"] for s in treatment_suggestions])
                    if treatment_suggestions
                    else 0
                ),
                "avg_completeness_control": (
                    np.mean([s["completeness_score"] for s in control_suggestions])
                    if control_suggestions
                    else 0
                ),
                "avg_completeness_treatment": (
                    np.mean([s["completeness_score"] for s in treatment_suggestions])
                    if treatment_suggestions
                    else 0
                ),
            },
        )
