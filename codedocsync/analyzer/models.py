"""
Data models for the analyzer module.

CRITICAL: These models define the contract between the analyzer and other modules.
All fields must be validated and documented.
"""

from dataclasses import dataclass, field
from typing import Any

from codedocsync.matcher import MatchedPair

# Issue type constants with default severities
ISSUE_TYPES = {
    "parameter_name_mismatch": "critical",
    "parameter_missing": "critical",
    "parameter_type_mismatch": "high",
    "return_type_mismatch": "high",
    "missing_raises": "medium",
    "parameter_order_different": "medium",
    "description_outdated": "medium",
    "example_invalid": "low",
    "missing_params": "critical",
    "missing_returns": "high",
    "undocumented_kwargs": "medium",
    "type_mismatches": "high",
    "default_mismatches": "medium",
    "parameter_count_mismatch": "critical",
}

# Severity weights for sorting (higher = more critical)
SEVERITY_WEIGHTS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

# Confidence thresholds for routing decisions
CONFIDENCE_THRESHOLDS = {
    "HIGH_CONFIDENCE": 0.9,  # Skip LLM if rule confidence >= this
    "MEDIUM_CONFIDENCE": 0.7,  # Requires LLM review
    "LOW_CONFIDENCE": 0.5,  # Always requires LLM
}


@dataclass
class InconsistencyIssue:
    """Single documentation inconsistency."""

    issue_type: str  # Must be from ISSUE_TYPES
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str  # Human-readable description
    suggestion: str  # Actionable fix suggestion
    line_number: int  # Line where issue occurs
    confidence: float = 1.0  # 0.0-1.0, determines if LLM needed
    details: dict[str, Any] = field(default_factory=dict)  # Additional context

    def __post_init__(self):
        """Validate issue fields."""
        # Validate issue_type
        if self.issue_type not in ISSUE_TYPES:
            raise ValueError(
                f"issue_type must be one of {list(ISSUE_TYPES.keys())}, "
                f"got '{self.issue_type}'"
            )

        # Validate severity
        valid_severities = ["critical", "high", "medium", "low"]
        if self.severity not in valid_severities:
            raise ValueError(
                f"severity must be one of {valid_severities}, got '{self.severity}'"
            )

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate line_number
        if self.line_number < 1:
            raise ValueError(f"line_number must be positive, got {self.line_number}")

        # Validate required strings are not empty
        if not self.description.strip():
            raise ValueError("description cannot be empty")

        if not self.suggestion.strip():
            raise ValueError("suggestion cannot be empty")

    @property
    def severity_weight(self) -> int:
        """Get numeric weight for severity sorting."""
        return SEVERITY_WEIGHTS[self.severity]

    def is_high_confidence(self) -> bool:
        """Check if this issue has high confidence."""
        return self.confidence >= CONFIDENCE_THRESHOLDS["HIGH_CONFIDENCE"]


@dataclass
class RuleCheckResult:
    """Result from a single rule check."""

    rule_name: str  # String identifier for the rule
    passed: bool  # Whether the rule passed
    confidence: float  # Float indicating certainty
    issues: list[InconsistencyIssue] = field(default_factory=list)
    execution_time_ms: float = 0.0  # Performance tracking

    def __post_init__(self):
        """Validate rule check result."""
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate execution_time_ms
        if self.execution_time_ms < 0:
            raise ValueError(
                f"execution_time_ms must be non-negative, got {self.execution_time_ms}"
            )

        # Validate rule_name is not empty
        if not self.rule_name.strip():
            raise ValueError("rule_name cannot be empty")

        # Logical validation: if passed=True, should have no issues
        if self.passed and self.issues:
            raise ValueError(
                f"Rule '{self.rule_name}' passed but has {len(self.issues)} issues"
            )


@dataclass
class AnalysisResult:
    """Complete analysis result for a matched pair."""

    matched_pair: MatchedPair
    issues: list[InconsistencyIssue] = field(default_factory=list)
    used_llm: bool = False
    analysis_time_ms: float = 0.0
    rule_results: list[RuleCheckResult] | None = None
    cache_hit: bool = False

    def __post_init__(self):
        """Validate analysis result."""
        # Validate analysis_time_ms
        if self.analysis_time_ms < 0:
            raise ValueError(
                f"analysis_time_ms must be non-negative, got {self.analysis_time_ms}"
            )

    @property
    def critical_issues(self) -> list[InconsistencyIssue]:
        """Get all critical issues."""
        return [issue for issue in self.issues if issue.severity == "critical"]

    @property
    def high_issues(self) -> list[InconsistencyIssue]:
        """Get all high severity issues."""
        return [issue for issue in self.issues if issue.severity == "high"]

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.critical_issues) > 0

    def get_issues_by_severity(self) -> dict[str, list[InconsistencyIssue]]:
        """Group issues by severity."""
        result = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def get_sorted_issues(self) -> list[InconsistencyIssue]:
        """Get issues sorted by severity (critical first)."""
        return sorted(self.issues, key=lambda x: x.severity_weight, reverse=True)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for this analysis."""
        issues_by_severity = self.get_issues_by_severity()
        return {
            "total_issues": len(self.issues),
            "critical": len(issues_by_severity["critical"]),
            "high": len(issues_by_severity["high"]),
            "medium": len(issues_by_severity["medium"]),
            "low": len(issues_by_severity["low"]),
            "used_llm": self.used_llm,
            "analysis_time_ms": self.analysis_time_ms,
            "cache_hit": self.cache_hit,
            "function_name": self.matched_pair.function.signature.name,
            "file_path": self.matched_pair.function.file_path,
        }


# Rule categories that MUST be implemented
RULE_CATEGORIES = {
    "structural": [
        "parameter_names",
        "parameter_types",
        "parameter_count",
        "return_type",
    ],
    "completeness": [
        "missing_params",
        "missing_returns",
        "missing_raises",
        "undocumented_kwargs",
    ],
    "consistency": ["type_mismatches", "default_mismatches", "parameter_order"],
}
