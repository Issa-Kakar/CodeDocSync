"""
LLM-specific data models for semantic analysis requests and responses.

This module defines the data structures used for communication with LLM services,
including request preparation and response parsing. All models include validation
and token estimation capabilities.

Critical Validations:
- analysis_types must be from allowed list
- Token estimation is critical for avoiding context overflow
- All response fields are required (no Optional)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from codedocsync.parser import ParsedDocstring, ParsedFunction

from .models import InconsistencyIssue, RuleCheckResult

# Valid analysis types for LLM requests
VALID_ANALYSIS_TYPES = {
    "behavior": "Analyze if function behavior matches documentation",
    "examples": "Validate code examples in docstrings",
    "edge_cases": "Check if edge cases are documented",
    "version_info": "Validate version/deprecation information",
    "type_consistency": "Complex type checking beyond structural rules",
    "performance": "Validate performance claims in documentation",
}


@dataclass
class LLMAnalysisRequest:
    """Request for LLM analysis."""

    function: ParsedFunction
    docstring: ParsedDocstring
    analysis_types: list[str]  # ['behavior', 'examples', 'edge_cases']
    rule_results: list[RuleCheckResult] = field(
        default_factory=list
    )  # Context from rule engine
    related_functions: list[ParsedFunction] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate request after initialization."""
        self._validate_analysis_types()
        self._validate_function()
        self._validate_docstring()
        self._validate_related_functions()

    def _validate_analysis_types(self) -> None:
        """Validate that all analysis types are supported."""
        if not self.analysis_types:
            raise ValueError("analysis_types cannot be empty")

        for analysis_type in self.analysis_types:
            if analysis_type not in VALID_ANALYSIS_TYPES:
                raise ValueError(
                    f"analysis_type '{analysis_type}' not supported. "
                    f"Valid types: {list(VALID_ANALYSIS_TYPES.keys())}"
                )

    def _validate_function(self) -> None:
        """Validate the function object."""
        if not self.function:
            raise ValueError("function cannot be None")

        if not self.function.signature:
            raise ValueError("function must have a signature")

        if not self.function.signature.name:
            raise ValueError("function signature must have a name")

    def _validate_docstring(self) -> None:
        """Validate the docstring object."""
        if not self.docstring:
            raise ValueError("docstring cannot be None")

        if not hasattr(self.docstring, "raw_text") or not self.docstring.raw_text:
            raise ValueError("docstring must have non-empty raw_text")

    def _validate_related_functions(self) -> None:
        """Validate related functions list."""
        if len(self.related_functions) > 10:
            raise ValueError(
                f"Too many related functions ({len(self.related_functions)}). "
                "Maximum allowed is 10 to prevent context overflow."
            )

    def estimate_tokens(self) -> int:
        """
        Estimate token count for this request.

        Uses rough approximation of ~4 characters per token.
        This is critical for avoiding context overflow.

        Returns:
            Estimated number of tokens for the complete request
        """
        total_chars = 0

        # Function signature and body
        if self.function.signature:
            # Function name and parameters
            total_chars += len(self.function.signature.name)

            for param in self.function.signature.parameters:
                total_chars += len(param.name)
                if param.type_str:
                    total_chars += len(param.type_str)
                if param.default_value:
                    total_chars += len(param.default_value)

            # Return type
            if self.function.signature.return_type:
                total_chars += len(self.function.signature.return_type)

        # Docstring content
        total_chars += len(self.docstring.raw_text)

        # Rule results summary
        for rule_result in self.rule_results:
            total_chars += len(rule_result.rule_name)
            for issue in rule_result.issues:
                total_chars += len(issue.description)
                total_chars += len(issue.suggestion)

        # Related functions (signatures only)
        for related_func in self.related_functions:
            if related_func.signature:
                total_chars += len(related_func.signature.name)
                total_chars += len(str(related_func.signature.parameters))

        # Add base prompt overhead (estimated 500-1000 chars for instructions)
        total_chars += 750

        # Convert to tokens (rough approximation: 4 chars per token)
        estimated_tokens = total_chars // 4

        return estimated_tokens

    def get_function_signature_str(self) -> str:
        """Get a readable string representation of the function signature."""
        sig = self.function.signature
        params = []

        for param in sig.parameters:
            param_str = param.name
            if param.type_str:
                param_str += f": {param.type_str}"
            if param.default_value:
                param_str += f" = {param.default_value}"
            params.append(param_str)

        param_str = ", ".join(params)
        return_str = f" -> {sig.return_type}" if sig.return_type else ""

        return f"def {sig.name}({param_str}){return_str}:"

    def get_rule_issues_summary(self) -> str:
        """Get a summary of rule engine issues for context."""
        if not self.rule_results:
            return "No rule engine issues found"

        failed_rules = [r for r in self.rule_results if not r.passed]
        if not failed_rules:
            return "All rule engine checks passed"

        summaries = []
        for rule_result in failed_rules:
            rule_name = rule_result.rule_name
            issue_count = len(rule_result.issues)
            confidence = rule_result.confidence
            summaries.append(
                f"{rule_name}: {issue_count} issues (confidence: {confidence:.2f})"
            )

        return "; ".join(summaries)

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the request context for debugging."""
        return {
            "function_name": self.function.signature.name,
            "function_file": self.function.file_path,
            "function_line": self.function.line_number,
            "docstring_length": len(self.docstring.raw_text),
            "analysis_types": self.analysis_types,
            "rule_results_count": len(self.rule_results),
            "related_functions_count": len(self.related_functions),
            "estimated_tokens": self.estimate_tokens(),
        }


@dataclass
class LLMAnalysisResponse:
    """Response from LLM analysis."""

    issues: list[InconsistencyIssue]
    raw_response: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    response_time_ms: float
    cache_hit: bool = False

    def __post_init__(self) -> None:
        """Validate response after initialization."""
        self._validate_model_used()
        self._validate_tokens()
        self._validate_response_time()
        self._validate_raw_response()

    def _validate_model_used(self) -> None:
        """Validate the model identifier."""
        if not self.model_used:
            raise ValueError("model_used cannot be empty")

        # Should be in format "provider/model" or just "model"
        if not re.match(r"^[\w\-\.\/]+$", self.model_used):
            raise ValueError(f"Invalid model identifier format: {self.model_used}")

    def _validate_tokens(self) -> None:
        """Validate token counts."""
        if self.prompt_tokens < 0:
            raise ValueError(
                f"prompt_tokens must be non-negative, got {self.prompt_tokens}"
            )

        if self.completion_tokens < 0:
            raise ValueError(
                f"completion_tokens must be non-negative, got {self.completion_tokens}"
            )

        # Sanity check: total tokens should be reasonable
        total_tokens = self.prompt_tokens + self.completion_tokens
        if total_tokens > 10000:
            raise ValueError(
                f"Total tokens ({total_tokens}) seems unreasonably high. "
                "Check for potential token counting errors."
            )

    def _validate_response_time(self) -> None:
        """Validate response time."""
        if self.response_time_ms < 0:
            raise ValueError(
                f"response_time_ms must be non-negative, got {self.response_time_ms}"
            )

        # Sanity check: response time should be reasonable (under 2 minutes)
        if self.response_time_ms > 120000:
            raise ValueError(
                f"Response time ({self.response_time_ms}ms) seems unreasonably high"
            )

    def _validate_raw_response(self) -> None:
        """Validate the raw response content."""
        if not self.raw_response:
            raise ValueError("raw_response cannot be empty")

        # Check for common problematic patterns
        if len(self.raw_response) > 50000:
            raise ValueError(
                f"Raw response is too long ({len(self.raw_response)} chars). "
                "This suggests an issue with response parsing."
            )

    @property
    def total_tokens(self) -> int:
        """Get total token count (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def issues_by_severity(self) -> dict[str, list[InconsistencyIssue]]:
        """Group issues by severity level."""
        result: dict[str, list[InconsistencyIssue]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    @property
    def critical_issues_count(self) -> int:
        """Get count of critical issues."""
        return len([issue for issue in self.issues if issue.severity == "critical"])

    @property
    def high_issues_count(self) -> int:
        """Get count of high severity issues."""
        return len([issue for issue in self.issues if issue.severity == "high"])

    def estimate_cost_usd(self, model_pricing: dict[str, float] | None = None) -> float:
        """
        Estimate the cost of this API call in USD.

        Args:
            model_pricing: Optional pricing per 1K tokens for different models
                          If None, uses default pricing estimates

        Returns:
            Estimated cost in USD
        """
        if model_pricing is None:
            # Default pricing estimates (per 1K tokens)
            model_pricing = {
                "gpt-4o-mini": 0.00015,  # $0.15 per 1M tokens
                "gpt-4o": 0.003,  # $3 per 1M tokens
                "gpt-4-turbo": 0.001,  # $1 per 1M tokens
                "gpt-4": 0.003,  # $3 per 1M tokens
                "gpt-3.5-turbo": 0.0002,  # $0.2 per 1M tokens
            }

        # Extract model name from model_used (handle "provider/model" format)
        model_name = (
            self.model_used.split("/")[-1]
            if "/" in self.model_used
            else self.model_used
        )

        # Get pricing for this model (default to gpt-4o-mini pricing if unknown)
        price_per_1k = model_pricing.get(model_name, model_pricing["gpt-4o-mini"])

        # Calculate cost based on total tokens
        cost = (self.total_tokens / 1000) * price_per_1k
        return round(cost, 6)  # Round to microseconds

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for this response."""
        return {
            "response_time_ms": self.response_time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit": self.cache_hit,
            "issues_found": len(self.issues),
            "critical_issues": self.critical_issues_count,
            "high_issues": self.high_issues_count,
            "estimated_cost_usd": self.estimate_cost_usd(),
            "tokens_per_ms": (
                self.total_tokens / self.response_time_ms
                if self.response_time_ms > 0
                else 0
            ),
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of this response."""
        metrics = self.get_performance_metrics()
        return (
            f"LLM Analysis completed in {self.response_time_ms:.0f}ms "
            f"using {self.model_used}. Found {len(self.issues)} issues "
            f"({self.critical_issues_count} critical, {self.high_issues_count} high). "
            f"Used {self.total_tokens} tokens (~${metrics['estimated_cost_usd']:.4f}). "
            f"{'Cache hit' if self.cache_hit else 'Fresh analysis'}."
        )
