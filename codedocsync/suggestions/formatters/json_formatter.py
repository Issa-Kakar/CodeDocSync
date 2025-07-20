"""
JSON output formatter for suggestions.

Provides structured JSON output for machine consumption, automation,
and integration with other tools and systems.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timezone

from ..models import Suggestion, SuggestionBatch, DocstringStyle
from ..integration import EnhancedAnalysisResult, EnhancedIssue

# Version for tracking output format changes
OUTPUT_FORMAT_VERSION = "1.0.0"


class JSONSuggestionFormatter:
    """Format suggestions for JSON output."""

    def __init__(
        self,
        indent: Optional[int] = 2,
        include_metadata: bool = True,
        include_timestamps: bool = True,
    ):
        """Initialize JSON formatter."""
        self.indent = indent
        self.include_metadata = include_metadata
        self.include_timestamps = include_timestamps

    def format_suggestion(self, suggestion: Suggestion) -> Dict[str, Any]:
        """Convert suggestion to JSON-serializable format."""
        result = {
            "suggestion_type": suggestion.suggestion_type.value,
            "original_text": suggestion.original_text,
            "suggested_text": suggestion.suggested_text,
            "confidence": suggestion.confidence,
            "style": (
                suggestion.style.value
                if isinstance(suggestion.style, DocstringStyle)
                else suggestion.style
            ),
            "copy_paste_ready": suggestion.copy_paste_ready,
        }

        # Add diff information if available
        if suggestion.diff:
            result["diff"] = {
                "original_lines": suggestion.diff.original_lines,
                "suggested_lines": suggestion.diff.suggested_lines,
                "start_line": suggestion.diff.start_line,
                "end_line": suggestion.diff.end_line,
                "unified_diff": suggestion.diff.to_unified_diff(),
                "lines_changed": len(suggestion.diff.suggested_lines)
                - len(suggestion.diff.original_lines),
                "additions": len(
                    [
                        line
                        for line in suggestion.diff.suggested_lines
                        if line not in suggestion.diff.original_lines
                    ]
                ),
                "deletions": len(
                    [
                        line
                        for line in suggestion.diff.original_lines
                        if line not in suggestion.diff.suggested_lines
                    ]
                ),
            }

        # Add metadata if available and requested
        if self.include_metadata and suggestion.metadata:
            result["metadata"] = {
                "generation_time_ms": suggestion.metadata.generation_time_ms,
                "generator_used": suggestion.metadata.generator_used,
                "llm_used": suggestion.metadata.llm_used,
                "cache_hit": suggestion.metadata.cache_hit,
                "validation_passed": suggestion.metadata.validation_passed,
                "quality_score": suggestion.metadata.quality_score,
            }

        # Add timestamps if requested
        if self.include_timestamps:
            result["generated_at"] = datetime.now(timezone.utc).isoformat()

        return result

    def format_enhanced_issue(self, issue: EnhancedIssue) -> Dict[str, Any]:
        """Convert enhanced issue to JSON format."""
        result = {
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "description": issue.description,
            "basic_suggestion": issue.suggestion,  # Basic suggestion from analyzer
            "line_number": issue.line_number,
            "confidence": issue.confidence,
            "details": issue.details,
        }

        # Add rich suggestion if available
        if issue.rich_suggestion:
            result["rich_suggestion"] = self.format_suggestion(issue.rich_suggestion)

        # Add ranking information if available
        if issue.ranking_score is not None:
            result["ranking_score"] = issue.ranking_score

        # Add formatted output if available
        if issue.formatted_output:
            result["formatted_output"] = issue.formatted_output

        return result

    def format_analysis_result(self, result: EnhancedAnalysisResult) -> Dict[str, Any]:
        """Convert analysis result to JSON format."""
        output = {
            "function": self._format_function_info(result.matched_pair.function),
            "file_path": result.matched_pair.function.file_path,
            "match_confidence": result.matched_pair.confidence.value,
            "match_type": result.matched_pair.match_type.value,
            "match_reason": result.matched_pair.match_reason,
            "analysis": {
                "used_llm": result.used_llm,
                "analysis_time_ms": result.analysis_time_ms,
                "suggestion_generation_time_ms": result.suggestion_generation_time_ms,
                "total_time_ms": result.total_time_ms,
                "cache_hit": result.cache_hit,
                "suggestions_generated": result.suggestions_generated,
                "suggestions_skipped": result.suggestions_skipped,
            },
            "issues": [self.format_enhanced_issue(issue) for issue in result.issues],
            "summary": {
                "total_issues": len(result.issues),
                "has_suggestions": result.has_suggestions,
                "critical_issues": len(
                    [i for i in result.issues if i.severity == "critical"]
                ),
                "high_issues": len([i for i in result.issues if i.severity == "high"]),
                "medium_issues": len(
                    [i for i in result.issues if i.severity == "medium"]
                ),
                "low_issues": len([i for i in result.issues if i.severity == "low"]),
            },
        }

        # Add documentation info if available
        if result.matched_pair.documentation:
            output["documentation"] = {
                "format": result.matched_pair.documentation.format,
                "summary": result.matched_pair.documentation.summary,
                "parameter_count": len(result.matched_pair.documentation.parameters),
                "has_returns": result.matched_pair.documentation.returns is not None,
                "exception_count": len(result.matched_pair.documentation.raises),
            }

        # Add metadata if requested
        if self.include_metadata:
            output["metadata"] = {
                "format_version": OUTPUT_FORMAT_VERSION,
                "processor": "CodeDocSync SuggestionFormatter",
            }

            if self.include_timestamps:
                output["metadata"]["generated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

        return output

    def format_batch_results(
        self, results: List[EnhancedAnalysisResult]
    ) -> Dict[str, Any]:
        """Format multiple analysis results."""
        output = {
            "results": [self.format_analysis_result(result) for result in results],
            "summary": self._create_batch_summary(results),
        }

        if self.include_metadata:
            output["metadata"] = {
                "format_version": OUTPUT_FORMAT_VERSION,
                "processor": "CodeDocSync SuggestionFormatter",
                "batch_size": len(results),
            }

            if self.include_timestamps:
                output["metadata"]["generated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

        return output

    def format_suggestion_batch(self, batch: SuggestionBatch) -> Dict[str, Any]:
        """Format a suggestion batch."""
        output = {
            "suggestions": [
                self.format_suggestion(suggestion) for suggestion in batch.suggestions
            ],
            "summary": {
                "total_suggestions": len(batch.suggestions),
                "total_issues": batch.total_issues,
                "functions_processed": batch.functions_processed,
                "generation_time_ms": batch.generation_time_ms,
            },
        }

        # Add confidence statistics
        if batch.suggestions:
            confidences = [s.confidence for s in batch.suggestions]
            output["summary"]["confidence_stats"] = {
                "average": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "high_confidence_count": len([c for c in confidences if c >= 0.8]),
            }

            # Add suggestion type breakdown
            type_counts = {}
            for suggestion in batch.suggestions:
                suggestion_type = suggestion.suggestion_type.value
                type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
            output["summary"]["suggestion_types"] = type_counts

        if self.include_metadata:
            output["metadata"] = {
                "format_version": OUTPUT_FORMAT_VERSION,
                "processor": "CodeDocSync SuggestionFormatter",
            }

            if self.include_timestamps:
                output["metadata"]["generated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

        return output

    def to_json_string(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, indent=self.indent, ensure_ascii=False)

    def _format_function_info(self, function) -> Dict[str, Any]:
        """Extract function information for JSON."""
        info = {
            "name": "Unknown",
            "line_number": 0,
            "parameters": [],
        }

        if hasattr(function, "signature"):
            signature = function.signature
            info["name"] = signature.name
            info["line_number"] = function.line_number

            # Add parameter information
            if hasattr(signature, "parameters"):
                for param in signature.parameters:
                    param_info = {
                        "name": param.name,
                        "is_required": param.is_required,
                    }
                    if param.type_annotation:
                        param_info["type_annotation"] = param.type_annotation
                    if param.default_value:
                        param_info["default_value"] = param.default_value
                    info["parameters"].append(param_info)

            # Add return type if available
            if hasattr(signature, "return_annotation") and signature.return_annotation:
                info["return_annotation"] = signature.return_annotation

        return info

    def _create_batch_summary(
        self, results: List[EnhancedAnalysisResult]
    ) -> Dict[str, Any]:
        """Create summary statistics for batch results."""
        if not results:
            return {
                "total_functions": 0,
                "total_issues": 0,
                "total_suggestions": 0,
                "average_analysis_time_ms": 0,
                "average_suggestion_time_ms": 0,
            }

        total_issues = sum(len(r.issues) for r in results)
        total_suggestions = sum(r.suggestions_generated for r in results)
        total_analysis_time = sum(r.analysis_time_ms for r in results)
        total_suggestion_time = sum(r.suggestion_generation_time_ms for r in results)

        # Severity breakdown
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for result in results:
            for issue in result.issues:
                severity_counts[issue.severity] += 1

        # Functions with issues
        functions_with_issues = len([r for r in results if r.issues])

        # Cache hit statistics
        cache_hits = len([r for r in results if r.cache_hit])
        llm_used = len([r for r in results if r.used_llm])

        return {
            "total_functions": len(results),
            "functions_with_issues": functions_with_issues,
            "functions_without_issues": len(results) - functions_with_issues,
            "total_issues": total_issues,
            "total_suggestions": total_suggestions,
            "average_issues_per_function": total_issues / len(results),
            "average_suggestions_per_function": total_suggestions / len(results),
            "average_analysis_time_ms": total_analysis_time / len(results),
            "average_suggestion_time_ms": total_suggestion_time / len(results),
            "severity_breakdown": severity_counts,
            "cache_hit_rate": cache_hits / len(results) if results else 0,
            "llm_usage_rate": llm_used / len(results) if results else 0,
        }


# Convenience functions for common use cases
def suggestion_to_json(suggestion: Suggestion, **kwargs) -> str:
    """Convert single suggestion to JSON string."""
    formatter = JSONSuggestionFormatter(**kwargs)
    data = formatter.format_suggestion(suggestion)
    return formatter.to_json_string(data)


def analysis_result_to_json(result: EnhancedAnalysisResult, **kwargs) -> str:
    """Convert analysis result to JSON string."""
    formatter = JSONSuggestionFormatter(**kwargs)
    data = formatter.format_analysis_result(result)
    return formatter.to_json_string(data)


def batch_results_to_json(results: List[EnhancedAnalysisResult], **kwargs) -> str:
    """Convert batch results to JSON string."""
    formatter = JSONSuggestionFormatter(**kwargs)
    data = formatter.format_batch_results(results)
    return formatter.to_json_string(data)


def suggestion_batch_to_json(batch: SuggestionBatch, **kwargs) -> str:
    """Convert suggestion batch to JSON string."""
    formatter = JSONSuggestionFormatter(**kwargs)
    data = formatter.format_suggestion_batch(batch)
    return formatter.to_json_string(data)
