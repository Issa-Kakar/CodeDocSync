"""
Prompt debugging and testing utilities.

This module provides utilities for debugging LLM prompts, estimating token usage,
and testing prompt effectiveness. Essential for ensuring prompt quality and
optimizing LLM interactions.

Key Features:
- Pretty-print prompts with syntax highlighting
- Token estimation and context window analysis
- Prompt effectiveness testing
- Response validation and analysis
"""

import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codedocsync.parser import ParsedDocstring, ParsedFunction

from .llm_models import LLMAnalysisRequest
from .llm_output_parser import LLMOutputParser, ParseResult
from .prompt_templates import format_prompt, get_available_analysis_types


@dataclass
class PromptAnalysis:
    """Analysis of a prompt's characteristics."""

    prompt_text: str
    estimated_tokens: int
    character_count: int
    line_count: int
    has_examples: bool
    has_output_format: bool
    complexity_score: float
    potential_issues: list[str]


class PromptDebugger:
    """Debug and analyze LLM prompts."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize with optional Rich console."""
        self.console = console or Console()
        self.parser = LLMOutputParser(strict_validation=True)

    def debug_prompt(
        self,
        analysis_type: str,
        function: ParsedFunction,
        docstring: ParsedDocstring,
        rule_issues: str | None = None,
        show_details: bool = True,
    ) -> PromptAnalysis:
        """
        Debug a specific prompt template with sample data.

        Args:
            analysis_type: Type of analysis ('behavior_analysis', etc.)
            function: Sample function to analyze
            docstring: Sample docstring
            rule_issues: Optional rule engine issues
            show_details: Whether to print detailed analysis

        Returns:
            PromptAnalysis with detailed metrics
        """
        # Generate the prompt
        try:
            signature = self._get_function_signature_str(function)
            source_code = self._get_function_source_str(function)
            docstring_text = docstring.raw_text

            prompt = format_prompt(
                analysis_type=analysis_type,
                signature=signature,
                source_code=source_code,
                docstring=docstring_text,
                rule_issues=rule_issues or "No rule issues",
            )

            analysis = self._analyze_prompt(prompt)

            if show_details:
                self._display_prompt_analysis(analysis_type, analysis, function)

            return analysis

        except Exception as e:
            self.console.print(f"[red]Error generating prompt: {e}[/red]")
            raise

    def debug_all_prompts(
        self,
        function: ParsedFunction,
        docstring: ParsedDocstring,
        rule_issues: str | None = None,
    ) -> dict[str, PromptAnalysis]:
        """Debug all available prompt templates with the same data."""
        results = {}

        self.console.print("\n[bold blue]Debugging All Prompt Templates[/bold blue]")
        self.console.print("=" * 60)

        for analysis_type in get_available_analysis_types():
            self.console.print(f"\n[yellow]Analyzing: {analysis_type}[/yellow]")

            try:
                analysis = self.debug_prompt(
                    analysis_type=analysis_type,
                    function=function,
                    docstring=docstring,
                    rule_issues=rule_issues,
                    show_details=False,
                )
                results[analysis_type] = analysis

                # Show summary
                self._show_prompt_summary(analysis_type, analysis)

            except Exception as e:
                self.console.print(f"[red]Failed: {e}[/red]")

        # Show overall comparison
        self._show_prompts_comparison(results)

        return results

    def test_response_parsing(
        self, sample_responses: list[str], show_details: bool = True
    ) -> dict[str, Any]:
        """
        Test LLM response parsing with sample responses.

        Args:
            sample_responses: List of sample LLM responses to test
            show_details: Whether to show detailed results

        Returns:
            Dictionary with parsing statistics and results
        """
        results = []

        for i, response in enumerate(sample_responses):
            result = self.parser.parse_analysis_response(response)
            results.append(result)

            if show_details:
                self._display_parse_result(f"Response {i + 1}", result)

        # Calculate statistics
        stats = self.parser.get_parsing_statistics(results)

        if show_details:
            self._display_parsing_statistics(stats)

        return {"results": results, "statistics": stats}

    def validate_prompt_token_usage(
        self, request: LLMAnalysisRequest, max_tokens: int = 4000
    ) -> dict[str, Any]:
        """
        Validate that a request fits within token limits.

        Args:
            request: LLM analysis request to validate
            max_tokens: Maximum allowed tokens

        Returns:
            Validation results with recommendations
        """
        estimated_tokens = request.estimate_tokens()

        validation_result: dict[str, Any] = {
            "estimated_tokens": estimated_tokens,
            "max_tokens": max_tokens,
            "within_limit": estimated_tokens <= max_tokens,
            "usage_percentage": (estimated_tokens / max_tokens) * 100,
            "recommendations": [],
        }

        if estimated_tokens > max_tokens:
            overage = estimated_tokens - max_tokens
            validation_result["recommendations"].extend(
                [
                    f"Request exceeds token limit by {overage} tokens",
                    "Consider reducing context or related functions",
                    "Truncate docstring or source code if necessary",
                ]
            )
        elif estimated_tokens > max_tokens * 0.8:
            validation_result["recommendations"].append(
                "Close to token limit - consider optimization"
            )
        else:
            validation_result["recommendations"].append(
                "Token usage is within acceptable limits"
            )

        # Show results
        self._display_token_validation(validation_result)

        return validation_result

    def _analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Analyze prompt characteristics."""
        # Basic metrics
        character_count = len(prompt)
        line_count = len(prompt.splitlines())
        estimated_tokens = character_count // 4  # Rough estimation

        # Check for examples
        has_examples = bool(re.search(r"example|Example|EXAMPLE", prompt))

        # Check for output format specification
        has_output_format = bool(re.search(r"JSON|json|format", prompt))

        # Calculate complexity score (0-1)
        complexity_factors = [
            len(prompt) > 1000,  # Long prompt
            prompt.count("{") > 10,  # Many placeholders
            has_examples,
            has_output_format,
            "CRITICAL" in prompt or "IMPORTANT" in prompt,
        ]
        complexity_score = sum(complexity_factors) / len(complexity_factors)

        # Identify potential issues
        potential_issues = []

        if character_count > 4000:
            potential_issues.append("Prompt may be too long for some models")

        if not has_examples:
            potential_issues.append(
                "No examples provided - may lead to inconsistent output"
            )

        if not has_output_format:
            potential_issues.append("Output format not clearly specified")

        if prompt.count("```") % 2 != 0:
            potential_issues.append("Unmatched code block markers")

        if re.search(r"\{[^}]*\{", prompt):
            potential_issues.append("Potential nested placeholder issues")

        return PromptAnalysis(
            prompt_text=prompt,
            estimated_tokens=estimated_tokens,
            character_count=character_count,
            line_count=line_count,
            has_examples=has_examples,
            has_output_format=has_output_format,
            complexity_score=complexity_score,
            potential_issues=potential_issues,
        )

    def _display_prompt_analysis(
        self, analysis_type: str, analysis: PromptAnalysis, function: ParsedFunction
    ) -> None:
        """Display detailed prompt analysis."""
        # Header
        self.console.print(
            f"\n[bold green]Prompt Analysis: {analysis_type}[/bold green]"
        )
        self.console.print("=" * 60)

        # Function info
        self.console.print(f"[blue]Function:[/blue] {function.signature.name}")
        self.console.print(
            f"[blue]File:[/blue] {function.file_path}:{function.line_number}"
        )

        # Metrics table
        table = Table(title="Prompt Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Assessment", style="yellow")

        # Token assessment
        token_assessment = "Good"
        if analysis.estimated_tokens > 3000:
            token_assessment = "High"
        elif analysis.estimated_tokens > 2000:
            token_assessment = "Moderate"

        table.add_row(
            "Estimated Tokens", str(analysis.estimated_tokens), token_assessment
        )
        table.add_row("Character Count", str(analysis.character_count), "")
        table.add_row("Line Count", str(analysis.line_count), "")
        table.add_row(
            "Has Examples",
            str(analysis.has_examples),
            "Good" if analysis.has_examples else "Missing",
        )
        table.add_row(
            "Has Output Format",
            str(analysis.has_output_format),
            "Good" if analysis.has_output_format else "Missing",
        )
        table.add_row("Complexity Score", f"{analysis.complexity_score:.2f}", "")

        self.console.print(table)

        # Potential issues
        if analysis.potential_issues:
            self.console.print("\n[red]Potential Issues:[/red]")
            for issue in analysis.potential_issues:
                self.console.print(f"  • {issue}")
        else:
            self.console.print("\n[green]No potential issues detected[/green]")

        # Show formatted prompt
        self.console.print("\n[blue]Generated Prompt:[/blue]")
        prompt_panel = Panel(
            Syntax(analysis.prompt_text, "text", theme="monokai", line_numbers=True),
            title="Formatted Prompt",
            border_style="blue",
        )
        self.console.print(prompt_panel)

    def _show_prompt_summary(
        self, analysis_type: str, analysis: PromptAnalysis
    ) -> None:
        """Show brief summary of prompt analysis."""
        status = "✅" if not analysis.potential_issues else "⚠️"
        tokens_status = (
            "🟢"
            if analysis.estimated_tokens < 2000
            else "🟡" if analysis.estimated_tokens < 3000 else "🔴"
        )

        self.console.print(
            f"  {status} {tokens_status} {analysis.estimated_tokens:4d} tokens, "
            f"{len(analysis.potential_issues):2d} issues"
        )

    def _show_prompts_comparison(self, results: dict[str, PromptAnalysis]) -> None:
        """Show comparison table of all prompt analyses."""
        self.console.print("\n[bold]Prompt Comparison Summary[/bold]")

        table = Table()
        table.add_column("Analysis Type", style="cyan")
        table.add_column("Tokens", justify="right", style="green")
        table.add_column("Lines", justify="right")
        table.add_column("Complexity", justify="right")
        table.add_column("Issues", justify="right", style="red")
        table.add_column("Status", justify="center")

        for analysis_type, analysis in results.items():
            status = "✅" if not analysis.potential_issues else "⚠️"

            table.add_row(
                analysis_type,
                str(analysis.estimated_tokens),
                str(analysis.line_count),
                f"{analysis.complexity_score:.2f}",
                str(len(analysis.potential_issues)),
                status,
            )

        self.console.print(table)

    def _display_parse_result(self, label: str, result: ParseResult) -> None:
        """Display parsing result details."""
        status = "✅ Success" if result.success else "❌ Failed"
        self.console.print(f"\n[bold]{label}:[/bold] {status}")

        if result.success:
            self.console.print(f"  Issues found: {len(result.issues)}")
            self.console.print(f"  Confidence: {result.confidence:.2f}")
            if result.analysis_notes:
                self.console.print(f"  Notes: {result.analysis_notes}")
        else:
            self.console.print(f"  Error: {result.error_message}")

        if result.raw_response:
            self.console.print(f"  Response length: {len(result.raw_response)} chars")

    def _display_parsing_statistics(self, stats: dict[str, Any]) -> None:
        """Display parsing statistics summary."""
        self.console.print("\n[bold]Parsing Statistics Summary[/bold]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Responses", str(stats["total_responses"]))
        table.add_row("Successful Parses", str(stats["successful_parses"]))
        table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
        table.add_row(
            "Avg Issues/Response", f"{stats['average_issues_per_response']:.1f}"
        )

        self.console.print(table)

        if stats["common_errors"]:
            self.console.print("\n[red]Common Errors:[/red]")
            for error, count in stats["common_errors"]:
                self.console.print(f"  • {error}: {count} times")

    def _display_token_validation(self, validation: dict[str, Any]) -> None:
        """Display token validation results."""
        status = "✅" if validation["within_limit"] else "❌"

        self.console.print(f"\n[bold]Token Validation:[/bold] {status}")
        self.console.print(f"  Estimated tokens: {validation['estimated_tokens']}")
        self.console.print(f"  Max tokens: {validation['max_tokens']}")
        self.console.print(f"  Usage: {validation['usage_percentage']:.1f}%")

        for rec in validation["recommendations"]:
            self.console.print(f"  • {rec}")

    def _get_function_signature_str(self, function: ParsedFunction) -> str:
        """Get readable function signature string."""
        sig = function.signature
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

    def _get_function_source_str(self, function: ParsedFunction) -> str:
        """Get function source code string (placeholder for testing)."""
        # In a real implementation, this would extract the actual source code
        # For debugging, we'll create a simple placeholder
        return f"""def {function.signature.name}():
    '''Placeholder source code for debugging.'''
    # Implementation would go here
    pass"""


# Convenience functions for common debugging scenarios
def debug_single_prompt(
    analysis_type: str,
    function: ParsedFunction,
    docstring: ParsedDocstring,
    console: Console | None = None,
) -> PromptAnalysis:
    """Debug a single prompt template."""
    debugger = PromptDebugger(console)
    return debugger.debug_prompt(analysis_type, function, docstring)


def test_sample_responses(
    responses: list[str], console: Console | None = None
) -> dict[str, Any]:
    """Test parsing of sample LLM responses."""
    debugger = PromptDebugger(console)
    return debugger.test_response_parsing(responses)


def validate_token_limits(
    request: LLMAnalysisRequest,
    max_tokens: int = 4000,
    console: Console | None = None,
) -> dict[str, Any]:
    """Validate that request fits within token limits."""
    debugger = PromptDebugger(console)
    return debugger.validate_prompt_token_usage(request, max_tokens)


# Sample data generators for testing
def create_sample_function() -> ParsedFunction:
    """Create a sample ParsedFunction for testing."""

    # This is a placeholder - in real usage, you'd import actual parser classes
    # For now, we'll create a simple mock
    class MockSignature:
        def __init__(self) -> None:
            self.name = "process_data"
            self.parameters = [
                type(
                    "MockParam",
                    (),
                    {
                        "name": "data",
                        "type_annotation": "List[Dict[str, Any]]",
                        "default_value": None,
                    },
                )(),
                type(
                    "MockParam",
                    (),
                    {
                        "name": "validate",
                        "type_annotation": "bool",
                        "default_value": "True",
                    },
                )(),
            ]
            self.return_type = "Dict[str, Any]"

    return type(
        "MockFunction",
        (),
        {"signature": MockSignature(), "file_path": "example.py", "line_number": 10},
    )()


def create_sample_docstring() -> ParsedDocstring:
    """Create a sample ParsedDocstring for testing."""
    return type(
        "MockDocstring",
        (),
        {
            "raw_text": """Process input data with optional validation.

        Args:
            data: List of dictionaries containing raw data
            validate: Whether to validate each item

        Returns:
            Dictionary with processed results

        Raises:
            ValueError: If validation fails

        Example:
            >>> result = process_data([{"id": 1, "value": "test"}])
            >>> print(result["count"])
            1
        """
        },
    )()


# Sample LLM responses for testing
SAMPLE_RESPONSES = [
    # Valid response
    """
    {
        "issues": [
            {
                "type": "behavior_mismatch",
                "description": "Function handles None input but docstring doesn't mention this",
                "suggestion": "Add note about None handling in the Args section",
                "confidence": 0.85,
                "line_number": 5,
                "details": {"missing_behavior": "None handling"}
            }
        ],
        "analysis_notes": "Checked parameter handling and return values",
        "confidence": 0.88
    }
    """,
    # Response with markdown formatting
    """
    ```json
    {
        "issues": [],
        "analysis_notes": "No issues found",
        "confidence": 0.95
    }
    ```
    """,
    # Malformed response
    """
    This function looks good to me. No issues found.
    """,
    # Partially valid response
    """
    {
        "issues": [
            {
                "type": "example_invalid",
                "description": "Example in docstring has syntax error"
                // Missing required fields
            }
        ],
        "confidence": 0.7
    }
    """,
]
