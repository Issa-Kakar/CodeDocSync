"""
Terminal output formatter for suggestions.

Provides rich, colored output optimized for command-line interfaces
with support for different verbosity levels and terminal capabilities.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ..models import Suggestion, SuggestionBatch
from ..integration import EnhancedAnalysisResult, EnhancedIssue

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.table import Table
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class OutputStyle(Enum):
    """Output style options."""

    RICH = "rich"  # Rich formatting with colors and panels
    PLAIN = "plain"  # Plain text for basic terminals
    MINIMAL = "minimal"  # Minimal output for scripting


@dataclass
class TerminalFormatterConfig:
    """Configuration for terminal formatter."""

    max_width: int = 88
    show_line_numbers: bool = True
    show_confidence: bool = True
    show_diff: bool = True
    syntax_theme: str = "monokai"
    use_unicode: bool = True
    compact_mode: bool = False


class TerminalSuggestionFormatter:
    """Format suggestions for terminal output."""

    def __init__(
        self,
        config: Optional[TerminalFormatterConfig] = None,
        style: OutputStyle = OutputStyle.RICH,
    ):
        """Initialize formatter with configuration."""
        self.config = config or TerminalFormatterConfig()
        self.style = style if RICH_AVAILABLE else OutputStyle.PLAIN
        self.console = Console(width=self.config.max_width) if RICH_AVAILABLE else None

    def format_suggestion(self, suggestion: Suggestion) -> str:
        """Format a single suggestion."""
        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            return self._format_rich_suggestion(suggestion)
        elif self.style == OutputStyle.MINIMAL:
            return self._format_minimal_suggestion(suggestion)
        else:
            return self._format_plain_suggestion(suggestion)

    def format_enhanced_issue(self, issue: EnhancedIssue) -> str:
        """Format an enhanced issue with its suggestion."""
        if not issue.rich_suggestion:
            return self._format_issue_only(issue)

        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            return self._format_rich_enhanced_issue(issue)
        elif self.style == OutputStyle.MINIMAL:
            return self._format_minimal_enhanced_issue(issue)
        else:
            return self._format_plain_enhanced_issue(issue)

    def format_analysis_result(self, result: EnhancedAnalysisResult) -> str:
        """Format a complete analysis result."""
        if not result.issues:
            return self._format_no_issues(result)

        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            return self._format_rich_analysis_result(result)
        elif self.style == OutputStyle.MINIMAL:
            return self._format_minimal_analysis_result(result)
        else:
            return self._format_plain_analysis_result(result)

    def format_batch_summary(self, batch: SuggestionBatch) -> str:
        """Format a summary of a suggestion batch."""
        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            return self._format_rich_batch_summary(batch)
        elif self.style == OutputStyle.MINIMAL:
            return self._format_minimal_batch_summary(batch)
        else:
            return self._format_plain_batch_summary(batch)

    # Rich formatting methods
    def _format_rich_suggestion(self, suggestion: Suggestion) -> str:
        """Format suggestion with rich styling."""
        if not RICH_AVAILABLE:
            return self._format_plain_suggestion(suggestion)

        # Create syntax-highlighted suggestion
        syntax = Syntax(
            suggestion.suggested_text,
            "python",
            theme=self.config.syntax_theme,
            line_numbers=self.config.show_line_numbers,
            word_wrap=True,
        )

        # Create diff if available and requested
        diff_content = ""
        if self.config.show_diff and suggestion.diff:
            diff_content = self._create_rich_diff(suggestion)

        # Confidence indicator
        confidence_text = ""
        if self.config.show_confidence:
            confidence = suggestion.confidence
            if confidence >= 0.9:
                conf_style = "bold green"
                conf_icon = "✅"
            elif confidence >= 0.7:
                conf_style = "bold yellow"
                conf_icon = "⚠️"
            else:
                conf_style = "bold red"
                conf_icon = "❌"

            confidence_text = (
                f"[{conf_style}]{conf_icon} Confidence: {confidence:.0%}[/]"
            )

        # Create panel content
        panel_content = f"""[bold blue]📝 Suggested Fix[/]
{confidence_text}

{syntax}"""

        if diff_content:
            panel_content += f"\n\n[bold]📊 Changes:[/]\n{diff_content}"

        panel = Panel(
            panel_content,
            title=f"[bold]{suggestion.suggestion_type.value.replace('_', ' ').title()}[/]",
            border_style="blue",
            box=box.ROUNDED,
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    def _format_rich_enhanced_issue(self, issue: EnhancedIssue) -> str:
        """Format enhanced issue with rich styling."""
        if not RICH_AVAILABLE:
            return self._format_plain_enhanced_issue(issue)

        # Severity styling
        severity_styles = {
            "critical": ("💥", "bold red"),
            "high": ("⚠️", "bold yellow"),
            "medium": ("ℹ️", "bold blue"),
            "low": ("💡", "dim"),
        }
        icon, style = severity_styles.get(issue.severity, ("•", ""))

        # Issue description
        issue_text = Text()
        issue_text.append(f"{icon} ", style=style)
        issue_text.append(f"{issue.description}", style=style)

        # Line number
        if issue.line_number > 0:
            issue_text.append(f" (line {issue.line_number})", style="dim")

        content = str(issue_text)

        # Add suggestion if available
        if issue.rich_suggestion:
            suggestion_text = self._format_rich_suggestion(issue.rich_suggestion)
            content += f"\n\n{suggestion_text}"

        return content

    def _format_rich_analysis_result(self, result: EnhancedAnalysisResult) -> str:
        """Format complete analysis result with rich styling."""
        if not RICH_AVAILABLE:
            return self._format_plain_analysis_result(result)

        # Header with function info
        function_name = "Unknown"
        file_path = "Unknown"
        if hasattr(result.matched_pair.function, "signature"):
            function_name = result.matched_pair.function.signature.name
            file_path = result.matched_pair.function.file_path

        header = Panel(
            f"[bold blue]Function:[/] {function_name}\n"
            f"[bold blue]File:[/] {file_path}\n"
            f"[bold blue]Issues Found:[/] {len(result.issues)}\n"
            f"[bold blue]Suggestions Generated:[/] {result.suggestions_generated}",
            title="[bold]Analysis Result[/]",
            border_style="green",
        )

        # Format each issue
        issue_outputs = []
        for i, issue in enumerate(result.issues, 1):
            issue_output = self._format_rich_enhanced_issue(issue)
            issue_panel = Panel(
                issue_output,
                title=f"[bold]Issue {i}[/]",
                border_style=(
                    "yellow" if issue.severity in ["critical", "high"] else "blue"
                ),
            )
            issue_outputs.append(issue_panel)

        # Combine all parts
        with self.console.capture() as capture:
            self.console.print(header)
            self.console.print()

            for issue_panel in issue_outputs:
                self.console.print(issue_panel)
                self.console.print()

        return capture.get()

    def _format_rich_batch_summary(self, batch: SuggestionBatch) -> str:
        """Format batch summary with rich styling."""
        if not RICH_AVAILABLE:
            return self._format_plain_batch_summary(batch)

        # Create summary table
        table = Table(title="Suggestion Generation Summary", box=box.ROUNDED)
        table.add_column("Metric", style="bold blue")
        table.add_column("Value", style="green")

        table.add_row("Functions Processed", str(batch.functions_processed))
        table.add_row("Total Issues", str(batch.total_issues))
        table.add_row("Suggestions Generated", str(len(batch.suggestions)))
        table.add_row("Generation Time", f"{batch.generation_time_ms:.1f}ms")

        if batch.suggestions:
            avg_confidence = sum(s.confidence for s in batch.suggestions) / len(
                batch.suggestions
            )
            table.add_row("Average Confidence", f"{avg_confidence:.1%}")

        with self.console.capture() as capture:
            self.console.print(table)

        return capture.get()

    def _create_rich_diff(self, suggestion: Suggestion) -> str:
        """Create rich diff display."""
        if not suggestion.diff or not RICH_AVAILABLE:
            return ""

        # Simple diff display for now
        lines_changed = len(suggestion.diff.suggested_lines) - len(
            suggestion.diff.original_lines
        )
        if lines_changed > 0:
            return f"[green]+{lines_changed} lines added[/]"
        elif lines_changed < 0:
            return f"[red]{lines_changed} lines removed[/]"
        else:
            return f"[yellow]{len(suggestion.diff.suggested_lines)} lines modified[/]"

    # Plain text formatting methods
    def _format_plain_suggestion(self, suggestion: Suggestion) -> str:
        """Format suggestion in plain text."""
        lines = []

        lines.append(
            f"=== {suggestion.suggestion_type.value.replace('_', ' ').title()} ==="
        )

        if self.config.show_confidence:
            lines.append(f"Confidence: {suggestion.confidence:.0%}")

        lines.append("")
        lines.append("Suggested code:")
        lines.append("-" * 40)

        # Add line numbers if requested
        if self.config.show_line_numbers:
            code_lines = suggestion.suggested_text.split("\n")
            for i, line in enumerate(code_lines, 1):
                lines.append(f"{i:3d}: {line}")
        else:
            lines.append(suggestion.suggested_text)

        lines.append("-" * 40)

        return "\n".join(lines)

    def _format_plain_enhanced_issue(self, issue: EnhancedIssue) -> str:
        """Format enhanced issue in plain text."""
        lines = []

        # Issue header
        severity_indicators = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
        }
        indicator = severity_indicators.get(issue.severity, "[ISSUE]")

        lines.append(f"{indicator} {issue.description}")
        if issue.line_number > 0:
            lines.append(f"  Line: {issue.line_number}")

        # Add suggestion if available
        if issue.rich_suggestion:
            lines.append("")
            lines.append(self._format_plain_suggestion(issue.rich_suggestion))

        return "\n".join(lines)

    def _format_plain_analysis_result(self, result: EnhancedAnalysisResult) -> str:
        """Format analysis result in plain text."""
        lines = []

        # Header
        function_name = "Unknown"
        file_path = "Unknown"
        if hasattr(result.matched_pair.function, "signature"):
            function_name = result.matched_pair.function.signature.name
            file_path = result.matched_pair.function.file_path

        lines.append("=" * 60)
        lines.append(f"Function: {function_name}")
        lines.append(f"File: {file_path}")
        lines.append(f"Issues: {len(result.issues)}")
        lines.append(f"Suggestions: {result.suggestions_generated}")
        lines.append("=" * 60)
        lines.append("")

        # Issues
        for i, issue in enumerate(result.issues, 1):
            lines.append(f"Issue {i}:")
            lines.append(self._format_plain_enhanced_issue(issue))
            lines.append("")

        return "\n".join(lines)

    def _format_plain_batch_summary(self, batch: SuggestionBatch) -> str:
        """Format batch summary in plain text."""
        lines = []

        lines.append("Suggestion Generation Summary")
        lines.append("=" * 30)
        lines.append(f"Functions Processed: {batch.functions_processed}")
        lines.append(f"Total Issues: {batch.total_issues}")
        lines.append(f"Suggestions Generated: {len(batch.suggestions)}")
        lines.append(f"Generation Time: {batch.generation_time_ms:.1f}ms")

        if batch.suggestions:
            avg_confidence = sum(s.confidence for s in batch.suggestions) / len(
                batch.suggestions
            )
            lines.append(f"Average Confidence: {avg_confidence:.1%}")

        return "\n".join(lines)

    # Minimal formatting methods
    def _format_minimal_suggestion(self, suggestion: Suggestion) -> str:
        """Format suggestion minimally."""
        return suggestion.suggested_text

    def _format_minimal_enhanced_issue(self, issue: EnhancedIssue) -> str:
        """Format enhanced issue minimally."""
        lines = [f"{issue.severity.upper()}: {issue.description}"]
        if issue.rich_suggestion:
            lines.append(issue.rich_suggestion.suggested_text)
        return "\n".join(lines)

    def _format_minimal_analysis_result(self, result: EnhancedAnalysisResult) -> str:
        """Format analysis result minimally."""
        lines = []
        for issue in result.issues:
            lines.append(self._format_minimal_enhanced_issue(issue))
        return "\n".join(lines)

    def _format_minimal_batch_summary(self, batch: SuggestionBatch) -> str:
        """Format batch summary minimally."""
        return f"Processed: {batch.functions_processed}, Issues: {batch.total_issues}, Suggestions: {len(batch.suggestions)}"

    # Utility methods
    def _format_issue_only(self, issue: EnhancedIssue) -> str:
        """Format issue without suggestion."""
        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            severity_styles = {
                "critical": ("💥", "bold red"),
                "high": ("⚠️", "bold yellow"),
                "medium": ("ℹ️", "bold blue"),
                "low": ("💡", "dim"),
            }
            icon, style = severity_styles.get(issue.severity, ("•", ""))

            text = Text()
            text.append(f"{icon} ", style=style)
            text.append(f"{issue.description}", style=style)
            if issue.line_number > 0:
                text.append(f" (line {issue.line_number})", style="dim")

            return str(text)
        else:
            severity_indicators = {
                "critical": "[CRITICAL]",
                "high": "[HIGH]",
                "medium": "[MEDIUM]",
                "low": "[LOW]",
            }
            indicator = severity_indicators.get(issue.severity, "[ISSUE]")
            line_info = f" (line {issue.line_number})" if issue.line_number > 0 else ""
            return f"{indicator} {issue.description}{line_info}"

    def _format_no_issues(self, result: EnhancedAnalysisResult) -> str:
        """Format when no issues found."""
        function_name = "Unknown"
        if hasattr(result.matched_pair.function, "signature"):
            function_name = result.matched_pair.function.signature.name

        if self.style == OutputStyle.RICH and RICH_AVAILABLE:
            return f"[bold green]✅ No issues found in {function_name}[/]"
        else:
            return f"✅ No issues found in {function_name}"
