"""
Formatting utilities for CLI output.

This module contains all the display and formatting helper functions
used across different CLI commands.
"""

import json
import sys
from io import StringIO

from rich.table import Table

from codedocsync.cli.console import console
from codedocsync.matcher import MatchResult
from codedocsync.parser import ParsedDocstring, RawDocstring


def serialize_docstring(
    docstring: ParsedDocstring | RawDocstring | None,
) -> dict | None:
    """Serialize docstring for JSON output."""
    if docstring is None:
        return None
    if isinstance(docstring, ParsedDocstring):
        return {
            "format": docstring.format.value,
            "summary": docstring.summary,
            "description": docstring.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_str,
                    "description": p.description,
                    "is_optional": p.is_optional,
                    "default_value": p.default_value,
                }
                for p in docstring.parameters
            ],
            "returns": (
                {
                    "type": docstring.returns.type_str,
                    "description": docstring.returns.description,
                }
                if docstring.returns
                else None
            ),
            "raises": [
                {
                    "exception_type": r.exception_type,
                    "description": r.description,
                }
                for r in docstring.raises
            ],
            "examples": docstring.examples,
            "is_valid": docstring.is_valid,
            "parse_errors": docstring.parse_errors,
        }
    else:
        # Raw docstring
        return {"raw_text": docstring.raw_text}


def display_match_results(result: MatchResult, show_unmatched: bool) -> None:
    """Display match results in terminal with Rich."""
    console.print("\n[bold]Matching Results[/bold]")
    console.print("=" * 50)

    # Summary table
    summary = result.get_summary()
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Functions", str(summary["total_functions"]))
    table.add_row("Matched", str(summary["matched"]))
    table.add_row("Unmatched", str(summary["unmatched"]))
    table.add_row("Match Rate", summary["match_rate"])
    table.add_row("Duration", f"{summary['duration_ms']:.1f}ms")

    console.print(table)

    # Match type breakdown
    if summary["matched"] > 0:
        console.print("\n[bold]Match Types:[/bold]")
        for match_type, count in summary["match_types"].items():
            if count > 0:
                console.print(f"  • {match_type}: {count}")

    # Show unmatched if requested
    if show_unmatched and result.unmatched_functions:
        console.print(
            f"\n[bold red]Unmatched Functions ({len(result.unmatched_functions)}):[/bold red]"
        )
        for func in result.unmatched_functions[:10]:  # Show first 10
            console.print(
                f"  • {func.signature.name} "
                f"([dim]{func.file_path}:{func.line_number}[/dim])"
            )
        if len(result.unmatched_functions) > 10:
            console.print(f"  ... and {len(result.unmatched_functions) - 10} more")


def format_json_contextual_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format contextual matching result as JSON."""
    output = {
        "summary": result.get_summary(),
        "matched_pairs": [
            {
                "function": pair.function.signature.name,
                "file": pair.function.file_path,
                "line": pair.function.line_number,
                "match_type": pair.match_type.value,
                "confidence": pair.confidence.overall,
                "reason": pair.match_reason,
            }
            for pair in result.matched_pairs
        ],
    }

    if show_unmatched:
        output["unmatched"] = [
            {
                "function": func.signature.name,
                "file": func.file_path,
                "line": func.line_number,
            }
            for func in result.unmatched_functions
        ]

    # Add performance metrics if available
    if hasattr(result, "metadata") and getattr(result, "metadata", None):
        output["metadata"] = getattr(result, "metadata", {})

    return json.dumps(output, indent=2)


def format_terminal_contextual_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format contextual matching result for terminal."""
    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        display_contextual_results(result, show_unmatched)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def display_contextual_results(result: MatchResult, show_unmatched: bool) -> None:
    """Display contextual matching results in terminal with Rich."""
    console.print("\n[bold]Contextual Matching Results[/bold]")
    console.print("=" * 60)

    # Summary table
    summary = result.get_summary()
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Functions", str(summary["total_functions"]))
    table.add_row("Matched", str(summary["matched"]))
    table.add_row("Unmatched", str(summary["unmatched"]))
    table.add_row("Match Rate", summary["match_rate"])

    console.print(table)

    # Match type breakdown
    if summary["matched"] > 0:
        console.print("\n[bold]Match Types:[/bold]")
        for match_type, count in summary["match_types"].items():
            if count > 0:
                console.print(f"  • {match_type}: {count}")

    # Show performance metrics if available
    if (
        hasattr(result, "metadata")
        and getattr(result, "metadata", None)
        and "performance" in getattr(result, "metadata", {})
    ):
        console.print("\n[bold]Performance Metrics:[/bold]")
        perf = getattr(result, "metadata", {})["performance"]
        console.print(f"  • Total time: {perf['total_time']:.2f}s")
        console.print(f"  • Files processed: {perf['files_processed']}")
        console.print(f"  • Parsing time: {perf['parsing_time']:.2f}s")
        console.print(f"  • Direct matching: {perf['direct_matching_time']:.2f}s")
        console.print(
            f"  • Contextual matching: {perf['contextual_matching_time']:.2f}s"
        )

    # Show detailed matches
    if result.matched_pairs:
        console.print(
            f"\n[bold]Matched Functions ({len(result.matched_pairs)}):[/bold]"
        )
        for pair in result.matched_pairs[:10]:  # Show first 10
            confidence_color = "green" if pair.confidence.overall >= 0.8 else "yellow"
            console.print(
                f"  • {pair.function.signature.name} "
                f"([{confidence_color}]{pair.confidence.overall:.2f}[/{confidence_color}]) "
                f"- {pair.match_type.value} "
                f"([dim]{pair.function.file_path}:{pair.function.line_number}[/dim])"
            )
            if pair.match_reason:
                console.print(f"    {pair.match_reason}")

        if len(result.matched_pairs) > 10:
            console.print(f"  ... and {len(result.matched_pairs) - 10} more")

    # Show unmatched if requested
    if show_unmatched and result.unmatched_functions:
        console.print(
            f"\n[bold red]Unmatched Functions ({len(result.unmatched_functions)}):[/bold red]"
        )
        for func in result.unmatched_functions[:10]:  # Show first 10
            console.print(
                f"  • {func.signature.name} "
                f"([dim]{func.file_path}:{func.line_number}[/dim])"
            )
        if len(result.unmatched_functions) > 10:
            console.print(f"  ... and {len(result.unmatched_functions) - 10} more")


def format_json_unified_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format unified matching result as comprehensive JSON."""
    output = {
        "summary": result.get_summary(),
        "matched_pairs": [
            {
                "function": pair.function.signature.name,
                "file": pair.function.file_path,
                "line": pair.function.line_number,
                "match_type": pair.match_type.value,
                "confidence": {
                    "overall": pair.confidence.overall,
                    "name_similarity": pair.confidence.name_similarity,
                    "location_score": pair.confidence.location_score,
                    "signature_similarity": pair.confidence.signature_similarity,
                },
                "reason": pair.match_reason,
                "docstring": (
                    serialize_docstring(pair.docstring) if pair.docstring else None
                ),
            }
            for pair in result.matched_pairs
        ],
    }

    if show_unmatched:
        output["unmatched"] = [
            {
                "function": func.signature.name,
                "file": func.file_path,
                "line": func.line_number,
                "signature": func.signature.to_string(),
            }
            for func in result.unmatched_functions
        ]

    # Add comprehensive metadata if available
    if hasattr(result, "metadata") and getattr(result, "metadata", None):
        metadata = getattr(result, "metadata", {})
        output["metadata"] = metadata

        # Add performance insights if available
        if "unified_stats" in metadata:
            unified_stats = metadata["unified_stats"]
            output["performance_insights"] = {
                "total_time": unified_stats.get("total_time_seconds", 0),
                "memory_usage": unified_stats.get("memory_usage", {}),
                "throughput": unified_stats.get("throughput", {}),
                "phase_breakdown": unified_stats.get("phase_times", {}),
                "efficiency": unified_stats.get("efficiency_metrics", {}),
                "error_summary": unified_stats.get("error_summary", {}),
            }

        # Add system information
        if "system_info" in metadata:
            output["system_info"] = metadata["system_info"]

        # Add performance recommendations
        if "performance_profile" in metadata:
            output["performance_profile"] = metadata["performance_profile"]

    return json.dumps(output, indent=2)


def format_terminal_unified_result(
    result: MatchResult, show_unmatched: bool, enable_semantic: bool = True
) -> str:
    """Format unified matching result for terminal with enhanced display."""
    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        display_unified_results(result, show_unmatched, enable_semantic)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def format_analysis_results(
    results: list, all_issues: list, total_time: float, show_summary: bool
) -> str:
    """Format analysis results for terminal output."""
    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        display_analysis_results(results, all_issues, total_time, show_summary)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def display_analysis_results(
    results: list, all_issues: list, total_time: float, show_summary: bool
) -> None:
    """Display analysis results in terminal with Rich."""
    console.print("\n[bold]Analysis Results[/bold]")
    console.print("=" * 60)

    if show_summary:
        # Summary table
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Functions", str(len(results)))
        table.add_row("Total Issues", str(len(all_issues)))
        table.add_row(
            "Critical Issues",
            str(len([i for i in all_issues if i.severity == "critical"])),
        )
        table.add_row(
            "High Issues", str(len([i for i in all_issues if i.severity == "high"]))
        )
        table.add_row(
            "Medium Issues", str(len([i for i in all_issues if i.severity == "medium"]))
        )
        table.add_row(
            "Low Issues", str(len([i for i in all_issues if i.severity == "low"]))
        )
        table.add_row("Analysis Time", f"{total_time:.2f}s")

        console.print(table)

    # Group issues by severity
    critical_issues = [i for i in all_issues if i.severity == "critical"]
    high_issues = [i for i in all_issues if i.severity == "high"]
    medium_issues = [i for i in all_issues if i.severity == "medium"]
    low_issues = [i for i in all_issues if i.severity == "low"]

    # Display critical issues first
    if critical_issues:
        console.print(
            f"\n[bold red]Critical Issues ({len(critical_issues)}):[/bold red]"
        )
        for issue in critical_issues[:10]:  # Show first 10
            console.print(f"  • {issue.description}")
            # Check if we have an enhanced suggestion
            if hasattr(issue, "rich_suggestion") and issue.rich_suggestion:
                # Track that the suggestion was displayed
                from codedocsync.suggestions.metrics import get_metrics_collector

                metrics_collector = get_metrics_collector()
                if issue.rich_suggestion.metadata.suggestion_id:
                    metrics_collector.mark_displayed(
                        issue.rich_suggestion.metadata.suggestion_id
                    )

                console.print(
                    f"    [dim]Line {issue.line_number}:[/dim] [green]Enhanced suggestion available[/green]"
                )
                console.print(
                    f"    [cyan]Confidence: {issue.rich_suggestion.confidence:.0%}[/cyan]"
                )
                console.print(
                    f"    [dim]{issue.rich_suggestion.suggested_text[:100]}...[/dim]"
                    if len(issue.rich_suggestion.suggested_text) > 100
                    else f"    [dim]{issue.rich_suggestion.suggested_text}[/dim]"
                )
            else:
                console.print(
                    f"    [dim]Line {issue.line_number}: {issue.suggestion}[/dim]"
                )
        if len(critical_issues) > 10:
            console.print(f"  ... and {len(critical_issues) - 10} more critical issues")

    # Display high issues
    if high_issues:
        console.print(
            f"\n[bold yellow]High Priority Issues ({len(high_issues)}):[/bold yellow]"
        )
        for issue in high_issues[:5]:  # Show first 5
            console.print(f"  • {issue.description}")
            # Check if we have an enhanced suggestion
            if hasattr(issue, "rich_suggestion") and issue.rich_suggestion:
                # Track that the suggestion was displayed
                from codedocsync.suggestions.metrics import get_metrics_collector

                metrics_collector = get_metrics_collector()
                if issue.rich_suggestion.metadata.suggestion_id:
                    metrics_collector.mark_displayed(
                        issue.rich_suggestion.metadata.suggestion_id
                    )

                console.print(
                    f"    [dim]Line {issue.line_number}:[/dim] [green]Enhanced suggestion available[/green]"
                )
                console.print(
                    f"    [cyan]Confidence: {issue.rich_suggestion.confidence:.0%}[/cyan]"
                )
                console.print(
                    f"    [dim]{issue.rich_suggestion.suggested_text[:100]}...[/dim]"
                    if len(issue.rich_suggestion.suggested_text) > 100
                    else f"    [dim]{issue.rich_suggestion.suggested_text}[/dim]"
                )
            else:
                console.print(
                    f"    [dim]Line {issue.line_number}: {issue.suggestion}[/dim]"
                )
        if len(high_issues) > 5:
            console.print(f"  ... and {len(high_issues) - 5} more high priority issues")

    # Show summary for medium/low issues
    if medium_issues:
        console.print(f"\n[dim]Medium Issues: {len(medium_issues)}[/dim]")
    if low_issues:
        console.print(f"[dim]Low Issues: {len(low_issues)}[/dim]")


def display_unified_results(
    result: MatchResult, show_unmatched: bool, enable_semantic: bool = True
) -> None:
    """Display unified matching results in terminal with comprehensive Rich formatting."""
    # Build header based on enabled strategies
    strategies = ["Direct", "Contextual"]
    if enable_semantic:
        strategies.append("Semantic")
    header_text = " + ".join(strategies)

    console.print(
        f"\n[bold magenta][>>] Unified Matching Results ({header_text})[/bold magenta]"
    )
    console.print("=" * 80)

    # Enhanced summary table
    summary = result.get_summary()
    summary_table = Table(title="[Summary]", show_header=True, header_style="bold blue")
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", width=15)
    summary_table.add_column("Details", style="dim", width=30)

    summary_table.add_row(
        "Total Functions", str(summary["total_functions"]), "Functions analyzed"
    )
    summary_table.add_row(
        "Matched", str(summary["matched"]), f"Success rate: {summary['match_rate']}"
    )
    summary_table.add_row(
        "Unmatched", str(summary["unmatched"]), "May need manual review"
    )

    console.print(summary_table)

    # Enhanced match type breakdown with icons
    if summary["matched"] > 0:
        match_table = Table(
            title="[Distribution]", show_header=True, header_style="bold green"
        )
        match_table.add_column("Strategy", style="yellow", width=15)
        match_table.add_column("Count", justify="right", style="green", width=8)
        match_table.add_column("Percentage", justify="right", style="blue", width=12)
        match_table.add_column("Description", style="dim", width=30)

        total_matches = summary["matched"]
        match_descriptions = {
            "exact": "Same name, same file",
            "fuzzy": "Similar names, patterns",
            "contextual": "Import analysis, cross-file",
            "semantic": "AI similarity matching",
        }

        for match_type, count in summary.get("match_types", {}).items():
            if count > 0:
                percentage = (
                    f"{count / total_matches * 100:.1f}%" if total_matches > 0 else "0%"
                )
                description = match_descriptions.get(match_type, "Unknown strategy")
                match_table.add_row(
                    match_type.title(), str(count), percentage, description
                )

        console.print(match_table)

    # Enhanced performance metrics with comprehensive details
    if (
        hasattr(result, "metadata")
        and getattr(result, "metadata", None)
        and "unified_stats" in getattr(result, "metadata", {})
    ):
        metadata = getattr(result, "metadata", {})
        unified_stats = metadata["unified_stats"]

        # Performance overview
        perf_table = Table(
            title="[Performance]", show_header=True, header_style="bold cyan"
        )
        perf_table.add_column("Phase", style="yellow", width=20)
        perf_table.add_column("Time", justify="right", style="green", width=10)
        perf_table.add_column("Percentage", justify="right", style="blue", width=12)
        perf_table.add_column("Status", style="magenta", width=15)

        total_time = unified_stats.get("total_time_seconds", 0)
        phase_times = unified_stats.get("phase_times", {})

        for phase, time_val in phase_times.items():
            if time_val > 0:
                percentage = (
                    f"{time_val / total_time * 100:.1f}%" if total_time > 0 else "0%"
                )
                # Determine status based on time
                if phase == "parsing" and time_val > total_time * 0.5:
                    status = "[SLOW]"
                elif phase == "semantic_matching" and time_val > 60:
                    status = "[HEAVY]"
                else:
                    status = "[GOOD]"

                perf_table.add_row(
                    phase.replace("_", " ").title(),
                    f"{time_val:.2f}s",
                    percentage,
                    status,
                )

        console.print(perf_table)

        # Memory and throughput metrics
        system_table = Table(
            title="[System]", show_header=True, header_style="bold yellow"
        )
        system_table.add_column("Metric", style="cyan", width=20)
        system_table.add_column("Value", style="green", width=15)
        system_table.add_column("Unit", style="dim", width=10)

        # Memory metrics
        memory_usage = unified_stats.get("memory_usage", {})
        if memory_usage:
            system_table.add_row(
                "Initial Memory", f"{memory_usage.get('initial_mb', 0):.1f}", "MB"
            )
            system_table.add_row(
                "Peak Memory", f"{memory_usage.get('peak_mb', 0):.1f}", "MB"
            )
            system_table.add_row(
                "Memory Growth",
                f"{memory_usage.get('peak_mb', 0) - memory_usage.get('initial_mb', 0):.1f}",
                "MB",
            )

        # Throughput metrics
        throughput = unified_stats.get("throughput", {})
        if throughput:
            system_table.add_row(
                "Functions/sec",
                f"{throughput.get('functions_per_second', 0):.1f}",
                "ops/s",
            )
            system_table.add_row(
                "Files/sec", f"{throughput.get('files_per_second', 0):.1f}", "files/s"
            )

        # Error metrics
        errors = unified_stats.get("error_summary", {})
        if errors and errors.get("total_errors", 0) > 0:
            system_table.add_row(
                "Parse Errors", str(errors.get("parsing_errors", 0)), "errors"
            )
            system_table.add_row(
                "Match Errors", str(errors.get("matching_errors", 0)), "errors"
            )
            system_table.add_row(
                "Error Rate",
                f"{errors.get('total_errors', 0) / max(unified_stats.get('functions_processed', 1), 1) * 100:.1f}",
                "%",
            )

        console.print(system_table)

        # Performance recommendations
        if "performance_profile" in metadata:
            profile = metadata["performance_profile"]
            console.print("\n[bold blue][Insights]:[/bold blue]")

            if profile.get("bottleneck") != "none":
                console.print(
                    f"  🚨 Bottleneck: {profile.get('bottleneck', 'unknown')}"
                )
                console.print(
                    f"  [TIP] Recommendation: {profile.get('recommendation', 'No specific advice')}"
                )
            else:
                console.print("  [OK] Performance is well balanced!")

            memory_concern = profile.get("memory_concern", "acceptable")
            if memory_concern != "acceptable":
                console.print(f"  📈 Memory usage: {memory_concern}")

    # Enhanced detailed matches with confidence breakdown
    if result.matched_pairs:
        console.print(
            f"\n[bold][Matches] Detailed Matches ({len(result.matched_pairs)}):[/bold]"
        )
        for i, pair in enumerate(result.matched_pairs[:15]):  # Show first 15
            confidence_color = (
                "green"
                if pair.confidence.overall >= 0.8
                else "yellow" if pair.confidence.overall >= 0.6 else "red"
            )
            match_icon = {
                "EXACT": "[EXACT]",
                "FUZZY": "[FUZZY]",
                "CONTEXTUAL": "[CONTEXT]",
                "SEMANTIC": "[SEMANTIC]",
            }.get(pair.match_type.value, "📝")

            console.print(
                f"  {i + 1:2d}. {match_icon} {pair.function.signature.name} "
                f"([{confidence_color}]{pair.confidence.overall:.2f}[/{confidence_color}]) "
                f"- {pair.match_type.value.lower()} "
                f"([dim]{pair.function.file_path}:{pair.function.line_number}[/dim])"
            )

            if pair.match_reason:
                console.print(f"      💬 {pair.match_reason}")

            # Show confidence breakdown for high-detail cases
            if pair.confidence.overall < 0.8:
                console.print(
                    f"      [Stats] Name: {pair.confidence.name_similarity:.2f}, "
                    f"Location: {pair.confidence.location_score:.2f}, "
                    f"Signature: {pair.confidence.signature_similarity:.2f}"
                )

        if len(result.matched_pairs) > 15:
            console.print(f"  ... and {len(result.matched_pairs) - 15} more matches")

    # Enhanced unmatched section if requested
    if show_unmatched and result.unmatched_functions:
        console.print(
            f"\n[bold red][UNMATCHED] Functions ({len(result.unmatched_functions)}):[/bold red]"
        )
        console.print(
            "[dim]These functions may need manual documentation review:[/dim]"
        )

        for i, func in enumerate(result.unmatched_functions[:10]):  # Show first 10
            console.print(
                f"  {i + 1:2d}. [WARNING] {func.signature.name} "
                f"([dim]{func.file_path}:{func.line_number}[/dim])"
            )
            # Show signature for context
            if hasattr(func.signature, "to_string"):
                console.print(
                    f"      📝 {func.signature.to_string()[:80]}{'...' if len(func.signature.to_string()) > 80 else ''}"
                )

        if len(result.unmatched_functions) > 10:
            console.print(
                f"  ... and {len(result.unmatched_functions) - 10} more unmatched functions"
            )

    console.print("\n[green][COMPLETE] Analysis complete![/green]")
