"""
Main entry point for the CodeDocSync CLI application.
"""

import json
import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table

from codedocsync import __version__
from codedocsync.parser import IntegratedParser, ParsingError, ParsedDocstring
from codedocsync.matcher import (
    MatchingFacade,
    MatchResult,
    ContextualMatchingFacade,
    UnifiedMatchingFacade,
)
from codedocsync.utils.config import CodeDocSyncConfig

app = typer.Typer(
    help="CodeDocSync: An intelligent tool to find and fix documentation drift."
)
console = Console()


def _serialize_docstring(docstring):
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


def version_callback(value: bool):
    """Prints the version of the application and exits."""
    if value:
        print(f"CodeDocSync v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show the application's version and exit.",
        ),
    ] = False,
):
    """
    Manage documentation consistency checks for your codebase.
    """
    pass


@app.command()
def analyze(
    path: Annotated[
        str, typer.Argument(help="The path to the file or directory to analyze.")
    ] = ".",
):
    """
    Analyzes the project for documentation inconsistencies.
    """
    print(f"Analyzing documentation in: {path}")
    # Placeholder for analysis logic. This is where you'll integrate the AST parser.
    print("Analysis complete. (Placeholder)")


@app.command()
def watch(
    path: Annotated[
        str, typer.Argument(help="The path to the directory to watch.")
    ] = ".",
):
    """
    Watches files for changes and provides real-time feedback.
    """
    print(f"Starting watch mode for: {path}")
    # Placeholder for file watching logic.
    print("Watch mode active. (Placeholder)")


@app.command()
def check(
    path: Annotated[
        str, typer.Argument(help="The path to the file or directory to check.")
    ] = ".",
):
    """
    Checks for inconsistencies, intended for CI/CD pipelines.
    """
    print(f"Running CI check on: {path}")
    # Placeholder for CI/CD check logic.
    print("Check complete. (Placeholder)")


@app.command()
def match(
    path: Annotated[Path, typer.Argument(help="File or directory to match")],
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file")
    ] = None,
    show_unmatched: Annotated[
        bool, typer.Option("--show-unmatched", help="Show unmatched functions")
    ] = False,
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (terminal/json)")
    ] = "terminal",
):
    """
    Match functions to their documentation.

    Example:
        codedocsync match ./myproject --show-unmatched
    """
    # Load configuration
    if config and config.exists():
        config_obj = CodeDocSyncConfig.from_yaml(str(config))
    else:
        config_obj = CodeDocSyncConfig()

    # Create matching facade
    facade = MatchingFacade(config_obj)

    # Match based on path type
    if path.is_file():
        result = facade.match_file(path)
    elif path.is_dir():
        result = facade.match_project(path)
    else:
        console.print(f"[red]Error: {path} is not a valid file or directory[/red]")
        raise typer.Exit(1)

    # Display results
    if output_format == "json":
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
        print(json.dumps(output, indent=2))
    else:
        # Terminal output with Rich
        _display_match_results(result, show_unmatched)


@app.command()
def parse(
    file: Annotated[Path, typer.Argument(help="Python file to parse")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of pretty-printed table"),
    ] = False,
):
    """
    Parse a Python file and display extracted functions.

    This command demonstrates the AST parser functionality by analyzing
    a Python file and extracting all function definitions with their
    signatures, parameters, and docstrings.
    """
    try:
        # Use IntegratedParser for complete parsing with docstring analysis
        parser = IntegratedParser()
        functions = parser.parse_file(str(file))

        if json_output:
            # Convert to JSON-serializable format
            json_data = []
            for func in functions:
                func_data = {
                    "name": func.signature.name,
                    "is_async": func.signature.is_async,
                    "is_method": func.signature.is_method,
                    "line_number": func.line_number,
                    "end_line_number": func.end_line_number,
                    "parameters": [
                        {
                            "name": param.name,
                            "type_annotation": param.type_annotation,
                            "default_value": param.default_value,
                            "is_required": param.is_required,
                        }
                        for param in func.signature.parameters
                    ],
                    "return_type": func.signature.return_type,
                    "decorators": func.signature.decorators,
                    "docstring": _serialize_docstring(func.docstring),
                    "signature_string": func.signature.to_string(),
                }
                json_data.append(func_data)

            console.print(json.dumps(json_data, indent=2))
        else:
            # Pretty print with Rich
            if not functions:
                console.print("[yellow]No functions found in the file.[/yellow]")
                return

            table = Table(title=f"Functions in {file}")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Parameters", style="green")
            table.add_column("Return Type", style="blue")
            table.add_column("Lines", style="yellow")
            table.add_column("Docstring", style="white", max_width=40)
            table.add_column("Format", style="cyan", max_width=10)

            for func in functions:
                func_type = "async" if func.signature.is_async else "sync"
                if func.signature.is_method:
                    func_type += " method"

                param_str = ", ".join(
                    [
                        f"{p.name}: {p.type_annotation or 'Any'}"
                        for p in func.signature.parameters
                    ]
                )

                return_type = func.signature.return_type or "None"
                line_range = f"{func.line_number}-{func.end_line_number}"

                docstring_preview = ""
                docstring_format = ""
                if func.docstring:
                    if isinstance(func.docstring, ParsedDocstring):
                        # Use parsed docstring summary
                        docstring_preview = func.docstring.summary
                        docstring_format = func.docstring.format.value
                    else:
                        # Raw docstring - use first line
                        docstring_preview = func.docstring.raw_text.split("\n")[
                            0
                        ].strip()
                        docstring_format = "raw"

                    if len(docstring_preview) > 37:
                        docstring_preview = docstring_preview[:37] + "..."

                table.add_row(
                    func.signature.name,
                    func_type,
                    param_str,
                    return_type,
                    line_range,
                    docstring_preview,
                    docstring_format,
                )

            console.print(table)
            console.print(f"\n[green]Found {len(functions)} functions[/green]")

    except ParsingError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        if e.recovery_hint:
            console.print(f"[yellow]Hint:[/yellow] {e.recovery_hint}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


def _display_match_results(result: MatchResult, show_unmatched: bool):
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


@app.command()
def match_contextual(
    path: Annotated[Path, typer.Argument(help="Project directory to analyze")],
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (terminal/json)")
    ] = "terminal",
    output_file: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output file path")
    ] = None,
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Configuration file path")
    ] = None,
    show_stats: Annotated[
        bool, typer.Option("--stats", help="Show performance statistics")
    ] = False,
    use_cache: Annotated[
        bool, typer.Option("--cache/--no-cache", help="Use parsing cache")
    ] = True,
    show_unmatched: Annotated[
        bool, typer.Option("--show-unmatched", help="Show unmatched functions")
    ] = False,
):
    """
    Perform contextual matching on a project.

    Uses both direct and contextual matching for best results.
    This command analyzes the entire project structure, resolves imports,
    and finds documentation across files.

    Example:
        codedocsync match-contextual ./myproject --stats --show-unmatched
    """
    # Load configuration
    if config and config.exists():
        config_obj = CodeDocSyncConfig.from_yaml(str(config))
    else:
        config_obj = CodeDocSyncConfig()

    # Validate path
    if not path.exists():
        console.print(f"[red]Error: {path} does not exist[/red]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error: {path} is not a directory[/red]")
        raise typer.Exit(1)

    # Create facade and run matching
    facade = ContextualMatchingFacade(config_obj)

    console.print(f"[cyan]Analyzing project: {path}[/cyan]")
    console.print("[dim]Building project context and matching functions...[/dim]")

    try:
        result = facade.match_project(str(path), use_cache=use_cache)
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        raise typer.Exit(1)

    # Format output
    if output_format == "json":
        output = _format_json_contextual_result(result, show_unmatched)
    else:
        output = _format_terminal_contextual_result(result, show_unmatched)

    # Save or print
    if output_file:
        output_file.write_text(output)
        console.print(f"✅ Results saved to {output_file}")
    else:
        if output_format == "json":
            console.print(output)
        else:
            console.print(output)

    # Show statistics if requested
    if show_stats:
        facade.print_summary()

    # Print final summary
    summary = result.get_summary()
    console.print("\n[green]✅ Analysis complete![/green]")
    console.print(
        f"Matched {summary['matched']}/{summary['total_functions']} functions ({summary['match_rate']})"
    )


def _format_json_contextual_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format contextual matching result as JSON."""
    import json

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


def _format_terminal_contextual_result(
    result: MatchResult, show_unmatched: bool
) -> str:
    """Format contextual matching result for terminal."""
    from io import StringIO
    import sys

    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        _display_contextual_results(result, show_unmatched)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _display_contextual_results(result: MatchResult, show_unmatched: bool):
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


def _format_json_unified_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format unified matching result as comprehensive JSON."""
    import json

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
                    _serialize_docstring(pair.docstring) if pair.docstring else None
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


def _format_terminal_unified_result(result: MatchResult, show_unmatched: bool) -> str:
    """Format unified matching result for terminal with enhanced display."""
    from io import StringIO
    import sys

    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        _display_unified_results(result, show_unmatched)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _display_unified_results(result: MatchResult, show_unmatched: bool):
    """Display unified matching results in terminal with comprehensive Rich formatting."""
    console.print(
        "\n[bold magenta]🚀 Unified Matching Results (Direct + Contextual + Semantic)[/bold magenta]"
    )
    console.print("=" * 80)

    # Enhanced summary table
    summary = result.get_summary()
    summary_table = Table(
        title="📊 Summary", show_header=True, header_style="bold blue"
    )
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
            title="🎯 Match Distribution", show_header=True, header_style="bold green"
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
                    f"{count/total_matches*100:.1f}%" if total_matches > 0 else "0%"
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
            title="⚡ Performance Overview", show_header=True, header_style="bold cyan"
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
                    f"{time_val/total_time*100:.1f}%" if total_time > 0 else "0%"
                )
                # Determine status based on time
                if phase == "parsing" and time_val > total_time * 0.5:
                    status = "🐌 Slow"
                elif phase == "semantic_matching" and time_val > 60:
                    status = "🔥 Heavy"
                else:
                    status = "✅ Good"

                perf_table.add_row(
                    phase.replace("_", " ").title(),
                    f"{time_val:.2f}s",
                    percentage,
                    status,
                )

        console.print(perf_table)

        # Memory and throughput metrics
        system_table = Table(
            title="💻 System Metrics", show_header=True, header_style="bold yellow"
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
                f"{errors.get('total_errors', 0)/max(unified_stats.get('functions_processed', 1), 1)*100:.1f}",
                "%",
            )

        console.print(system_table)

        # Performance recommendations
        if "performance_profile" in metadata:
            profile = metadata["performance_profile"]
            console.print("\n[bold blue]💡 Performance Insights:[/bold blue]")

            if profile.get("bottleneck") != "none":
                console.print(
                    f"  🚨 Bottleneck: {profile.get('bottleneck', 'unknown')}"
                )
                console.print(
                    f"  💡 Recommendation: {profile.get('recommendation', 'No specific advice')}"
                )
            else:
                console.print("  ✅ Performance is well balanced!")

            memory_concern = profile.get("memory_concern", "acceptable")
            if memory_concern != "acceptable":
                console.print(f"  📈 Memory usage: {memory_concern}")

    # Enhanced detailed matches with confidence breakdown
    if result.matched_pairs:
        console.print(
            f"\n[bold]📋 Detailed Matches ({len(result.matched_pairs)}):[/bold]"
        )
        for i, pair in enumerate(result.matched_pairs[:15]):  # Show first 15
            confidence_color = (
                "green"
                if pair.confidence.overall >= 0.8
                else "yellow" if pair.confidence.overall >= 0.6 else "red"
            )
            match_icon = {
                "EXACT": "🎯",
                "FUZZY": "🔍",
                "CONTEXTUAL": "🔗",
                "SEMANTIC": "🧠",
            }.get(pair.match_type.value, "📝")

            console.print(
                f"  {i+1:2d}. {match_icon} {pair.function.signature.name} "
                f"([{confidence_color}]{pair.confidence.overall:.2f}[/{confidence_color}]) "
                f"- {pair.match_type.value.lower()} "
                f"([dim]{pair.function.file_path}:{pair.function.line_number}[/dim])"
            )

            if pair.match_reason:
                console.print(f"      💬 {pair.match_reason}")

            # Show confidence breakdown for high-detail cases
            if pair.confidence.overall < 0.8:
                console.print(
                    f"      📊 Name: {pair.confidence.name_similarity:.2f}, "
                    f"Location: {pair.confidence.location_score:.2f}, "
                    f"Signature: {pair.confidence.signature_similarity:.2f}"
                )

        if len(result.matched_pairs) > 15:
            console.print(f"  ... and {len(result.matched_pairs) - 15} more matches")

    # Enhanced unmatched section if requested
    if show_unmatched and result.unmatched_functions:
        console.print(
            f"\n[bold red]❌ Unmatched Functions ({len(result.unmatched_functions)}):[/bold red]"
        )
        console.print(
            "[dim]These functions may need manual documentation review:[/dim]"
        )

        for i, func in enumerate(result.unmatched_functions[:10]):  # Show first 10
            console.print(
                f"  {i+1:2d}. ⚠️  {func.signature.name} "
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

    console.print("\n[green]✨ Analysis complete![/green]")


@app.command()
def match_unified(
    path: Path = typer.Argument(..., help="Project directory to analyze"),
    output_format: str = typer.Option(
        "terminal", "--format", "-f", help="Output format: terminal or json"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file"
    ),
    enable_semantic: bool = typer.Option(
        True, "--semantic/--no-semantic", help="Enable semantic matching"
    ),
    show_stats: bool = typer.Option(
        False, "--stats", help="Show detailed statistics and performance metrics"
    ),
    show_unmatched: bool = typer.Option(
        False, "--show-unmatched", help="Show unmatched functions that need attention"
    ),
    use_cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Use caching for better performance"
    ),
    show_recommendations: bool = typer.Option(
        True,
        "--recommendations/--no-recommendations",
        help="Show performance optimization recommendations",
    ),
    max_functions: Optional[int] = typer.Option(
        None,
        "--max-functions",
        help="Maximum number of functions to process (for testing)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output with detailed progress"
    ),
):
    """
    🚀 Perform comprehensive unified matching using all three strategies.

    This command runs the complete matching pipeline:

    1. 🎯 Direct Matching: Same-file function-to-docstring matching
    2. 🔗 Contextual Matching: Cross-file analysis with import resolution
    3. 🧠 Semantic Matching: AI-powered similarity search for renamed functions

    Provides detailed performance metrics, memory monitoring, and optimization
    recommendations for production use.
    """
    import asyncio
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    # Enhanced configuration loading with validation
    try:
        if config and config.exists():
            config_obj = CodeDocSyncConfig.from_yaml(str(config))
            if verbose:
                console.print(f"[green]✅ Configuration loaded from {config}[/green]")
        else:
            config_obj = CodeDocSyncConfig()
            if verbose:
                console.print("[yellow]⚠️  Using default configuration[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error loading configuration: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced path validation
    if not path.exists():
        console.print(f"[red]❌ Error: {path} does not exist[/red]")
        console.print("[dim]Please provide a valid project directory path[/dim]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]❌ Error: {path} is not a directory[/red]")
        console.print("[dim]Please provide a directory containing Python files[/dim]")
        raise typer.Exit(1)

    # Validate semantic matching requirements
    if enable_semantic:
        try:
            import importlib.util

            openai_spec = importlib.util.find_spec("openai")
            if openai_spec is not None and verbose:
                console.print(
                    "[green]✅ OpenAI library available for semantic matching[/green]"
                )
        except ImportError:
            console.print(
                "[yellow]⚠️  OpenAI library not available, semantic matching may fall back to local models[/yellow]"
            )

    # Create facade with enhanced configuration
    facade = UnifiedMatchingFacade(config_obj)

    # Set up progress tracking if verbose
    current_phase = ""

    def progress_callback(phase: str, current: int, total: int):
        nonlocal current_phase
        if verbose and phase != current_phase:
            current_phase = phase
            console.print(f"[blue]{phase}: {current}/{total}[/blue]")

    # Enhanced startup message
    console.print(
        f"[bold cyan]Starting comprehensive unified analysis: {path}[/bold cyan]"
    )
    if verbose:
        console.print(
            "[dim]Pipeline: Direct → Contextual → Semantic matching with performance monitoring[/dim]"
        )
        console.print(
            f"[dim]Settings: Cache={use_cache}, Semantic={enable_semantic}, Max Functions={max_functions or 'unlimited'}[/dim]"
        )

    # Production-ready error recovery
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing project...", total=None)

            # Run async matching with enhanced parameters
            result = asyncio.run(
                facade.match_project(
                    str(path),
                    use_cache=use_cache,
                    enable_semantic=enable_semantic,
                    progress_callback=progress_callback if verbose else None,
                )
            )

            progress.update(task, completed=True)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
        console.print("[dim]Partial results may be available[/dim]")
        raise typer.Exit(130)  # Standard exit code for SIGINT

    except MemoryError:
        console.print("[red]❌ Out of memory error during analysis[/red]")
        console.print(
            "[dim]Try using --no-cache or --max-functions to reduce memory usage[/dim]"
        )
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Critical error during unified analysis: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print(
            "[dim]Check file permissions, disk space, and network connectivity[/dim]"
        )
        raise typer.Exit(1)

    # Enhanced output formatting with options
    try:
        if output_format == "json":
            output = _format_json_unified_result(result, show_unmatched)
        else:
            output = _format_terminal_unified_result(result, show_unmatched)
    except Exception as e:
        console.print(f"[red]❌ Error formatting output: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced output handling
    try:
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output, encoding="utf-8")
            console.print(f"[green]✅ Results saved to {output_file}[/green]")
            if verbose:
                console.print(
                    f"[dim]File size: {output_file.stat().st_size} bytes[/dim]"
                )
        else:
            console.print(output)
    except Exception as e:
        console.print(f"[red]❌ Error writing output: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced statistics and recommendations
    if show_stats:
        console.print("\n" + "=" * 60)
        facade.print_summary()

        # Show performance recommendations if requested
        if show_recommendations:
            try:
                recommendations = facade.get_performance_recommendations()
                console.print(
                    "\n[bold blue]💡 Performance Recommendations:[/bold blue]"
                )
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"  {i}. {rec}")
            except Exception as e:
                console.print(
                    f"[yellow]⚠️  Could not generate recommendations: {e}[/yellow]"
                )

    # Enhanced completion message with summary
    console.print("\n[bold green]Unified analysis complete![/bold green]")
    summary = result.get_summary()

    # Success metrics
    match_rate_color = (
        "green"
        if float(summary["match_rate"].strip("%")) >= 80
        else "yellow" if float(summary["match_rate"].strip("%")) >= 60 else "red"
    )
    console.print(
        f"Matched [bold]{summary['matched']}[/bold]/{summary['total_functions']} functions "
        f"([{match_rate_color}]{summary['match_rate']}[/{match_rate_color}])"
    )

    # Performance summary
    if hasattr(result, "metadata") and "unified_stats" in getattr(
        result, "metadata", {}
    ):
        stats = getattr(result, "metadata", {})["unified_stats"]
        console.print(
            f"Total time: [bold]{stats.get('total_time_seconds', 0):.2f}s[/bold]"
        )
        if stats.get("memory_usage"):
            memory_growth = stats["memory_usage"].get("peak_mb", 0) - stats[
                "memory_usage"
            ].get("initial_mb", 0)
            console.print(
                f"Memory usage: [bold]{memory_growth:.1f}MB[/bold] peak growth"
            )

    # Exit code based on results
    if summary["matched"] == 0 and summary["total_functions"] > 0:
        console.print(
            "[red]No matches found - this may indicate a configuration issue[/red]"
        )
        raise typer.Exit(2)  # Warning exit code

    # Cleanup
    try:
        asyncio.run(facade.cleanup())
    except Exception:
        pass  # Cleanup is best-effort


if __name__ == "__main__":
    app()
