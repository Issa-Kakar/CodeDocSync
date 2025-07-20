"""
Main entry point for the CodeDocSync CLI application.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from codedocsync import __version__
from codedocsync.analyzer import (
    AnalysisCache,
    analyze_matched_pair,
    analyze_multiple_pairs,
    get_development_config,
    get_fast_config,
    get_thorough_config,
)
from codedocsync.matcher import (
    ContextualMatchingFacade,
    MatchingFacade,
    MatchResult,
    UnifiedMatchingFacade,
)
from codedocsync.parser import IntegratedParser, ParsedDocstring, ParsingError
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
        Path,
        typer.Argument(
            help="File or directory to analyze for documentation inconsistencies"
        ),
    ] = Path("."),
    rules_only: Annotated[
        bool,
        typer.Option("--rules-only", help="Skip LLM analysis, use only rule engine"),
    ] = False,
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold", help="Confidence threshold for LLM routing"
        ),
    ] = 0.9,
    cache_dir: Annotated[
        Path | None, typer.Option("--cache-dir", help="Directory for analysis cache")
    ] = None,
    parallel: Annotated[
        bool, typer.Option("--parallel/--sequential", help="Use parallel analysis")
    ] = True,
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (terminal/json)")
    ] = "terminal",
    output_file: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path")
    ] = None,
    config_profile: Annotated[
        str,
        typer.Option("--profile", help="Analysis profile (fast/thorough/development)"),
    ] = "development",
    show_summary: Annotated[
        bool, typer.Option("--summary/--no-summary", help="Show analysis summary")
    ] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """
    Analyze code for documentation inconsistencies using rules and LLM.

    This command performs comprehensive analysis of function-documentation pairs
    to identify inconsistencies, missing documentation, and outdated descriptions.

    Examples:
        codedocsync analyze ./src --rules-only
        codedocsync analyze ./project --profile thorough --format json
    """
    import asyncio
    import time

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    # Validate path
    if not path.exists():
        console.print(f"[red]Error: {path} does not exist[/red]")
        raise typer.Exit(1)

    # Get configuration profile
    if config_profile == "fast":
        config = get_fast_config()
    elif config_profile == "thorough":
        config = get_thorough_config()
    else:
        config = get_development_config()

    # Apply command line overrides
    if rules_only:
        config.use_llm = False
    config.rule_engine.confidence_threshold = confidence_threshold
    config.parallel_analysis = parallel

    # Set up cache
    cache = AnalysisCache() if config.enable_cache else None

    if verbose:
        console.print(f"[cyan]Starting analysis of: {path}[/cyan]")
        console.print(
            f"[dim]Profile: {config_profile}, LLM: {config.use_llm}, Parallel: {parallel}[/dim]"
        )

    start_time = time.time()

    try:
        # First, get matched pairs using unified matching
        facade = UnifiedMatchingFacade()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
            transient=not verbose,
        ) as progress:
            # Match functions to documentation
            match_task = progress.add_task(
                "Matching functions to documentation...", total=None
            )

            if path.is_file():
                match_result = facade.match_file(path)
            else:
                match_result = asyncio.run(facade.match_project(str(path)))

            progress.update(match_task, completed=True)

            if not match_result.matched_pairs:
                console.print(
                    "[yellow]No function-documentation pairs found to analyze[/yellow]"
                )
                return

            # Analyze matched pairs
            analysis_task = progress.add_task(
                f"Analyzing {len(match_result.matched_pairs)} pairs...",
                total=len(match_result.matched_pairs),
            )

            # Run analysis
            results = await analyze_multiple_pairs(
                match_result.matched_pairs, config=config, cache=cache
            )

            progress.update(analysis_task, completed=len(results))

        total_time = time.time() - start_time

        # Aggregate results
        all_issues = []
        total_analysis_time = 0
        used_llm_count = 0
        cache_hits = 0

        for result in results:
            all_issues.extend(result.issues)
            total_analysis_time += result.analysis_time_ms
            if result.used_llm:
                used_llm_count += 1
            if result.cache_hit:
                cache_hits += 1

        # Format output
        if output_format == "json":
            output_data = {
                "summary": {
                    "total_pairs": len(results),
                    "total_issues": len(all_issues),
                    "critical_issues": len(
                        [i for i in all_issues if i.severity == "critical"]
                    ),
                    "high_issues": len([i for i in all_issues if i.severity == "high"]),
                    "medium_issues": len(
                        [i for i in all_issues if i.severity == "medium"]
                    ),
                    "low_issues": len([i for i in all_issues if i.severity == "low"]),
                    "used_llm": used_llm_count,
                    "cache_hits": cache_hits,
                    "total_time_seconds": total_time,
                    "analysis_time_ms": total_analysis_time,
                },
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "suggestion": issue.suggestion,
                        "line_number": issue.line_number,
                        "confidence": issue.confidence,
                        "details": issue.details,
                    }
                    for issue in all_issues
                ],
                "results": [
                    {
                        "function_name": result.matched_pair.function.signature.name,
                        "file_path": result.matched_pair.function.file_path,
                        "line_number": result.matched_pair.function.line_number,
                        "issues_count": len(result.issues),
                        "used_llm": result.used_llm,
                        "cache_hit": result.cache_hit,
                        "analysis_time_ms": result.analysis_time_ms,
                    }
                    for result in results
                ],
            }

            output_text = json.dumps(output_data, indent=2)
        else:
            # Terminal output
            output_text = _format_analysis_results(
                results, all_issues, total_time, show_summary
            )

        # Save or print output
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output_text, encoding="utf-8")
            console.print(f"[green]Results saved to {output_file}[/green]")
        else:
            console.print(output_text)

        # Print final summary
        if show_summary:
            console.print("\n[green]Analysis complete![/green]")
            console.print(
                f"Analyzed {len(results)} function-documentation pairs in {total_time:.2f}s"
            )
            console.print(f"Found {len(all_issues)} total issues")

            critical_count = len([i for i in all_issues if i.severity == "critical"])
            if critical_count > 0:
                console.print(
                    f"[red]{critical_count} critical issues require immediate attention[/red]"
                )

    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


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
        Path | None, typer.Option("--config", "-c", help="Config file")
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
        Path | None, typer.Option("--output", "-o", help="Output file path")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration file path")
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
        console.print(f"[OK] Results saved to {output_file}")
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
    console.print("\n[green][DONE] Analysis complete![/green]")
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
    import sys
    from io import StringIO

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
    import sys
    from io import StringIO

    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        _display_unified_results(result, show_unmatched)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _format_analysis_results(results, all_issues, total_time, show_summary):
    """Format analysis results for terminal output."""
    import sys
    from io import StringIO

    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        _display_analysis_results(results, all_issues, total_time, show_summary)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def _display_analysis_results(results, all_issues, total_time, show_summary):
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


def _display_unified_results(result: MatchResult, show_unmatched: bool):
    """Display unified matching results in terminal with comprehensive Rich formatting."""
    console.print(
        "\n[bold magenta][>>] Unified Matching Results (Direct + Contextual + Semantic)[/bold magenta]"
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


@app.command()
def match_unified(
    path: Path = typer.Argument(..., help="Project directory to analyze"),
    output_format: str = typer.Option(
        "terminal", "--format", "-f", help="Output format: terminal or json"
    ),
    output_file: Path | None = typer.Option(None, "--output", "-o", help="Output file"),
    config: Path | None = typer.Option(
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
    max_functions: int | None = typer.Option(
        None,
        "--max-functions",
        help="Maximum number of functions to process (for testing)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output with detailed progress"
    ),
):
    """
    Perform comprehensive unified matching using all three strategies.

    This command runs the complete matching pipeline:

    1. * Direct Matching: Same-file function-to-docstring matching
    2. * Contextual Matching: Cross-file analysis with import resolution
    3. * Semantic Matching: AI-powered similarity search for renamed functions

    Provides detailed performance metrics, memory monitoring, and optimization
    recommendations for production use.
    """
    import asyncio

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    # Enhanced configuration loading with validation
    try:
        if config and config.exists():
            config_obj = CodeDocSyncConfig.from_yaml(str(config))
            if verbose:
                console.print(f"[green][OK] Configuration loaded from {config}[/green]")
        else:
            config_obj = CodeDocSyncConfig()
            if verbose:
                console.print("[yellow][WARNING] Using default configuration[/yellow]")
    except Exception as e:
        console.print(f"[red][ERROR] Error loading configuration: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced path validation
    if not path.exists():
        console.print(f"[red][ERROR] Error: {path} does not exist[/red]")
        console.print("[dim]Please provide a valid project directory path[/dim]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red][ERROR] Error: {path} is not a directory[/red]")
        console.print("[dim]Please provide a directory containing Python files[/dim]")
        raise typer.Exit(1)

    # Validate semantic matching requirements
    if enable_semantic:
        try:
            import importlib.util

            openai_spec = importlib.util.find_spec("openai")
            if openai_spec is not None and verbose:
                console.print(
                    "[green][OK] OpenAI library available for semantic matching[/green]"
                )
        except ImportError:
            console.print(
                "[yellow][WARNING] OpenAI library not available, semantic matching may fall back to local models[/yellow]"
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
        console.print("\n[yellow][WARNING] Analysis interrupted by user[/yellow]")
        console.print("[dim]Partial results may be available[/dim]")
        raise typer.Exit(130)  # Standard exit code for SIGINT

    except MemoryError:
        console.print("[red][ERROR] Out of memory error during analysis[/red]")
        console.print(
            "[dim]Try using --no-cache or --max-functions to reduce memory usage[/dim]"
        )
        raise typer.Exit(1)

    except Exception as e:
        console.print(
            f"[red][ERROR] Critical error during unified analysis: {str(e)}[/red]"
        )
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
        console.print(f"[red][ERROR] Error formatting output: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced output handling
    try:
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output, encoding="utf-8")
            console.print(f"[green][OK] Results saved to {output_file}[/green]")
            if verbose:
                console.print(
                    f"[dim]File size: {output_file.stat().st_size} bytes[/dim]"
                )
        else:
            console.print(output)
    except Exception as e:
        console.print(f"[red][ERROR] Error writing output: {e}[/red]")
        raise typer.Exit(1)

    # Enhanced statistics and recommendations
    if show_stats:
        console.print("\n" + "=" * 60)
        facade.print_summary()

        # Show performance recommendations if requested
        if show_recommendations:
            try:
                recommendations = facade.get_performance_recommendations()
                console.print("\n[bold blue][Recommendations]:[/bold blue]")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"  {i}. {rec}")
            except Exception as e:
                console.print(
                    f"[yellow][WARNING] Could not generate recommendations: {e}[/yellow]"
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


@app.command()
def analyze_function(
    file: Annotated[Path, typer.Argument(help="Python file containing the function")],
    function_name: Annotated[
        str, typer.Argument(help="Name of the function to analyze")
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed analysis")
    ] = False,
    rules_only: Annotated[
        bool, typer.Option("--rules-only", help="Skip LLM analysis")
    ] = False,
    config_profile: Annotated[
        str,
        typer.Option("--profile", help="Analysis profile (fast/thorough/development)"),
    ] = "development",
):
    """
    Analyze a specific function in detail.

    This command analyzes a single function and its documentation,
    providing detailed insights and suggestions for improvement.

    Examples:
        codedocsync analyze-function ./myfile.py process_user --verbose
        codedocsync analyze-function ./src/utils.py validate_data --rules-only
    """
    from codedocsync.parser import IntegratedParser

    # Validate file
    if not file.exists():
        console.print(f"[red]Error: {file} does not exist[/red]")
        raise typer.Exit(1)

    if not file.is_file():
        console.print(f"[red]Error: {file} is not a file[/red]")
        raise typer.Exit(1)

    try:
        # Parse the file to find the function
        parser = IntegratedParser()
        functions = parser.parse_file(str(file))

        target_function = None
        for func in functions:
            if func.signature.name == function_name:
                target_function = func
                break

        if not target_function:
            console.print(
                f"[red]Error: Function '{function_name}' not found in {file}[/red]"
            )
            available_functions = [f.signature.name for f in functions]
            if available_functions:
                console.print(
                    f"[dim]Available functions: {', '.join(available_functions)}[/dim]"
                )
            raise typer.Exit(1)

        # Try to match with documentation
        facade = UnifiedMatchingFacade()
        match_result = facade.match_file(file)

        target_pair = None
        for pair in match_result.matched_pairs:
            if pair.function.signature.name == function_name:
                target_pair = pair
                break

        if not target_pair:
            console.print(
                f"[yellow]Warning: No documentation found for function '{function_name}'[/yellow]"
            )
            console.print("[dim]Analysis will focus on function structure only[/dim]")
            # Create a dummy pair for analysis
            from codedocsync.matcher import MatchConfidence, MatchedPair, MatchType

            target_pair = MatchedPair(
                function=target_function,
                documentation=None,
                confidence=MatchConfidence.HIGH,
                match_type=MatchType.DIRECT,
                match_reason="No documentation found",
            )

        # Get configuration
        if config_profile == "fast":
            config = get_fast_config()
        elif config_profile == "thorough":
            config = get_thorough_config()
        else:
            config = get_development_config()

        if rules_only:
            config.use_llm = False

        # Analyze the function
        console.print(f"[cyan]Analyzing function: {function_name}[/cyan]")
        if verbose:
            console.print(f"[dim]File: {file}[/dim]")
            console.print(f"[dim]Line: {target_function.line_number}[/dim]")
            console.print(
                f"[dim]Profile: {config_profile}, LLM: {config.use_llm}[/dim]"
            )

        result = await analyze_matched_pair(target_pair, config=config)

        # Display detailed results
        console.print(f"\n[bold]Analysis Results for {function_name}[/bold]")
        console.print("=" * 50)

        # Function information
        info_table = Table(title="Function Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Name", target_function.signature.name)
        info_table.add_row("File", str(file))
        info_table.add_row("Line", str(target_function.line_number))
        info_table.add_row(
            "Type", "async" if target_function.signature.is_async else "sync"
        )
        info_table.add_row("Parameters", str(len(target_function.signature.parameters)))
        info_table.add_row(
            "Return Type", target_function.signature.return_annotation or "None"
        )

        console.print(info_table)

        # Analysis summary
        console.print("\n[bold]Analysis Summary[/bold]")
        summary_table = Table()
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Issues", str(len(result.issues)))
        summary_table.add_row(
            "Critical", str(len([i for i in result.issues if i.severity == "critical"]))
        )
        summary_table.add_row(
            "High", str(len([i for i in result.issues if i.severity == "high"]))
        )
        summary_table.add_row(
            "Medium", str(len([i for i in result.issues if i.severity == "medium"]))
        )
        summary_table.add_row(
            "Low", str(len([i for i in result.issues if i.severity == "low"]))
        )
        summary_table.add_row("Used LLM", "Yes" if result.used_llm else "No")
        summary_table.add_row("Analysis Time", f"{result.analysis_time_ms:.1f}ms")

        console.print(summary_table)

        # Detailed issues
        if result.issues:
            console.print("\n[bold]Detected Issues[/bold]")
            for i, issue in enumerate(result.issues, 1):
                severity_color = {
                    "critical": "red",
                    "high": "yellow",
                    "medium": "blue",
                    "low": "dim",
                }.get(issue.severity, "white")

                console.print(
                    f"\n{i}. [{severity_color}]{issue.severity.upper()}[/{severity_color}]: {issue.issue_type}"
                )
                console.print(f"   {issue.description}")
                console.print(f"   [green]Suggestion:[/green] {issue.suggestion}")
                if verbose and issue.details:
                    console.print(f"   [dim]Details: {issue.details}[/dim]")
                console.print(f"   [dim]Confidence: {issue.confidence:.2f}[/dim]")
        else:
            console.print(
                "\n[green]No issues found! Function documentation is consistent.[/green]"
            )

        # Show function signature if verbose
        if verbose:
            console.print("\n[bold]Function Signature[/bold]")
            console.print(f"[dim]{target_function.signature.to_string()}[/dim]")

            if target_function.docstring:
                console.print("\n[bold]Docstring[/bold]")
                from codedocsync.parser import ParsedDocstring

                if isinstance(target_function.docstring, ParsedDocstring):
                    console.print(f"Format: {target_function.docstring.format}")
                    console.print(f"Summary: {target_function.docstring.summary}")
                else:
                    console.print(f"Raw: {target_function.docstring.raw_text[:200]}...")

    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def clear_cache(
    llm_only: Annotated[
        bool, typer.Option("--llm-only", help="Clear only LLM analysis cache")
    ] = False,
    older_than_days: Annotated[
        int, typer.Option("--older-than-days", help="Clear entries older than N days")
    ] = 7,
    confirm: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
):
    """
    Clear analysis cache.

    This command clears cached analysis results to free up disk space
    or force fresh analysis.

    Examples:
        codedocsync clear-cache --llm-only
        codedocsync clear-cache --older-than-days 30 --yes
    """
    from pathlib import Path

    from codedocsync.analyzer.llm_analyzer import LLMCache

    # Get cache directories
    cache_dir = Path.home() / ".cache" / "codedocsync"

    if not cache_dir.exists():
        console.print("[yellow]No cache directory found - nothing to clear[/yellow]")
        return

    # Calculate what will be cleared
    total_cleared = 0

    if llm_only:
        # Only clear LLM cache
        llm_cache_path = cache_dir / "llm_analysis_cache.db"
        if llm_cache_path.exists():
            if not confirm:
                response = typer.confirm(
                    "Clear LLM analysis cache? This will force re-analysis of all functions."
                )
                if not response:
                    console.print("[yellow]Cache clear cancelled[/yellow]")
                    return

            try:
                llm_cache = LLMCache(cache_dir=cache_dir)
                cleared_count = llm_cache.clear(older_than_hours=older_than_days * 24)
                total_cleared += cleared_count
                console.print(
                    f"[green]Cleared {cleared_count} LLM cache entries[/green]"
                )
            except Exception as e:
                console.print(f"[red]Error clearing LLM cache: {e}[/red]")
    else:
        # Clear all caches
        if not confirm:
            response = typer.confirm(
                f"Clear all analysis caches older than {older_than_days} days? "
                "This will force re-analysis and re-matching."
            )
            if not response:
                console.print("[yellow]Cache clear cancelled[/yellow]")
                return

        # Clear LLM cache
        llm_cache_path = cache_dir / "llm_analysis_cache.db"
        if llm_cache_path.exists():
            try:
                llm_cache = LLMCache(cache_dir=cache_dir)
                llm_cleared = llm_cache.clear(older_than_hours=older_than_days * 24)
                total_cleared += llm_cleared
                console.print(f"[green]Cleared {llm_cleared} LLM cache entries[/green]")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Error clearing LLM cache: {e}[/yellow]"
                )

        # Clear other cache files (AST cache, embedding cache, etc.)
        import time

        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        for cache_file in cache_dir.glob("**/*"):
            if cache_file.is_file() and cache_file.name != "llm_analysis_cache.db":
                try:
                    if cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        total_cleared += 1
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not remove {cache_file}: {e}[/yellow]"
                    )

        console.print(f"[green]Cleared {total_cleared} total cache entries[/green]")

    # Show remaining cache size
    try:
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        console.print(f"[dim]Remaining cache size: {total_size_mb:.1f} MB[/dim]")
    except Exception:
        pass  # Size calculation is best-effort


if __name__ == "__main__":
    app()
