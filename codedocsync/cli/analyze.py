"""
Analyze commands for CodeDocSync CLI.

This module contains commands for analyzing code for documentation inconsistencies.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codedocsync.analyzer import (
    AnalysisCache,
    analyze_matched_pair,
    analyze_multiple_pairs,
    get_development_config,
    get_fast_config,
    get_thorough_config,
)
from codedocsync.cli.formatting import format_analysis_results
from codedocsync.matcher import (
    MatchConfidence,
    MatchedPair,
    MatchType,
    UnifiedMatchingFacade,
)
from codedocsync.parser import IntegratedParser, ParsedDocstring

console = Console()


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
) -> None:
    """
    Analyze code for documentation inconsistencies using rules and LLM.

    This command performs comprehensive analysis of function-documentation pairs
    to identify inconsistencies, missing documentation, and outdated descriptions.

    Examples:
        codedocsync analyze ./src --rules-only
        codedocsync analyze ./project --profile thorough --format json
    """
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
            results = asyncio.run(
                analyze_multiple_pairs(
                    match_result.matched_pairs, config=config, cache=cache
                )
            )

            progress.update(analysis_task, completed=len(results))

        total_time = time.time() - start_time

        # Aggregate results
        all_issues = []
        total_analysis_time = 0.0
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
            output_text = format_analysis_results(
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
        raise typer.Exit(1) from None


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
) -> None:
    """
    Analyze a specific function in detail.

    This command analyzes a single function and its documentation,
    providing detailed insights and suggestions for improvement.

    Examples:
        codedocsync analyze-function ./myfile.py process_user --verbose
        codedocsync analyze-function ./src/utils.py validate_data --rules-only
    """
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
            target_pair = MatchedPair(
                function=target_function,
                docstring=None,
                confidence=MatchConfidence(
                    overall=1.0,
                    name_similarity=1.0,
                    location_score=1.0,
                    signature_similarity=1.0,
                ),
                match_type=MatchType.EXACT,
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

        result = asyncio.run(analyze_matched_pair(target_pair, config=config))

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
            "Return Type", target_function.signature.return_type or "None"
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
        raise typer.Exit(1) from None
