"""
Match commands for CodeDocSync CLI.

This module contains commands for matching functions to their documentation.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from codedocsync.cli.formatting import (
    display_match_results,
    format_json_contextual_result,
    format_json_unified_result,
    format_terminal_contextual_result,
    format_terminal_unified_result,
)
from codedocsync.matcher import (
    ContextualMatchingFacade,
    MatchingFacade,
    UnifiedMatchingFacade,
)
from codedocsync.utils.config import CodeDocSyncConfig

console = Console()


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
        display_match_results(result, show_unmatched)


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
        raise typer.Exit(1) from None

    # Format output
    if output_format == "json":
        output = format_json_contextual_result(result, show_unmatched)
    else:
        output = format_terminal_contextual_result(result, show_unmatched)

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
        raise typer.Exit(1) from None

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
        raise typer.Exit(130) from None  # Standard exit code for SIGINT

    except MemoryError:
        console.print("[red][ERROR] Out of memory error during analysis[/red]")
        console.print(
            "[dim]Try using --no-cache or --max-functions to reduce memory usage[/dim]"
        )
        raise typer.Exit(1) from None

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
        raise typer.Exit(1) from None

    # Enhanced output formatting with options
    try:
        if output_format == "json":
            output = format_json_unified_result(result, show_unmatched)
        else:
            output = format_terminal_unified_result(result, show_unmatched)
    except Exception as e:
        console.print(f"[red][ERROR] Error formatting output: {e}[/red]")
        raise typer.Exit(1) from None

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
        raise typer.Exit(1) from None

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
