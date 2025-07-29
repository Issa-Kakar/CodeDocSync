"""
RAG corpus management commands for CodeDocSync CLI.

This module provides commands for managing the self-improving RAG corpus.
"""

import time
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.table import Table

from ..analyzer.models import ISSUE_TYPES
from ..cli.console import console
from ..parser import parse_python_file
from ..suggestions.acceptance_simulator import AcceptanceSimulator
from ..suggestions.metrics import get_metrics_collector
from ..suggestions.rag_corpus import RAGCorpusManager


def rag_stats(
    corpus_dir: Annotated[
        str,
        typer.Option(
            "--corpus-dir",
            "-d",
            help="Directory containing corpus JSON files",
        ),
    ] = "data",
) -> None:
    """
    Display statistics about the RAG corpus.

    Shows information about corpus size, quality distribution,
    and retrieval performance.
    """
    try:
        # Initialize RAG manager
        manager = RAGCorpusManager(corpus_dir=corpus_dir)

        # Get stats
        stats = manager.get_stats()

        # Create main stats table
        stats_table = Table(
            title="RAG Corpus Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        stats_table.add_column("Metric", style="bold", width=30)
        stats_table.add_column("Value", justify="right")

        # Add rows
        stats_table.add_row("Total Corpus Size", f"{stats['corpus_size']:,}")
        stats_table.add_row("Bootstrap Examples", f"{stats['examples_loaded']:,}")
        stats_table.add_row("Added Examples", f"{stats['examples_added']:,}")
        stats_table.add_row("", "")  # Separator
        stats_table.add_row(
            "Retrievals Performed", f"{stats['retrievals_performed']:,}"
        )

        if stats["retrievals_performed"] > 0:
            stats_table.add_row(
                "Avg Retrieval Time", f"{stats['average_retrieval_time_ms']:.2f} ms"
            )

        # Vector store stats
        vs_stats = stats.get("vector_store_stats", {})
        if vs_stats:
            stats_table.add_row("", "")  # Separator
            stats_table.add_row(
                "Embeddings Stored", f"{vs_stats.get('embeddings_stored', 0):,}"
            )
            stats_table.add_row(
                "Searches Performed", f"{vs_stats.get('searches_performed', 0):,}"
            )

            if vs_stats.get("searches_performed", 0) > 0:
                stats_table.add_row(
                    "Avg Search Time",
                    f"{vs_stats.get('average_search_time_ms', 0):.2f} ms",
                )
                stats_table.add_row(
                    "Cache Hit Rate", f"{vs_stats.get('cache_hit_rate', 0) * 100:.1f}%"
                )

        # Display
        console.print(stats_table)

        # Show corpus location
        corpus_path = Path(corpus_dir).absolute()
        console.print(f"\n[dim]Corpus location: {corpus_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: Failed to get RAG stats: {str(e)}[/red]")
        raise typer.Exit(1) from e


def accept_suggestion(
    file_path: Annotated[str, typer.Argument(help="Python file path")],
    function_name: Annotated[str, typer.Argument(help="Function name")],
    issue_type: Annotated[str, typer.Argument(help="Issue type that was fixed")],
    suggested_docstring: Annotated[
        str | None,
        typer.Option(
            "--docstring",
            "-d",
            help="The accepted docstring content (reads from stdin if not provided)",
        ),
    ] = None,
    docstring_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Docstring format (google, numpy, sphinx)",
        ),
    ] = "google",
    corpus_dir: Annotated[
        str,
        typer.Option(
            "--corpus-dir",
            help="Directory containing corpus JSON files",
        ),
    ] = "data",
) -> None:
    """
    Record an accepted suggestion to improve future suggestions.

    This command tells the RAG system that a user accepted a suggested
    docstring, allowing it to learn from successful patterns.

    Example:
        codedocsync accept-suggestion myfile.py calculate_total missing_docstring
    """
    try:
        # Parse the file to find the function
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            console.print(f"[red]Error: File not found: {file_path_obj}[/red]")
            raise typer.Exit(1)

        functions = parse_python_file(str(file_path_obj))

        # Find the specific function
        target_function = None
        for func in functions:
            if func.signature.name == function_name:
                target_function = func
                break

        if not target_function:
            console.print(
                f"[red]Error: Function '{function_name}' not found in {file_path_obj}[/red]"
            )
            raise typer.Exit(1)

        # Read docstring from stdin if not provided
        if not suggested_docstring:
            console.print("[dim]Enter the accepted docstring (Ctrl+D to finish):[/dim]")
            import sys

            suggested_docstring = sys.stdin.read().strip()

        if not suggested_docstring:
            console.print("[red]Error: No docstring content provided[/red]")
            raise typer.Exit(1)

        # Validate issue type
        if issue_type not in ISSUE_TYPES:
            valid_types = ", ".join(sorted(ISSUE_TYPES.keys()))
            console.print(
                f"[red]Error: Invalid issue type '{issue_type}'. Valid types: {valid_types}[/red]"
            )
            raise typer.Exit(1)

        # Initialize RAG manager and add the accepted suggestion
        manager = RAGCorpusManager(corpus_dir=corpus_dir)

        # Track acceptance (add before corpus update)
        metrics_collector = get_metrics_collector()

        # Note: Since we don't have the suggestion_id in the current flow,
        # we need to search for it based on context
        # This is a limitation that could be improved in the future

        manager.add_accepted_suggestion(
            function=target_function,
            suggested_docstring=suggested_docstring,
            docstring_format=docstring_format,
            issue_type=issue_type,
        )

        # Success message
        console.print(
            f"[green]Success: Added accepted suggestion for '{function_name}' to RAG corpus[/green]"
        )

        # Show brief stats
        stats = manager.get_stats()
        console.print(f"[dim]Corpus now contains {stats['corpus_size']} examples[/dim]")

        # Save metrics
        metrics_collector.save_session_metrics()

    except Exception as e:
        console.print(f"[red]Error: Failed to record suggestion: {str(e)}[/red]")
        raise typer.Exit(1) from e


def metrics_report(
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Save report to file",
        ),
    ] = None,
    last_days: Annotated[
        int,
        typer.Option(
            "--days",
            help="Include metrics from last N days",
        ),
    ] = 7,
) -> None:
    """Generate improvement metrics report."""
    from ..suggestions.metrics import ImprovementCalculator, MetricsCollector

    try:
        # Load metrics
        collector = MetricsCollector()

        # Get metrics from last N days
        cutoff = time.time() - (last_days * 86400)
        recent_metrics = [
            m
            for m in (collector.historical_metrics + collector.current_session)
            if m.timestamp > cutoff
        ]

        if not recent_metrics:
            console.print("[yellow]No metrics found for the specified period[/yellow]")
            return

        # Calculate improvements
        calculator = ImprovementCalculator(recent_metrics)
        report = calculator.generate_report()

        # Display report
        console.print(report)

        # Save to file if requested
        if output_file:
            output_file.write_text(report)
            console.print(f"\n[green]Report saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(code=1) from e


def simulate_acceptances(
    count: Annotated[
        int, typer.Argument(help="Number of acceptances to simulate")
    ] = 100,
    acceptance_rate_control: Annotated[
        float,
        typer.Option(
            "--acceptance-rate-control",
            help="Acceptance rate for control group",
        ),
    ] = 0.25,
    acceptance_rate_treatment: Annotated[
        float,
        typer.Option(
            "--acceptance-rate-treatment",
            help="Acceptance rate for treatment group",
        ),
    ] = 0.40,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Output directory for results",
        ),
    ] = Path("data/simulated"),
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = 42,
) -> None:
    """
    Simulate user acceptances to validate A/B testing framework.

    This command generates realistic acceptance patterns to test whether
    the RAG-enhanced suggestion system improves acceptance rates.
    """
    try:
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[bold blue]Simulating acceptances...", total=None)

            # Initialize simulator
            simulator = AcceptanceSimulator(
                control_acceptance_rate=acceptance_rate_control,
                treatment_acceptance_rate=acceptance_rate_treatment,
                output_dir=output_dir,
                seed=seed,
            )

            # Run simulation
            results = simulator.simulate(count)

            progress.remove_task(task)

        # Display results
        console.print(
            Panel(
                f"[green]✓[/green] Generated {results.total_suggestions} suggestions\n"
                f"[green]✓[/green] Control group: {results.control_accepted}/{results.control_total} "
                f"({results.control_rate:.1%} acceptance)\n"
                f"[green]✓[/green] Treatment group: {results.treatment_accepted}/{results.treatment_total} "
                f"({results.treatment_rate:.1%} acceptance)\n"
                f"[green]✓[/green] Overall improvement: [bold]{results.improvement_percentage:.1f}%[/bold]",
                title="Simulation Complete",
                title_align="left",
            )
        )

        # Show quality metrics
        metrics_table = Table(
            title="Quality Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Control", justify="right")
        metrics_table.add_column("Treatment", justify="right")
        metrics_table.add_column("Improvement", justify="right", style="green")

        # Calculate improvements
        completeness_improvement = (
            (
                results.metrics["avg_completeness_treatment"]
                - results.metrics["avg_completeness_control"]
            )
            / results.metrics["avg_completeness_control"]
            * 100
        )
        quality_improvement = (
            (
                results.metrics["avg_quality_treatment"]
                - results.metrics["avg_quality_control"]
            )
            / results.metrics["avg_quality_control"]
            * 100
        )

        metrics_table.add_row(
            "Avg Completeness",
            f"{results.metrics['avg_completeness_control']:.2f}",
            f"{results.metrics['avg_completeness_treatment']:.2f}",
            f"+{completeness_improvement:.1f}%",
        )
        metrics_table.add_row(
            "Avg Quality",
            f"{results.metrics['avg_quality_control']:.2f}",
            f"{results.metrics['avg_quality_treatment']:.2f}",
            f"+{quality_improvement:.1f}%",
        )

        console.print(metrics_table)

        # Show output location
        console.print(f"\n[dim]Results saved to: {output_dir.absolute()}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: Failed to run simulation: {str(e)}[/red]")
        raise typer.Exit(1) from e
