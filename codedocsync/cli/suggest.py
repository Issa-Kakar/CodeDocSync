"""
Suggest commands for CodeDocSync CLI.

This module contains commands for generating documentation fix suggestions.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from codedocsync.analyzer import analyze_multiple_pairs, get_development_config
from codedocsync.matcher import UnifiedMatchingFacade
from codedocsync.suggestions import (
    SuggestionConfig,
    enhance_multiple_with_suggestions,
)
from codedocsync.suggestions.formatters import (
    JSONSuggestionFormatter,
    TerminalSuggestionFormatter,
)
from codedocsync.suggestions.ranking import create_balanced_ranker

console = Console()


@typer.command()
def suggest(
    path: Annotated[
        Path,
        typer.Argument(help="File or function to generate suggestions for"),
    ] = Path("."),
    style: Annotated[
        str | None,
        typer.Option("--style", "-s", help="Docstring style (google/numpy/sphinx)"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (terminal/json)"),
    ] = "terminal",
    apply: Annotated[
        bool,
        typer.Option("--apply", "-a", help="Apply suggestions directly to files"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Interactive suggestion review"),
    ] = False,
    confidence_threshold: Annotated[
        float,
        typer.Option("--confidence", help="Minimum confidence for suggestions"),
    ] = 0.7,
    severity: Annotated[
        str | None,
        typer.Option(
            "--severity", help="Filter by severity (critical/high/medium/low)"
        ),
    ] = None,
    issue_type: Annotated[
        str | None,
        typer.Option("--issue-type", help="Filter by issue type"),
    ] = None,
    show_diff: Annotated[
        bool,
        typer.Option("--diff", "-d", help="Show diff view of suggestions"),
    ] = True,
    backup: Annotated[
        bool,
        typer.Option("--backup/--no-backup", help="Create backup before applying"),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be changed without applying"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
):
    """
    Generate fix suggestions for documentation issues.

    This command analyzes code for documentation inconsistencies and generates
    actionable, copy-paste ready suggestions to fix them.

    Examples:
        codedocsync suggest ./src/utils.py --style google
        codedocsync suggest ./project --severity critical --format json
        codedocsync suggest ./main.py::process_data --interactive
        codedocsync suggest . --apply --dry-run
    """
    # Validate path
    if not path.exists():
        console.print(f"[red]Error: {path} does not exist[/red]")
        raise typer.Exit(1)

    # Parse function specification if provided (file.py::function_name)
    target_function = None
    if "::" in str(path):
        path_str, function_name = str(path).split("::", 1)
        path = Path(path_str)
        target_function = function_name

    if verbose:
        console.print(f"[cyan]Generating suggestions for: {path}[/cyan]")
        if target_function:
            console.print(f"[dim]Target function: {target_function}[/dim]")
        console.print(f"[dim]Style: {style or 'auto-detect'}[/dim]")

    try:
        # First, analyze the code to find issues
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Step 1: Match functions to documentation
            match_task = progress.add_task(
                "Finding documentation issues...", total=None
            )

            # Use unified matching to get pairs
            facade = UnifiedMatchingFacade()
            if path.is_file():
                match_result = facade.match_file(path)
            else:
                match_result = asyncio.run(facade.match_project(str(path)))

            progress.update(match_task, completed=True)

            if not match_result.matched_pairs:
                console.print("[yellow]No function-documentation pairs found[/yellow]")
                return

            # Filter by target function if specified
            if target_function:
                match_result.matched_pairs = [
                    p
                    for p in match_result.matched_pairs
                    if p.function.signature.name == target_function
                ]
                if not match_result.matched_pairs:
                    console.print(f"[red]Function '{target_function}' not found[/red]")
                    raise typer.Exit(1)

            # Step 2: Analyze for issues
            analysis_task = progress.add_task(
                f"Analyzing {len(match_result.matched_pairs)} functions...",
                total=len(match_result.matched_pairs),
            )

            # Get analysis configuration
            analysis_config = get_development_config()
            analysis_config.use_llm = False  # Suggestions don't need LLM analysis

            # Analyze all pairs
            analysis_results = asyncio.run(analyze_multiple_pairs(
                match_result.matched_pairs, config=analysis_config
            ))

            progress.update(analysis_task, completed=len(analysis_results))

            # Step 3: Generate suggestions
            suggest_task = progress.add_task(
                "Generating suggestions...", total=len(analysis_results)
            )

            # Create suggestion configuration
            suggestion_config = SuggestionConfig(
                default_style=style or "google",
                preserve_descriptions=True,
                include_types=True,
                confidence_threshold=confidence_threshold,
            )

            # Enhance results with suggestions
            enhanced_results = enhance_multiple_with_suggestions(
                analysis_results, suggestion_config
            )

            progress.update(suggest_task, completed=len(enhanced_results))

        # Filter and rank suggestions
        all_suggestions = []
        for result in enhanced_results:
            for issue in result.enhanced_issues:
                if issue.suggestion_object:
                    # Apply filters
                    if severity and issue.severity != severity:
                        continue
                    if issue_type and issue.issue_type != issue_type:
                        continue
                    if issue.suggestion_object.confidence < confidence_threshold:
                        continue

                    all_suggestions.append((result, issue))

        if not all_suggestions:
            console.print(
                "[yellow]No suggestions match the specified criteria[/yellow]"
            )
            return

        # Rank suggestions
        ranker = create_balanced_ranker()
        ranked_issues = ranker.rank_suggestions(
            [issue for _, issue in all_suggestions], ranker.config
        )

        # Prepare output
        if output_format == "json":
            formatter = JSONSuggestionFormatter()
            output_data = {
                "summary": {
                    "total_suggestions": len(all_suggestions),
                    "total_functions": len(enhanced_results),
                    "confidence_threshold": confidence_threshold,
                    "style": style or "auto-detected",
                },
                "suggestions": [
                    {
                        "function": result.matched_pair.function.signature.name,
                        "file": result.matched_pair.function.file_path,
                        "line": result.matched_pair.function.line_number,
                        "issue": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "suggestion": formatter.format(issue.suggestion_object),
                    }
                    for result, issue in all_suggestions
                    if issue in ranked_issues
                ],
            }
            console.print(json.dumps(output_data, indent=2))

        else:
            # Terminal output
            formatter = TerminalSuggestionFormatter()

            # Display summary
            console.print(
                Panel(
                    f"[bold]Found {len(all_suggestions)} suggestions[/bold]\n"
                    f"Functions analyzed: {len(enhanced_results)}\n"
                    f"Style: {style or 'auto-detected'}",
                    title="Suggestion Summary",
                    border_style="blue",
                )
            )

            # Interactive mode
            if interactive:
                console.print("\n[bold]Interactive Suggestion Review[/bold]")
                console.print(
                    "[dim]Press Enter to accept, 's' to skip, 'q' to quit[/dim]\n"
                )

                for i, (result, issue) in enumerate(all_suggestions):
                    if issue not in ranked_issues:
                        continue

                    console.print(
                        f"\n[bold]Suggestion {i + 1}/{len(all_suggestions)}[/bold]"
                    )
                    console.print(
                        formatter.format(
                            issue.suggestion_object,
                            style=formatter.OutputStyle.RICH,
                            show_diff=show_diff,
                        )
                    )

                    if apply and not dry_run:
                        response = Confirm.ask("Apply this suggestion?", default=True)
                        if response:
                            apply_suggestion(
                                result.matched_pair.function.file_path,
                                issue.suggestion_object,
                                backup=backup,
                            )
                            console.print("[green]Applied![/green]")

            else:
                # Batch display
                displayed = 0
                for result, issue in all_suggestions:
                    if issue not in ranked_issues:
                        continue

                    console.print("\n" + "=" * 80)
                    console.print(
                        formatter.format(
                            issue.suggestion_object,
                            style=formatter.OutputStyle.RICH,
                            show_diff=show_diff,
                        )
                    )

                    displayed += 1
                    if displayed >= 10 and not verbose:
                        remaining = len(ranked_issues) - displayed
                        if remaining > 0:
                            console.print(
                                f"\n[dim]... and {remaining} more suggestions. "
                                f"Use --verbose to see all.[/dim]"
                            )
                        break

        # Apply suggestions if requested
        if apply and not interactive:
            if dry_run:
                console.print(
                    "\n[yellow]DRY RUN: The following changes would be made:[/yellow]"
                )
                for result, issue in all_suggestions[:5]:
                    console.print(
                        f"  • {result.matched_pair.function.file_path}:"
                        f"{issue.line_number} - {issue.issue_type}"
                    )
                if len(all_suggestions) > 5:
                    console.print(f"  ... and {len(all_suggestions) - 5} more")
            else:
                if not Confirm.ask(
                    f"\nApply {len(all_suggestions)} suggestions?", default=False
                ):
                    console.print("[yellow]Cancelled[/yellow]")
                    return

                applied = 0
                failed = 0

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    apply_task = progress.add_task(
                        "Applying suggestions...", total=len(all_suggestions)
                    )

                    for result, issue in all_suggestions:
                        try:
                            apply_suggestion(
                                result.matched_pair.function.file_path,
                                issue.suggestion_object,
                                backup=backup,
                            )
                            applied += 1
                        except Exception as e:
                            if verbose:
                                console.print(
                                    f"[red]Failed to apply suggestion: {e}[/red]"
                                )
                            failed += 1

                        progress.update(apply_task, advance=1)

                console.print(
                    f"\n[green]Successfully applied {applied} suggestions[/green]"
                )
                if failed > 0:
                    console.print(f"[red]Failed to apply {failed} suggestions[/red]")

    except Exception as e:
        console.print(f"[red]Error generating suggestions: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


def apply_suggestion(file_path: str, suggestion, backup: bool = True) -> None:
    """Apply a suggestion to a file."""
    file = Path(file_path)

    # Create backup if requested
    if backup:
        backup_path = file.with_suffix(file.suffix + ".bak")
        shutil.copy2(file, backup_path)

    # Read current content
    content = file.read_text(encoding="utf-8")

    # Apply the suggestion
    # This is a simplified implementation - in production you'd want
    # more sophisticated replacement logic
    if suggestion.original_text in content:
        new_content = content.replace(
            suggestion.original_text, suggestion.suggested_text, 1
        )
        file.write_text(new_content, encoding="utf-8")
    else:
        raise ValueError("Could not find original text in file")


@typer.command()
def suggest_interactive(
    path: Annotated[
        Path,
        typer.Argument(help="File or directory to analyze"),
    ] = Path("."),
):
    """
    Launch interactive suggestion browser.

    This provides a TUI (Text User Interface) for browsing and applying
    documentation fix suggestions interactively.

    Examples:
        codedocsync suggest-interactive ./src
        codedocsync suggest-interactive ./main.py
    """
    console.print(
        "[yellow]Interactive mode is not yet implemented in this version.[/yellow]"
    )
    console.print("[dim]Use 'suggest --interactive' for basic interactive mode.[/dim]")
    raise typer.Exit(0)
