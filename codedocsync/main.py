"""
Main entry point for the CodeDocSync CLI application.
"""

import json
import typer
from pathlib import Path
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table

from codedocsync import __version__
from codedocsync.parser import IntegratedParser, ParsingError, ParsedDocstring
from codedocsync.matcher import MatchingFacade, MatchResult
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
    config: Annotated[Path, typer.Option("--config", "-c", help="Config file")] = None,
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


if __name__ == "__main__":
    app()
