"""
Parse command for CodeDocSync CLI.

This module contains the parse command which demonstrates the AST parser
functionality by analyzing Python files and extracting function definitions.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from codedocsync.cli.console import console
from codedocsync.cli.formatting import serialize_docstring
from codedocsync.parser import IntegratedParser, ParsedDocstring, ParsingError


def parse(
    path: Annotated[Path, typer.Argument(help="Path to Python file or directory")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of pretty-printed table"),
    ] = False,
) -> None:
    """
    Parse Python files and display extracted functions.

    This command demonstrates the AST parser functionality by analyzing
    Python files and extracting all function definitions with their
    signatures, parameters, and docstrings.
    """
    try:
        # Handle directory vs file
        if path.is_dir():
            # Find all Python files in directory
            python_files = list(path.rglob("*.py"))
            if not python_files:
                console.print(f"[red]No Python files found in {path}[/red]")
                raise typer.Exit(1)
        elif path.is_file() and path.suffix == ".py":
            python_files = [path]
        else:
            console.print(f"[red]Error: {path} is not a Python file or directory[/red]")
            raise typer.Exit(1)

        # Parse all files
        all_functions = []
        parser = IntegratedParser()

        for file_path in python_files:
            try:
                functions = parser.parse_file(str(file_path))
                all_functions.extend(functions)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to parse {file_path}: {e}[/yellow]"
                )
                continue

        functions = all_functions

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
                    "docstring": serialize_docstring(func.docstring),
                    "signature_string": func.signature.to_string(),
                }
                json_data.append(func_data)

            console.print(json.dumps(json_data, indent=2))
        else:
            # Pretty print with Rich
            if not functions:
                console.print("[yellow]No functions found in the file.[/yellow]")
                return

            # Create appropriate title
            if len(python_files) == 1:
                title = f"Functions in {python_files[0]}"
            else:
                title = f"Functions in {len(python_files)} files from {path}"

            table = Table(title=title)

            # Add file column if parsing multiple files
            if len(python_files) > 1:
                table.add_column("File", style="dim")

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

                # Build row data
                row_data = []

                # Add file name if parsing multiple files
                if len(python_files) > 1:
                    file_name = (
                        Path(func.file_path).name if func.file_path else "unknown"
                    )
                    row_data.append(file_name)

                row_data.extend(
                    [
                        func.signature.name,
                        func_type,
                        param_str,
                        return_type,
                        line_range,
                        docstring_preview,
                        docstring_format,
                    ]
                )

                table.add_row(*row_data)

            console.print(table)
            console.print(f"\n[green]Found {len(functions)} functions[/green]")

    except ParsingError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        if e.recovery_hint:
            console.print(f"[yellow]Hint:[/yellow] {e.recovery_hint}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1) from None
