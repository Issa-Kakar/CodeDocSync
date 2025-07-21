"""
Parse command for CodeDocSync CLI.

This module contains the parse command which demonstrates the AST parser
functionality by analyzing Python files and extracting function definitions.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from codedocsync.cli.formatting import serialize_docstring
from codedocsync.parser import IntegratedParser, ParsedDocstring, ParsingError

console = Console()


def parse(
    file: Annotated[Path, typer.Argument(help="Python file to parse")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of pretty-printed table"),
    ] = False,
) -> None:
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
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1) from None
