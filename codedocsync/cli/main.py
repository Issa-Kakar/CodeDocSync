"""
Main entry point for the CodeDocSync CLI application.

This module sets up the Typer application and registers all commands
from the various submodules.
"""

from typing import Annotated

import typer

from codedocsync import __version__

# Import commands from submodules
from codedocsync.cli.analyze import analyze, analyze_function
from codedocsync.cli.cache import clear_cache
from codedocsync.cli.match import match, match_contextual, match_unified
from codedocsync.cli.parse import parse
from codedocsync.cli.placeholders import check, watch
from codedocsync.cli.suggest import suggest, suggest_interactive

# Create the main app
app = typer.Typer(
    help="CodeDocSync: An intelligent tool to find and fix documentation drift."
)


def version_callback(value: bool) -> None:
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
) -> None:
    """
    Manage documentation consistency checks for your codebase.
    """
    pass


# Register all commands
app.command("analyze")(analyze)
app.command("analyze-function")(analyze_function)
app.command("match")(match)
app.command("match-contextual")(match_contextual)
app.command("match-unified")(match_unified)
app.command("parse")(parse)
app.command("clear-cache")(clear_cache)
app.command("suggest")(suggest)
app.command("suggest-interactive")(suggest_interactive)
app.command("watch")(watch)
app.command("check")(check)


if __name__ == "__main__":
    app()
