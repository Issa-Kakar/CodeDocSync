"""
Main entry point for the CodeDocSync CLI application.

This module sets up the Typer application and registers all commands
from the various submodules.
"""

from typing import Annotated

import typer
from dotenv import load_dotenv

from codedocsync import __version__

# Import console for reconfiguration
from codedocsync.cli import console as console_module

# Import commands from submodules
from codedocsync.cli.analyze import analyze, analyze_function
from codedocsync.cli.cache import clear_cache
from codedocsync.cli.match import match, match_contextual, match_unified
from codedocsync.cli.parse import parse
from codedocsync.cli.placeholders import check, watch
from codedocsync.cli.rag import accept_suggestion, rag_stats
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
    ctx: typer.Context,
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
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output", envvar="NO_COLOR"),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Use plain text output (no Unicode or colors)"),
    ] = False,
    force_color: Annotated[
        bool,
        typer.Option(
            "--force-color",
            help="Force colored output even if terminal doesn't support it",
        ),
    ] = False,
) -> None:
    """
    Manage documentation consistency checks for your codebase.
    """
    # Load environment variables from .env file
    load_dotenv()
    # Handle terminal output configuration
    if plain or no_color:
        # Recreate console with plain settings
        from rich.console import Console

        console_module.console = Console(
            no_color=True,
            force_terminal=False,
            highlight=False,
            emoji=False,
            markup=False,
        )
    elif force_color:
        # Force color output
        from rich.console import Console

        console_module.console = Console(
            force_terminal=True, force_interactive=True, color_system="standard"
        )

    # Store in context for subcommands (if needed)
    ctx.obj = {"console": console_module.console}


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
app.command("rag-stats")(rag_stats)
app.command("accept-suggestion")(accept_suggestion)


if __name__ == "__main__":
    app()
