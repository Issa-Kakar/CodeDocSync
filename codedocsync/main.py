"""
Main entry point for the CodeDocSync CLI application.
"""

import typer
from typing_extensions import Annotated

from codedocsync import __version__

app = typer.Typer(
    help="CodeDocSync: An intelligent tool to find and fix documentation drift."
)


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


if __name__ == "__main__":
    app()
