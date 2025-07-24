"""
Console configuration for CodeDocSync CLI.

This module provides a centralized console configuration that handles
Windows terminal compatibility and Unicode rendering issues.
"""

import os
import platform
import sys

from rich.console import Console


def supports_unicode() -> bool:
    """Check if the terminal supports Unicode characters."""
    # Check if PYTHONIOENCODING is set to utf-8
    if os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
        return True
    # Windows Terminal sets WT_SESSION
    if os.environ.get("WT_SESSION"):
        return True
    # VS Code terminal
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    # Modern color terminal
    if os.environ.get("COLORTERM"):
        return True
    # Check for UTF-8 locale
    if os.environ.get("LANG", "").lower().endswith("utf-8"):
        return True
    # Check if stdout encoding supports unicode
    try:
        encoding = sys.stdout.encoding or "ascii"
        return encoding.lower().startswith("utf")
    except Exception:
        pass
    # Default to False on Windows
    return platform.system() != "Windows"


def create_console() -> Console:
    """Create a properly configured console for the current environment."""
    is_windows = platform.system() == "Windows"
    supports_fancy = supports_unicode()

    # Check for NO_COLOR environment variable
    no_color = bool(os.environ.get("NO_COLOR"))

    # Force ASCII output on Windows without proper Unicode support
    if is_windows and not supports_fancy:
        # Set environment to force ASCII everywhere in Rich
        os.environ["PYTHONIOENCODING"] = "ascii"

        # Legacy Windows terminal - use safe settings
        return Console(
            force_terminal=False,
            force_interactive=False,
            force_jupyter=False,
            legacy_windows=True,
            no_color=no_color,
            safe_box=True,  # Use ASCII box characters
            highlight=False,  # Disable syntax highlighting
            soft_wrap=True,
            emoji=False,  # Disable emoji
            _environ={"TERM": "dumb"},  # Force simple terminal
        )
    else:
        # Modern terminal with full support
        return Console(
            force_terminal=True,
            force_interactive=True,
            legacy_windows=False,
            no_color=no_color,
        )


# Create a singleton console instance
console = create_console()


def create_progress(*columns, **kwargs):
    """Create a Progress instance that respects console configuration."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    # If no columns specified, use defaults
    if not columns:
        if console.legacy_windows:
            # Use simple text indicator for legacy Windows (no spinner)
            columns = (
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[dim]...[/dim]"),
            )
        else:
            # Use fancy spinner for modern terminals
            columns = (
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
            )

    # Always use our configured console
    kwargs["console"] = console

    # Disable refresh on legacy Windows to avoid Unicode issues
    if console.legacy_windows:
        kwargs["refresh_per_second"] = 0.5  # Slower refresh rate
        kwargs["disable"] = False

    return Progress(*columns, **kwargs)
