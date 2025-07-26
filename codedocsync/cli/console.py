"""
Console configuration for CodeDocSync CLI.

This module provides a centralized console configuration that handles
Windows terminal compatibility and Unicode rendering issues.
"""

import os
import platform
import sys

from rich.console import Console

# Fix Windows encoding issues
if platform.system() == "Windows":
    # Set environment for future processes
    if not os.environ.get("PYTHONIOENCODING"):
        os.environ["PYTHONIOENCODING"] = "utf-8"

    # Fix current process encoding
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def supports_unicode() -> bool:
    """Check if the terminal supports Unicode characters."""
    # Check for explicit encoding
    if os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
        return True

    # Check for Windows Terminal (modern, supports everything)
    if os.environ.get("WT_SESSION"):
        return True

    # Check for VS Code terminal
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True

    # Check for Git Bash / MINGW
    if "MINGW" in os.environ.get("MSYSTEM", ""):
        return True

    # Check for Git Bash via path
    if "Git" in os.environ.get("PATH", "") and os.environ.get("TERM"):
        return True

    # CRITICAL: Check for PowerShell specifically
    # PowerShell sets PSModulePath, and without Windows Terminal, it needs legacy mode
    if os.environ.get("PSModulePath") and not os.environ.get("WT_SESSION"):
        # Old PowerShell without Windows Terminal wrapper
        return False

    # Check for PowerShell ISE (always needs legacy mode)
    if os.environ.get("PSModulePath") and os.environ.get("PSISE"):
        return False

    # Check for modern color terminal
    if os.environ.get("COLORTERM"):
        return True

    # Check for UTF-8 locale
    lang = os.environ.get("LANG", "").lower()
    if lang.endswith("utf-8") or lang.endswith("utf8"):
        return True

    # Check stdout encoding
    try:
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            encoding = sys.stdout.encoding.lower()
            # Explicitly reject Windows codepages that can't handle Unicode
            if encoding.startswith("cp") and encoding != "cp65001":
                return False
            return encoding in ("utf-8", "utf8", "utf-8-sig", "cp65001")
    except Exception:
        pass

    # Default based on platform
    return platform.system() != "Windows"


def create_console() -> Console:
    """Create a properly configured console for the current environment."""
    is_windows = platform.system() == "Windows"

    # Detect if we need legacy Windows mode
    if is_windows:
        # Try to enable virtual terminal processing on Windows
        try:
            import ctypes

            # Check if windll exists (Windows-specific)
            if hasattr(ctypes, "windll"):
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences (ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004)
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                mode.value |= 0x0004
                kernel32.SetConsoleMode(handle, mode)
        except Exception:
            # If we can't enable it, we'll use legacy mode
            pass

    supports_fancy = supports_unicode()

    # Check for NO_COLOR environment variable
    no_color = bool(os.environ.get("NO_COLOR"))

    # Force ASCII output on Windows without proper Unicode support
    if is_windows and not supports_fancy:
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
            markup=True,  # Keep markup for color codes
            log_time_format="[%X]",
        )
    else:
        # Modern terminal with full support
        return Console(
            force_terminal=True,
            force_interactive=True,
            legacy_windows=False,
            no_color=no_color,
            highlight=True,
            emoji=True,
            markup=True,
            log_time_format="[%X]",
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
