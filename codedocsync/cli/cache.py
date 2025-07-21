"""
Cache management commands for CodeDocSync CLI.

This module contains commands for managing the analysis cache.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def clear_cache(
    llm_only: Annotated[
        bool, typer.Option("--llm-only", help="Clear only LLM analysis cache")
    ] = False,
    older_than_days: Annotated[
        int, typer.Option("--older-than-days", help="Clear entries older than N days")
    ] = 7,
    confirm: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
) -> None:
    """
    Clear analysis cache.

    This command clears cached analysis results to free up disk space
    or force fresh analysis.

    Examples:
        codedocsync clear-cache --llm-only
        codedocsync clear-cache --older-than-days 30 --yes
    """
    # Get cache directories
    cache_dir = Path.home() / ".cache" / "codedocsync"

    if not cache_dir.exists():
        console.print("[yellow]No cache directory found - nothing to clear[/yellow]")
        return

    # Calculate what will be cleared
    total_cleared = 0

    if llm_only:
        # Only clear LLM cache
        llm_cache_path = cache_dir / "llm_analysis_cache.db"
        if llm_cache_path.exists():
            if not confirm:
                response = typer.confirm(
                    "Clear LLM analysis cache? This will force re-analysis of all functions."
                )
                if not response:
                    console.print("[yellow]Cache clear cancelled[/yellow]")
                    return

            try:
                # LLMCache doesn't have a clear method, so we'll remove the cache file
                cache_size_mb = llm_cache_path.stat().st_size / (1024 * 1024)
                llm_cache_path.unlink()
                total_cleared += 1
                console.print(
                    f"[green]Cleared LLM cache ({cache_size_mb:.1f} MB)[/green]"
                )
            except Exception as e:
                console.print(f"[red]Error clearing LLM cache: {e}[/red]")
    else:
        # Clear all caches
        if not confirm:
            response = typer.confirm(
                f"Clear all analysis caches older than {older_than_days} days? "
                "This will force re-analysis and re-matching."
            )
            if not response:
                console.print("[yellow]Cache clear cancelled[/yellow]")
                return

        # Clear LLM cache
        llm_cache_path = cache_dir / "llm_analysis_cache.db"
        if llm_cache_path.exists():
            try:
                # LLMCache doesn't have a clear method, so we'll remove the cache file
                cache_size_mb = llm_cache_path.stat().st_size / (1024 * 1024)
                llm_cache_path.unlink()
                total_cleared += 1
                console.print(
                    f"[green]Cleared LLM cache ({cache_size_mb:.1f} MB)[/green]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Error clearing LLM cache: {e}[/yellow]"
                )

        # Clear other cache files (AST cache, embedding cache, etc.)
        import time

        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        for cache_file in cache_dir.glob("**/*"):
            if cache_file.is_file() and cache_file.name != "llm_analysis_cache.db":
                try:
                    if cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        total_cleared += 1
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not remove {cache_file}: {e}[/yellow]"
                    )

        console.print(f"[green]Cleared {total_cleared} total cache entries[/green]")

    # Show remaining cache size
    try:
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        console.print(f"[dim]Remaining cache size: {total_size_mb:.1f} MB[/dim]")
    except Exception:
        pass  # Size calculation is best-effort
