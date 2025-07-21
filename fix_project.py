#!/usr/bin/env python3
"""
Project Fix Coordinator - Systematically fixes linting and type issues
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class ProjectFixer:
    """Coordinates systematic fixing of linting and type issues."""

    def __init__(self):
        self.console = console
        self.progress_file = Path(".fix_progress.json")
        self.load_progress()
        # Use direct Python path for Windows compatibility
        self.python_path = r"C:\Users\issak\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe"

    def load_progress(self):
        """Load progress from JSON file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "phase": "start",
                "ruff_fixed": [],
                "mypy_fixed": [],
                "timestamp": datetime.now().isoformat(),
            }

    def save_progress(self):
        """Save progress to JSON file."""
        self.progress["timestamp"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def run_command(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run command and return (returncode, stdout, stderr)."""
        # Replace poetry run commands with direct Python path
        if cmd[0] == "poetry" and cmd[1] == "run":
            # Convert to direct Python execution
            tool = cmd[2]
            args = cmd[3:]
            if tool in ["ruff", "black", "mypy"]:
                cmd = [self.python_path, "-m", tool] + args
            else:
                cmd = [self.python_path, "-m"] + cmd[2:]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr

    def fix_ruff_auto(self):
        """Phase 1: Auto-fix ruff issues."""
        self.console.print(
            "\n[bold blue]Phase 1: Auto-fixing Ruff issues...[/bold blue]"
        )

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("Running ruff --fix...", total=None)
            returncode, stdout, stderr = self.run_command(
                ["poetry", "run", "ruff", "check", ".", "--fix"]
            )
            progress.update(task, completed=True)

        if returncode == 0:
            self.console.print("[green]✓ Ruff auto-fixes applied successfully![/green]")
        else:
            self.console.print("[yellow]⚠ Ruff completed with warnings[/yellow]")
            if stderr:
                self.console.print(f"[dim]{stderr}[/dim]")

    def format_black(self):
        """Phase 2: Format with Black."""
        self.console.print("\n[bold blue]Phase 2: Formatting with Black...[/bold blue]")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("Running black...", total=None)
            returncode, stdout, stderr = self.run_command(
                ["poetry", "run", "black", "."]
            )
            progress.update(task, completed=True)

        if stdout:
            self.console.print(stdout)
        self.console.print("[green]✓ Black formatting complete![/green]")

    def get_remaining_issues(self) -> tuple[int, int]:
        """Get current count of issues."""
        # Ruff
        _, ruff_out, _ = self.run_command(
            ["poetry", "run", "ruff", "check", ".", "--format", "json"]
        )
        ruff_issues = 0
        if ruff_out:
            try:
                ruff_data = json.loads(ruff_out)
                ruff_issues = len(ruff_data)
            except json.JSONDecodeError:
                # Fallback to counting lines
                _, ruff_out, _ = self.run_command(
                    ["poetry", "run", "ruff", "check", "."]
                )
                ruff_issues = len(
                    [
                        line
                        for line in ruff_out.splitlines()
                        if line.strip() and not line.startswith("warning:")
                    ]
                )

        # Mypy - count lines since JSON format gives one error per line
        _, mypy_out, _ = self.run_command(
            ["poetry", "run", "mypy", ".", "--format", "json"]
        )
        mypy_issues = len(mypy_out.strip().splitlines()) if mypy_out else 0

        return ruff_issues, mypy_issues

    def show_status(self):
        """Display current status."""
        ruff_count, mypy_count = self.get_remaining_issues()

        table = Table(title="Project Status")
        table.add_column("Issue Type", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Status", style="green")

        ruff_status = "✓ Complete" if ruff_count == 0 else f"⚠ {ruff_count} remaining"
        mypy_status = "✓ Complete" if mypy_count == 0 else f"⚠ {mypy_count} remaining"

        table.add_row("Ruff Issues", str(ruff_count), ruff_status)
        table.add_row("Mypy Issues", str(mypy_count), mypy_status)

        self.console.print(table)

        # Show breakdown of ruff issues
        if ruff_count > 0:
            self.show_ruff_breakdown()

    def show_ruff_breakdown(self):
        """Show breakdown of remaining ruff issues by type."""
        _, output, _ = self.run_command(
            ["poetry", "run", "ruff", "check", ".", "--format", "concise"]
        )

        if not output:
            return

        # Count by error code
        error_counts = {}
        for line in output.splitlines():
            if ":" in line and not line.startswith("warning:"):
                parts = line.split(":")
                if len(parts) >= 3:
                    # Extract error code from the message
                    msg_part = parts[2].strip()
                    if " " in msg_part:
                        code = msg_part.split()[0]
                        if code.startswith(("B", "UP", "F", "C")):  # Common ruff codes
                            error_counts[code] = error_counts.get(code, 0) + 1

        if error_counts:
            self.console.print("\n[bold]Ruff Issues by Type:[/bold]")
            for code, count in sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            ):
                self.console.print(f"  {code}: {count} occurrences")
                # Show description for common codes
                descriptions = {
                    "B904": "raise-without-from - Add 'from e' to exception chains",
                    "UP038": "isinstance - Use X | Y instead of (X, Y)",
                    "B007": "unused-loop-control-variable - Rename to _ or use the variable",
                    "F841": "unused-variable - Remove or use the variable",
                    "B017": "assert-raises-exception - Don't assert blind Exception",
                    "UP035": "deprecated-import - Use modern imports",
                    "B023": "function-uses-loop-variable - Function doesn't bind loop variable",
                }
                if code in descriptions:
                    self.console.print(f"    [dim]{descriptions[code]}[/dim]")

    def get_fix_patterns(self) -> dict[str, str]:
        """Get common fix patterns for ruff issues."""
        return {
            "B904": """
# Fix: Add 'from e' to exception chains
# Before:
except SomeError as e:
    raise NewError("message")

# After:
except SomeError as e:
    raise NewError("message") from e
""",
            "UP038": """
# Fix: Use union types instead of tuples in isinstance
# Before:
isinstance(x, (str, int))

# After:
isinstance(x, str | int)
""",
            "B007": """
# Fix: Rename unused loop variables to _
# Before:
for item in items:
    print("processing")

# After:
for _ in items:
    print("processing")
""",
            "F841": """
# Fix: Remove or use unused variables
# Before:
result = some_function()  # never used

# After:
# Remove the line or use the variable
""",
        }

    def show_fix_examples(self, error_code: str | None = None):
        """Show examples of how to fix common issues."""
        patterns = self.get_fix_patterns()

        if error_code and error_code in patterns:
            self.console.print(
                Panel(patterns[error_code], title=f"How to fix {error_code}")
            )
        else:
            self.console.print("\n[bold]Common Fix Patterns:[/bold]")
            for code, pattern in patterns.items():
                self.console.print(Panel(pattern, title=code))


@click.command()
@click.option(
    "--phase",
    type=click.Choice(["all", "ruff", "mypy", "status", "examples"]),
    default="status",
    help="Which phase to run",
)
@click.option("--file", help="Fix specific file (mypy only)")
@click.option("--error-code", help="Show fix example for specific error code")
def main(phase: str, file: str | None, error_code: str | None):
    """Systematic project fixer for CodeDocSync."""
    fixer = ProjectFixer()

    if phase == "status":
        fixer.show_status()
        if error_code:
            fixer.show_fix_examples(error_code)
    elif phase == "examples":
        fixer.show_fix_examples(error_code)
    elif phase == "all":
        fixer.fix_ruff_auto()
        fixer.format_black()
        fixer.show_status()
    elif phase == "ruff":
        fixer.fix_ruff_auto()
        fixer.format_black()
        fixer.show_status()
    elif phase == "mypy" and file:
        # TODO: Implement specific file fixing for mypy
        console.print(
            f"[yellow]Mypy file-specific fixing not yet implemented for {file}[/yellow]"
        )
        console.print(
            "[dim]This will be implemented as part of the systematic mypy fix strategy[/dim]"
        )
    else:
        console.print("[bold]CodeDocSync Project Fixer[/bold]")
        console.print("\nUsage:")
        console.print("  python fix_project.py --phase status  # Show current status")
        console.print("  python fix_project.py --phase all     # Run all auto-fixes")
        console.print("  python fix_project.py --phase examples # Show fix examples")
        console.print(
            "  python fix_project.py --phase examples --error-code B904  # Show specific fix"
        )


if __name__ == "__main__":
    main()
