#!/usr/bin/env python3
"""
Fix Tracker - Comprehensive tool for tracking and fixing linting and type issues in CodeDocSync

This tool combines progress tracking, automated fixing, and strategic planning for
systematically resolving ruff and mypy errors across the codebase.
"""

import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class FixTracker:
    """Tracks progress and provides utilities for fixing ruff and mypy issues systematically."""

    def __init__(self) -> None:
        self.progress_file = Path("fix_progress.json")
        self.python_path = r"C:\Users\issak\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe"
        self.load_progress()

    def load_progress(self) -> None:
        """Load progress from JSON file or initialize empty progress."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "ruff_fixed_files": [],
                "mypy_fixed_files": [],
                "current_phase": "ruff",
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "initial_ruff_errors": 0,
                    "initial_mypy_errors": 0,
                    "current_ruff_errors": 0,
                    "current_mypy_errors": 0,
                },
            }

    def save_progress(self) -> None:
        """Save current progress to JSON file."""
        self.progress["timestamp"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def run_command(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run command with proper Windows compatibility."""
        # Use direct Python path instead of poetry run
        if cmd[0] == "poetry" and cmd[1] == "run":
            tool = cmd[2]
            args = cmd[3:]
            if tool in ["ruff", "black", "mypy"]:
                cmd = [self.python_path, "-m", tool] + args
            else:
                cmd = [self.python_path, "-m"] + cmd[2:]

        # Set encoding to avoid Unicode issues
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return result.returncode, result.stdout, result.stderr

    def get_ruff_errors_by_file(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """Get ruff errors grouped by file, sorted by error count."""
        returncode, stdout, stderr = self.run_command(
            ["poetry", "run", "ruff", "check", ".", "--format", "json"]
        )

        if returncode == 0 and not stdout.strip():
            return []

        try:
            errors = json.loads(stdout) if stdout else []
        except json.JSONDecodeError:
            print(f"Error parsing ruff output: {stdout}")
            return []

        # Group by file
        by_file = defaultdict(list)
        for error in errors:
            file = error.get("filename", "unknown")
            by_file[file].append(error)

        # Sort by error count (fix files with fewer errors first)
        sorted_files = sorted(by_file.items(), key=lambda x: len(x[1]))

        # Update statistics
        total_errors = sum(len(errors) for _, errors in sorted_files)
        if self.progress["statistics"]["initial_ruff_errors"] == 0:
            self.progress["statistics"]["initial_ruff_errors"] = total_errors
        self.progress["statistics"]["current_ruff_errors"] = total_errors

        return sorted_files

    def get_ruff_errors_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Get ruff errors grouped by error code."""
        returncode, stdout, stderr = self.run_command(
            ["poetry", "run", "ruff", "check", ".", "--format", "json"]
        )

        if returncode == 0 and not stdout.strip():
            return {}

        try:
            errors = json.loads(stdout) if stdout else []
        except json.JSONDecodeError:
            return {}

        # Group by error code
        by_code = defaultdict(list)
        for error in errors:
            code = error.get("code", "unknown")
            by_code[code].append(error)

        return dict(sorted(by_code.items(), key=lambda x: len(x[1]), reverse=True))

    def mark_file_fixed(self, file_path: str, tool: str = "ruff") -> None:
        """Mark a file as fixed for the given tool."""
        if tool == "ruff":
            if file_path not in self.progress["ruff_fixed_files"]:
                self.progress["ruff_fixed_files"].append(file_path)
        elif tool == "mypy":
            if file_path not in self.progress["mypy_fixed_files"]:
                self.progress["mypy_fixed_files"].append(file_path)
        self.save_progress()

    def get_next_files_to_fix(self, count: int = 5) -> list[tuple[str, int]]:
        """Get the next files to fix based on error count."""
        files_with_errors = self.get_ruff_errors_by_file()
        unfixed_files = [
            (file, len(errors))
            for file, errors in files_with_errors
            if file not in self.progress["ruff_fixed_files"]
        ]
        return unfixed_files[:count]

    def fix_ruff_auto(self) -> None:
        """Auto-fix ruff issues using --fix flag."""
        print("\n=== Auto-fixing Ruff issues... ===")

        print("Running ruff --fix...")
        returncode, stdout, stderr = self.run_command(
            ["poetry", "run", "ruff", "check", ".", "--fix"]
        )

        if returncode == 0:
            print("[OK] Ruff auto-fixes applied successfully!")
        else:
            print("[WARNING] Ruff completed with warnings")
            if stderr:
                print(f"  {stderr}")

    def format_black(self) -> None:
        """Format code with Black."""
        print("\n=== Formatting with Black... ===")

        print("Running black...")
        returncode, stdout, stderr = self.run_command(["poetry", "run", "black", "."])

        if stdout:
            print(stdout)
        print("[OK] Black formatting complete!")

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

    def show_fix_examples(self, error_code: Optional[str] = None) -> None:
        """Show examples of how to fix common issues."""
        patterns = self.get_fix_patterns()

        if error_code and error_code in patterns:
            print(f"\n=== How to fix {error_code} ===")
            print(patterns[error_code])
        else:
            print("\n=== Common Fix Patterns ===")
            for code, pattern in patterns.items():
                print(f"\n--- {code} ---")
                print(pattern)

    def print_summary(self) -> None:
        """Print a summary of current progress."""
        ruff_by_file = self.get_ruff_errors_by_file()
        ruff_by_type = self.get_ruff_errors_by_type()

        print("\n=== Fix Progress Summary ===")
        print(f"Phase: {self.progress['current_phase']}")
        print(f"Last updated: {self.progress['timestamp']}")

        print("\n--- Ruff Statistics ---")
        print(f"Initial errors: {self.progress['statistics']['initial_ruff_errors']}")
        print(f"Current errors: {self.progress['statistics']['current_ruff_errors']}")
        print(f"Files fixed: {len(self.progress['ruff_fixed_files'])}")
        print(f"Files with errors: {len(ruff_by_file)}")

        print("\n--- Top Error Types ---")
        for code, errors in list(ruff_by_type.items())[:5]:
            print(f"{code}: {len(errors)} errors")

        print("\n--- Next Files to Fix ---")
        next_files = self.get_next_files_to_fix()
        for file, error_count in next_files:
            print(f"{file}: {error_count} errors")

    def get_errors_for_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all ruff errors for a specific file."""
        errors_by_file = dict(self.get_ruff_errors_by_file())
        return errors_by_file.get(file_path, [])

    def get_remaining_issues(self) -> tuple[int, int]:
        """Get current count of ruff and mypy issues."""
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


class MypyFixStrategy:
    """Strategic approach for fixing mypy errors."""

    def __init__(self, tracker: FixTracker):
        self.tracker = tracker

    def get_mypy_errors_by_file(self) -> dict[str, list[dict[str, Any]]]:
        """Get mypy errors grouped by file."""
        # Run mypy with explicit format
        cmd = [
            self.tracker.python_path,
            "-m",
            "mypy",
            "codedocsync",
            "--no-error-summary",
            "--show-error-codes",
        ]
        returncode, stdout, stderr = self.tracker.run_command(cmd)

        errors_by_file = defaultdict(list)
        for line in stdout.splitlines():
            if ": error:" in line and not line.startswith("Found "):
                # Parse format: file.py:line: error: message [error-code]
                match = re.match(r"(.+?):(\d+):\s*error:\s*(.+?)\s*\[(.+?)\]", line)
                if match:
                    file_path = match.group(1)
                    errors_by_file[file_path].append(
                        {
                            "file": file_path,
                            "line": int(match.group(2)),
                            "message": match.group(3),
                            "code": match.group(4),
                        }
                    )

        # Update statistics
        total_errors = sum(len(errors) for errors in errors_by_file.values())
        if self.tracker.progress["statistics"]["initial_mypy_errors"] == 0:
            self.tracker.progress["statistics"]["initial_mypy_errors"] = total_errors
        self.tracker.progress["statistics"]["current_mypy_errors"] = total_errors

        return dict(errors_by_file)

    def categorize_files(self) -> dict[str, list[tuple[str, int]]]:
        """Categorize files by priority and complexity."""
        errors = self.get_mypy_errors_by_file()

        # Categorize files by priority and complexity
        categories: dict[str, list[tuple[str, int]]] = {
            "critical_path": [],  # Main API files
            "generators": [],  # Suggestion generators
            "utilities": [],  # Helper files
            "tests": [],  # Test files (lowest priority)
        }

        for file, errs in errors.items():
            error_count = len(errs)

            # Skip already fixed files
            if file in self.tracker.progress["mypy_fixed_files"]:
                continue

            # Categorize based on path and error count
            if "cli/" in file or "__init__.py" in file:
                categories["critical_path"].append((file, error_count))
            elif "generators/" in file or "templates/" in file:
                categories["generators"].append((file, error_count))
            elif "test" in file:
                categories["tests"].append((file, error_count))
            else:
                categories["utilities"].append((file, error_count))

        # Sort each category by error count (ascending)
        for cat in categories:
            categories[cat].sort(key=lambda x: x[1])

        return categories

    def print_mypy_strategy(self) -> None:
        """Print the strategic approach for fixing mypy errors."""
        categories = self.categorize_files()
        errors_by_file = self.get_mypy_errors_by_file()

        print("\n=== Mypy Fix Strategy ===")
        print(
            f"Total mypy errors: {self.tracker.progress['statistics']['current_mypy_errors']}"
        )
        print(f"Files with errors: {len(errors_by_file)}")
        print(f"Files already fixed: {len(self.tracker.progress['mypy_fixed_files'])}")

        print("\n--- Files by Priority ---")
        for category, files in categories.items():
            print(f"\n{category.upper()} ({len(files)} files):")
            for file, error_count in files[:3]:  # Show top 3 in each category
                print(f"  {file}: {error_count} errors")

        # Get error type statistics
        error_types: dict[str, int] = defaultdict(int)
        for file_errors in errors_by_file.values():
            for error in file_errors:
                error_types[error.get("code", "unknown")] += 1

        print("\n--- Top Mypy Error Types ---")
        for code, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"{code}: {count} occurrences")


def main() -> None:
    """Main entry point for the fix tracker.

    Commands:
        status: Show comprehensive status of both ruff and mypy errors
        summary: Show fix progress summary (ruff focused)
        mypy: Show mypy-specific strategy
        ruff-fix: Run ruff auto-fixes and black formatting
        file <path>: Show errors for a specific file
        mark-fixed <path> [tool]: Mark a file as fixed
        examples [code]: Show fix examples for error codes
    """
    import sys

    tracker = FixTracker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            # Comprehensive status showing both ruff and mypy
            ruff_count, mypy_count = tracker.get_remaining_issues()

            print("\n=== CodeDocSync Fix Status ===")
            print("Issue Type    Count    Status")
            print("-" * 40)

            ruff_status = "[DONE]" if ruff_count == 0 else f"[!] {ruff_count} remaining"
            mypy_status = "[DONE]" if mypy_count == 0 else f"[!] {mypy_count} remaining"

            print(f"Ruff Issues   {ruff_count:<8} {ruff_status}")
            print(f"Mypy Issues   {mypy_count:<8} {mypy_status}")

            if ruff_count > 0:
                print("\n--- Ruff Breakdown ---")
                ruff_by_type = tracker.get_ruff_errors_by_type()
                for code, errors in list(ruff_by_type.items())[:5]:
                    print(f"{code}: {len(errors)} errors")

            if mypy_count > 0:
                print("\n--- Mypy Breakdown ---")
                strategy = MypyFixStrategy(tracker)
                errors_by_file = strategy.get_mypy_errors_by_file()
                error_types: dict[str, int] = defaultdict(int)
                for file_errors in errors_by_file.values():
                    for error in file_errors:
                        error_types[error.get("code", "unknown")] += 1

                for code, count in sorted(
                    error_types.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    print(f"{code}: {count} occurrences")

        elif command == "summary":
            tracker.print_summary()

        elif command == "mypy":
            strategy = MypyFixStrategy(tracker)
            strategy.print_mypy_strategy()

        elif command == "ruff-fix":
            tracker.fix_ruff_auto()
            tracker.format_black()
            tracker.print_summary()

        elif command == "file" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            errors = tracker.get_errors_for_file(file_path)
            print(f"\nErrors in {file_path}:")
            for error in errors:
                print(f"  Line {error['line']}: {error['code']} - {error['message']}")

        elif command == "mark-fixed" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            tool = sys.argv[3] if len(sys.argv) > 3 else "ruff"
            tracker.mark_file_fixed(file_path, tool)
            print(f"Marked {file_path} as fixed for {tool}")

        elif command == "examples":
            error_code = sys.argv[2] if len(sys.argv) > 2 else None
            tracker.show_fix_examples(error_code)

        else:
            print(
                "Usage: python fix_tracker.py [status|summary|mypy|ruff-fix|file <path>|mark-fixed <path> [tool]|examples [code]]"
            )
            print("\nCommands:")
            print("  status      - Show comprehensive status")
            print("  summary     - Show fix progress summary")
            print("  mypy        - Show mypy fix strategy")
            print("  ruff-fix    - Run auto-fixes")
            print("  file        - Show errors for a file")
            print("  mark-fixed  - Mark file as fixed")
            print("  examples    - Show fix examples")

    else:
        # Default: show comprehensive status
        tracker.print_summary()
        print("\nRun 'python fix_tracker.py status' for complete status")


if __name__ == "__main__":
    main()
