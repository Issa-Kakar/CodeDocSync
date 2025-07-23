#!/usr/bin/env python3
"""Collect all mypy errors from test files."""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_mypy_on_file(file_path: Path) -> list[str]:
    """Run mypy on a single file and return errors."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", str(file_path), "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return result.stdout.strip().split("\n") if result.stdout else []
    except Exception as e:
        print(f"Error running mypy on {file_path}: {e}")
    return []


def main():
    """Collect all mypy errors from test files."""
    test_dir = Path("tests")
    all_errors = []
    error_counts = defaultdict(int)
    file_error_counts = defaultdict(int)

    # Get all Python files
    py_files = list(test_dir.rglob("*.py"))
    total_files = len(py_files)

    print(f"Running mypy on {total_files} test files...")

    for i, py_file in enumerate(py_files):
        print(f"\rProgress: {i + 1}/{total_files}", end="", flush=True)
        errors = run_mypy_on_file(py_file)

        for error in errors:
            if error and "error:" in error:
                all_errors.append(error)

                # Count by error type
                if "[" in error and "]" in error:
                    error_type = error.split("[")[-1].split("]")[0]
                    error_counts[error_type] += 1

                # Count by file
                file_error_counts[str(py_file)] += 1

    print(f"\n\nTotal errors found: {len(all_errors)}")

    # Write full report
    with open("full_mypy_report.txt", "w", encoding="utf-8") as f:
        f.write("=== MyPy Error Report for Test Files ===\n")
        f.write(f"Total errors: {len(all_errors)}\n")
        f.write(f"Files with errors: {len(file_error_counts)}\n\n")

        f.write("=== All Errors ===\n")
        for error in all_errors:
            f.write(f"{error}\n")

        f.write("\n=== Error Type Summary ===\n")
        for error_type, count in sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"{error_type}: {count} errors\n")

        f.write("\n=== Files with Most Errors ===\n")
        for file_path, count in sorted(
            file_error_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            f.write(f"{file_path}: {count} errors\n")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total errors: {len(all_errors)}")
    print("\nTop 5 error types:")
    for error_type, count in sorted(
        error_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {error_type}: {count} errors")

    print("\nTop 5 files with errors:")
    for file_path, count in sorted(
        file_error_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {file_path}: {count} errors")


if __name__ == "__main__":
    main()
