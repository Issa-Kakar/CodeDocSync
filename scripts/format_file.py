#!/usr/bin/env python
"""Auto-format Python files using black, ruff, and mypy."""

import subprocess
import sys
from pathlib import Path


def get_venv_python() -> str:
    """Get the path to the venv Python executable."""
    # Check if we're already in venv
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        return sys.executable

    # Otherwise, use the .venv Python
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists():
        return str(venv_python.absolute())

    # Fallback to current Python
    return sys.executable


def run_command(cmd: list[str], file_path: str) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        # Replace 'python' with venv Python
        if cmd[0] == "python":
            cmd[0] = get_venv_python()

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def format_file(file_path: str) -> int:
    """Format a Python file with black, ruff, and check with mypy."""
    path = Path(file_path)

    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return 1

    if not path.suffix == ".py":
        print(f"[SKIP] Non-Python file: {file_path}")
        return 0

    # Skip excluded files
    if "syntax_error.py" in str(path):
        print(f"[SKIP] Excluded file: {file_path}")
        return 0

    print(f"\n[FORMAT] Processing: {file_path}")

    # Track if any tool made changes
    any_changes = False
    any_errors = False

    # Run Black
    print("  - Running Black...", end="")
    code, out, err = run_command(["python", "-m", "black", file_path], file_path)
    if code == 0:
        if "reformatted" in out:
            print(" DONE (formatted)")
            any_changes = True
        else:
            print(" DONE (no changes)")
    else:
        print(f" ERROR: {err.strip()}")
        any_errors = True

    # Run Ruff with --fix
    print("  - Running Ruff...", end="")
    code, out, err = run_command(
        ["python", "-m", "ruff", "check", "--fix", file_path], file_path
    )
    if code == 0:
        if "Fixed" in out or out.strip():
            print(" DONE (fixed issues)")
            any_changes = True
        else:
            print(" DONE (no issues)")
    else:
        # Ruff returns non-zero if there are unfixable issues
        if "Fixed" in err or "Fixed" in out:
            print(" PARTIAL (fixed some issues, others remain)")
            any_changes = True
        else:
            print(" WARNING (unfixable issues)")

    # Run Mypy (report only, don't fail)
    print("  - Running Mypy...", end="")
    code, out, err = run_command(["python", "-m", "mypy", file_path], file_path)
    if code == 0:
        print(" DONE (no type errors)")
    else:
        # Count errors
        error_lines = [line for line in out.split("\n") if ": error:" in line]
        if error_lines:
            print(
                f" WARNING ({len(error_lines)} type error{'s' if len(error_lines) > 1 else ''})"
            )
            for line in error_lines[:3]:  # Show first 3 errors
                print(f"     -> {line.strip()}")
            if len(error_lines) > 3:
                print(f"     -> ... and {len(error_lines) - 3} more")
        else:
            print(" DONE")

    # Summary
    if any_changes:
        print("\n[SUCCESS] File formatted successfully!")
    elif not any_errors:
        print("\n[OK] File already well-formatted!")

    return 0  # Always return success so hooks don't block


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python format_file.py <file_path>")
        return 1

    file_path = sys.argv[1]
    return format_file(file_path)


if __name__ == "__main__":
    sys.exit(main())
