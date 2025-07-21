#!/usr/bin/env python3
"""Test script to verify mypy fixes"""

import subprocess
import sys


def main():
    """Run mypy on the fixed files and report results."""
    files = [
        "codedocsync/suggestions/type_formatter.py",
        "codedocsync/suggestions/style_detector.py",
    ]

    print("Testing mypy fixes...")
    print("-" * 50)

    for file in files:
        print(f"\nChecking {file}:")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                file,
                "--show-error-codes",
                "--no-error-summary",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("[PASS] No mypy errors found!")
        else:
            print("[FAIL] Mypy errors:")
            print(result.stdout)
            if result.stderr:
                print("stderr:", result.stderr)

    print("\n" + "-" * 50)
    print("Test complete!")


if __name__ == "__main__":
    main()
