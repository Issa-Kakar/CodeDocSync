#!/usr/bin/env python
"""Run suggestion tests safely with proper output handling."""

import os
import subprocess
import sys
import time


def run_tests_safely() -> int:
    """Run suggestion tests with proper resource management."""
    # Ensure we're in the right directory
    os.chdir("tests/suggestions")

    # Use venv Python
    python_cmd = "../../.venv/Scripts/python.exe"

    print("Running suggestion tests safely...")
    print("=" * 60)

    # Run tests with limited output and stop on first failure
    cmd = [
        python_cmd,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--maxfail=10",  # Maximum 10 failures
        "--durations=10",  # Show slowest 10 tests
    ]

    start_time = time.time()

    try:
        # Run tests without redirecting to file (which might cause issues)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute timeout
        )

        # Print output in chunks to avoid overwhelming the terminal
        output = result.stdout + "\n" + result.stderr
        lines = output.split("\n")

        # Print first 100 lines
        print("\n".join(lines[:100]))

        if len(lines) > 100:
            print(f"\n... ({len(lines) - 100} more lines) ...\n")
            # Print last 50 lines for summary
            print("\n".join(lines[-50:]))

        # Save full output to file for reference
        with open("test_results_summary.txt", "w") as f:
            f.write(f"Test run at: {time.ctime()}\n")
            f.write(f"Duration: {time.time() - start_time:.2f} seconds\n")
            f.write("=" * 60 + "\n")
            f.write(output)

        print("\nFull results saved to: test_results_summary.txt")
        print(f"Test duration: {time.time() - start_time:.2f} seconds")

        return result.returncode

    except subprocess.TimeoutExpired:
        print("\nERROR: Tests timed out after 120 seconds!")
        print("This might indicate an infinite loop or hanging test.")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


def count_test_results() -> None:
    """Count passing and failing tests from the output file."""
    try:
        with open("test_results_summary.txt") as f:
            content = f.read()

        passed = content.count(" PASSED")
        failed = content.count(" FAILED")
        errors = content.count(" ERROR")

        print("\nTest Summary:")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Errors: {errors}")
        print(f"  Total:  {passed + failed + errors}")

    except FileNotFoundError:
        print("No test results file found.")


if __name__ == "__main__":
    exit_code = run_tests_safely()
    count_test_results()
    sys.exit(exit_code)
