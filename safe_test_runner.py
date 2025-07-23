#!/usr/bin/env python
"""Safe test runner with resource limits and monitoring."""

import os
import subprocess
import sys
import time

import psutil


def monitor_process(
    proc: subprocess.Popen,
    max_memory_mb: int = 500,
    max_cpu_percent: int = 80,
    timeout: int = 30,
) -> bool:
    """Monitor a process and kill it if it exceeds resource limits."""
    start_time = time.time()
    process = psutil.Process(proc.pid)

    while proc.poll() is None:
        try:
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"TIMEOUT: Process exceeded {timeout}s limit")
                proc.terminate()
                return False

            # Check memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > max_memory_mb:
                print(
                    f"MEMORY LIMIT: Process using {memory_mb:.1f}MB (limit: {max_memory_mb}MB)"
                )
                proc.terminate()
                return False

            # Check CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            if cpu_percent > max_cpu_percent:
                print(
                    f"CPU LIMIT: Process using {cpu_percent:.1f}% CPU (limit: {max_cpu_percent}%)"
                )
                # Don't kill for CPU, just warn

            time.sleep(0.5)

        except psutil.NoSuchProcess:
            break

    return True


def run_single_test_file(test_file: str, venv_python: str) -> bool:
    """Run a single test file with resource monitoring."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {test_file}")
    print(f"{'=' * 60}")

    cmd = [venv_python, "-m", "pytest", test_file, "-v", "--tb=short", "-x"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="tests/suggestions",
        )

        # Monitor the process
        success = monitor_process(proc, max_memory_mb=500, timeout=30)

        if success:
            stdout, stderr = proc.communicate()
            print(stdout)
            if stderr:
                print("STDERR:", stderr)
        else:
            print("Process was terminated due to resource limits")

        return success

    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main() -> int:
    """Run tests safely one at a time."""
    # Ensure we're using the virtual environment Python
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        print("ERROR: Virtual environment not found!")
        sys.exit(1)

    # List of test files to check
    test_files = [
        "test_generators.py",
        "test_performance.py",
        "test_e2e_integration.py",
        "test_specific_issues.py",
        "test_validation.py",
        "test_suggestion_generator.py",
    ]

    problem_files = []

    for test_file in test_files:
        if not run_single_test_file(test_file, venv_python):
            problem_files.append(test_file)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if problem_files:
        print(f"Problem files: {', '.join(problem_files)}")
    else:
        print("All test files ran successfully within resource limits")

    return len(problem_files)


if __name__ == "__main__":
    sys.exit(main())
