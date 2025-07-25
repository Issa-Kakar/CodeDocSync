#!/usr/bin/env python3
"""
Memory-safe test runner for CodeDocSync.

This runner prevents test-related memory leaks from crashing the system by:
1. Setting memory limits per test process
2. Running tests in isolated subprocesses
3. Cleaning up resources between test runs
4. Providing detailed memory usage reporting
"""

import argparse
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

# Platform-specific imports
if sys.platform != "win32":
    import resource


class MemorySafeTestRunner:
    """Run tests with memory safety constraints."""

    def __init__(self, memory_limit_mb: int = 2048):
        self.memory_limit_mb = memory_limit_mb
        self.temp_dirs = []
        self.process = None

    def set_memory_limit(self, limit_mb: int) -> None:
        """Set memory limit for current process (Linux/Mac only)."""
        if sys.platform != "win32":
            limit_bytes = limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    def clean_temp_files(self) -> None:
        """Clean up temporary files and directories."""
        patterns = [
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            ".coverage*",
            "htmlcov",
            ".mypy_cache",
            ".ruff_cache",
            "*.tmp",
            "test_*.txt",
        ]

        for pattern in patterns:
            for path in Path(".").rglob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except Exception as e:
                    print(f"Warning: Could not remove {path}: {e}")

        # Clean tracked temp directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        # Force garbage collection
        gc.collect()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def run_tests_in_subprocess(
        self, test_pattern: str = "", specific_test: str = ""
    ) -> int:
        """Run tests in an isolated subprocess with memory limits."""
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]

        if specific_test:
            cmd.append(specific_test)
        elif test_pattern:
            cmd.extend(["-k", test_pattern])
        else:
            cmd.append("tests/")

        # Add pytest options to reduce memory usage
        cmd.extend(
            [
                "-p",
                "no:cacheprovider",  # Disable cache
                "--maxfail=3",  # Stop after 3 failures
            ]
        )

        print(f"Running command: {' '.join(cmd)}")
        print(f"Memory limit: {self.memory_limit_mb}MB")

        # Create process
        if sys.platform == "win32":
            # Windows: Monitor memory usage externally
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            # Linux/Mac: Use resource limits
            def preexec_fn():
                self.set_memory_limit(self.memory_limit_mb)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=preexec_fn,
            )

        # Monitor process
        start_time = time.time()
        max_memory = 0
        output_lines = []

        try:
            while True:
                # Check if process finished
                retcode = self.process.poll()
                if retcode is not None:
                    break

                # Read output
                line = self.process.stdout.readline()
                if line:
                    print(line.rstrip())
                    output_lines.append(line)

                # Monitor memory
                try:
                    proc_info = psutil.Process(self.process.pid)
                    current_memory = proc_info.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, current_memory)

                    # Kill if approaching limit
                    if current_memory > self.memory_limit_mb * 0.9:
                        print(
                            f"\nWARNING: Process approaching memory limit ({current_memory:.0f}MB)"
                        )
                        self.process.terminate()
                        time.sleep(2)
                        if self.process.poll() is None:
                            self.process.kill()
                        retcode = -1
                        break
                except psutil.NoSuchProcess:
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.process.terminate()
            retcode = -2

        # Get remaining output
        remaining_output = self.process.stdout.read()
        if remaining_output:
            print(remaining_output)
            output_lines.append(remaining_output)

        elapsed_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Test run completed in {elapsed_time:.1f}s")
        print(f"Peak memory usage: {max_memory:.0f}MB")
        print(f"Return code: {retcode}")

        return retcode

    def run_safe_tests(self, test_pattern: str = "", specific_test: str = "") -> int:
        """Run tests with safety measures."""
        print("Starting memory-safe test run...")
        print(f"Initial memory usage: {self.get_memory_usage():.0f}MB")

        # Clean before running
        print("\nCleaning temporary files...")
        self.clean_temp_files()

        # Run tests
        retcode = self.run_tests_in_subprocess(test_pattern, specific_test)

        # Clean after running
        print("\nCleaning up after tests...")
        self.clean_temp_files()

        print(f"Final memory usage: {self.get_memory_usage():.0f}MB")

        return retcode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CodeDocSync tests with memory safety constraints"
    )
    parser.add_argument(
        "--pattern", "-k", help="Test pattern to match (pytest -k syntax)", default=""
    )
    parser.add_argument(
        "--test", "-t", help="Specific test file or test to run", default=""
    )
    parser.add_argument(
        "--memory-limit",
        "-m",
        type=int,
        default=2048,
        help="Memory limit in MB (default: 2048)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean temporary files without running tests",
    )

    args = parser.parse_args()

    runner = MemorySafeTestRunner(memory_limit_mb=args.memory_limit)

    if args.clean_only:
        print("Cleaning temporary files only...")
        runner.clean_temp_files()
        print("Cleanup complete!")
        return 0

    return runner.run_safe_tests(test_pattern=args.pattern, specific_test=args.test)


if __name__ == "__main__":
    sys.exit(main())
