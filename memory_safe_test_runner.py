#!/usr/bin/env python
"""
Memory-safe test runner that prevents system crashes during pytest runs.

This script runs tests in isolated subprocesses with memory monitoring and limits,
preventing runaway memory usage that can crash the system.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MemorySafeTestRunner:
    """Run pytest tests with memory safety controls."""
    
    def __init__(
        self,
        memory_limit_mb: int = 1024,  # 1GB default
        timeout_seconds: int = 300,    # 5 minutes default
        cleanup_between_tests: bool = True
    ):
        self.memory_limit_mb = memory_limit_mb
        self.timeout_seconds = timeout_seconds
        self.cleanup_between_tests = cleanup_between_tests
        self.python_executable = sys.executable
        self.results: List[Dict] = []
        
    def cleanup_test_artifacts(self) -> None:
        """Clean up database files and test artifacts."""
        print("[CLEAN] Cleaning up test artifacts...")
        
        # Patterns to clean up
        cleanup_patterns = [
            ".codedocsync_cache",
            ".test_cache",
            "*.db",
            "*.sqlite",
            "*.sqlite3",
            ".chroma",
            "__pycache__",
            ".pytest_cache",
            "chroma_*",
            "embeddings.db",
        ]
        
        # Clean up in current directory and tests directory
        for base_path in [Path.cwd(), Path("tests")]:
            if not base_path.exists():
                continue
                
            for pattern in cleanup_patterns:
                # Handle directory patterns
                if not pattern.startswith("*"):
                    for path in base_path.rglob(pattern):
                        if path.is_dir():
                            try:
                                shutil.rmtree(path)
                                print(f"  [OK] Removed directory: {path}")
                            except Exception as e:
                                print(f"  [WARN] Failed to remove {path}: {e}")
                        elif path.is_file():
                            try:
                                path.unlink()
                                print(f"  [OK] Removed file: {path}")
                            except Exception as e:
                                print(f"  [WARN] Failed to remove {path}: {e}")
                else:
                    # Handle file patterns
                    for path in base_path.rglob(pattern):
                        if path.is_file():
                            try:
                                path.unlink()
                                print(f"  [OK] Removed file: {path}")
                            except Exception as e:
                                print(f"  [WARN] Failed to remove {path}: {e}")
    
    def monitor_process_memory(self, process: psutil.Process) -> Tuple[bool, float]:
        """
        Monitor a process's memory usage.
        
        Returns:
            Tuple of (exceeded_limit, peak_memory_mb)
        """
        peak_memory_mb = 0.0
        
        try:
            while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                peak_memory_mb = max(peak_memory_mb, memory_mb)
                
                if memory_mb > self.memory_limit_mb:
                    # Kill the process if it exceeds memory limit
                    print(f"\n[MEMORY] Memory limit exceeded: {memory_mb:.1f}MB > {self.memory_limit_mb}MB")
                    process.terminate()
                    time.sleep(1)
                    if process.is_running():
                        process.kill()
                    return True, peak_memory_mb
                
                time.sleep(0.1)  # Check every 100ms
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
        return False, peak_memory_mb
    
    def run_test_file(self, test_file: Path) -> Dict:
        """Run a single test file in a subprocess with monitoring."""
        print(f"\n[RUN] Running: {test_file}")
        print(f"   Memory limit: {self.memory_limit_mb}MB, Timeout: {self.timeout_seconds}s")
        
        start_time = time.time()
        result = {
            "file": str(test_file),
            "status": "unknown",
            "duration": 0.0,
            "peak_memory_mb": 0.0,
            "output": "",
            "error": ""
        }
        
        # Clean up before test if requested
        if self.cleanup_between_tests:
            self.cleanup_test_artifacts()
        
        # Run pytest in subprocess
        cmd = [
            self.python_executable,
            "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--no-header",
            "--no-summary",
            "-q"
        ]
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # Convert to psutil Process for monitoring
            ps_process = psutil.Process(process.pid)
            
            # Monitor memory in a separate thread-like approach
            memory_exceeded = False
            peak_memory = 0.0
            
            # Use communicate with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                memory_info = ps_process.memory_info()
                peak_memory = memory_info.rss / (1024 * 1024)
                
                # Check if process completed successfully
                if process.returncode == 0:
                    result["status"] = "passed"
                    print(f"   [PASS] PASSED")
                else:
                    result["status"] = "failed"
                    print(f"   [FAIL] FAILED")
                    
                result["output"] = stdout
                result["error"] = stderr
                
            except subprocess.TimeoutExpired:
                # Kill the process on timeout
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                    
                result["status"] = "timeout"
                print(f"   [TIMEOUT] TIMEOUT after {self.timeout_seconds}s")
                
                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    result["output"] = stdout if stdout else ""
                    result["error"] = stderr if stderr else ""
                except:
                    pass
            
            # Check memory usage periodically during execution
            check_interval = 0.5  # seconds
            total_checks = int(self.timeout_seconds / check_interval)
            
            for _ in range(total_checks):
                if process.poll() is not None:  # Process finished
                    break
                    
                try:
                    memory_mb = ps_process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, memory_mb)
                    
                    if memory_mb > self.memory_limit_mb:
                        process.terminate()
                        time.sleep(1)
                        if process.poll() is None:
                            process.kill()
                        
                        result["status"] = "memory_exceeded"
                        memory_exceeded = True
                        print(f"   [MEMORY] MEMORY EXCEEDED: {memory_mb:.1f}MB > {self.memory_limit_mb}MB")
                        break
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
                time.sleep(check_interval)
            
            result["peak_memory_mb"] = peak_memory
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   [WARN] ERROR: {e}")
        
        result["duration"] = time.time() - start_time
        print(f"   Duration: {result['duration']:.2f}s, Peak memory: {result['peak_memory_mb']:.1f}MB")
        
        return result
    
    def find_test_files(self, pattern: Optional[str] = None, specific_file: Optional[str] = None) -> List[Path]:
        """Find test files to run."""
        if specific_file:
            path = Path(specific_file)
            if path.exists():
                return [path]
            else:
                print(f"[WARN] File not found: {specific_file}")
                return []
        
        # Find all test files
        test_files = []
        test_dir = Path("tests")
        
        if test_dir.exists():
            for test_file in test_dir.rglob("test_*.py"):
                if pattern:
                    if pattern.lower() in str(test_file).lower():
                        test_files.append(test_file)
                else:
                    test_files.append(test_file)
        
        return sorted(test_files)
    
    def run_all_tests(self, pattern: Optional[str] = None, specific_file: Optional[str] = None) -> None:
        """Run all tests with memory safety."""
        print("[START] Memory-Safe Test Runner")
        print(f"   Memory limit: {self.memory_limit_mb}MB")
        print(f"   Timeout: {self.timeout_seconds}s")
        print(f"   Cleanup between tests: {self.cleanup_between_tests}")
        
        # Initial cleanup
        self.cleanup_test_artifacts()
        
        # Find test files
        test_files = self.find_test_files(pattern, specific_file)
        
        if not test_files:
            print("[WARN] No test files found!")
            return
        
        print(f"\n[INFO] Found {len(test_files)} test files to run")
        
        # Run each test file
        passed = 0
        failed = 0
        memory_exceeded = 0
        timed_out = 0
        errors = 0
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\n[{i}/{len(test_files)}] ", end="")
            result = self.run_test_file(test_file)
            self.results.append(result)
            
            # Update counters
            if result["status"] == "passed":
                passed += 1
            elif result["status"] == "failed":
                failed += 1
                # Show error output for failed tests
                if result["error"]:
                    print(f"   Error output:\n{result['error'][:500]}...")
            elif result["status"] == "memory_exceeded":
                memory_exceeded += 1
            elif result["status"] == "timeout":
                timed_out += 1
            else:
                errors += 1
        
        # Final cleanup
        self.cleanup_test_artifacts()
        
        # Print summary
        print("\n" + "=" * 60)
        print("[SUMMARY] Test Run Summary")
        print("=" * 60)
        print(f"Total test files: {len(test_files)}")
        print(f"[PASS] Passed: {passed}")
        print(f"[FAIL] Failed: {failed}")
        print(f"[MEMORY] Memory exceeded: {memory_exceeded}")
        print(f"[TIMEOUT] Timed out: {timed_out}")
        print(f"[WARN] Errors: {errors}")
        
        # Memory statistics
        memory_values = [r["peak_memory_mb"] for r in self.results if r["peak_memory_mb"] > 0]
        if memory_values:
            print(f"\n[MEMORY STATS] Memory Usage Statistics:")
            print(f"   Average: {sum(memory_values) / len(memory_values):.1f}MB")
            print(f"   Maximum: {max(memory_values):.1f}MB")
            print(f"   Minimum: {min(memory_values):.1f}MB")
        
        # Save results
        results_file = "memory_safe_test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total": len(test_files),
                    "passed": passed,
                    "failed": failed,
                    "memory_exceeded": memory_exceeded,
                    "timed_out": timed_out,
                    "errors": errors
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n[SAVED] Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if failed + memory_exceeded + timed_out + errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pytest tests with memory safety controls"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=1024,
        help="Memory limit in MB (default: 1024MB = 1GB)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test file in seconds (default: 300s = 5 minutes)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable cleanup between tests"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Only run test files matching this pattern"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Run a specific test file"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean up test artifacts and exit"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = MemorySafeTestRunner(
        memory_limit_mb=args.memory_limit,
        timeout_seconds=args.timeout,
        cleanup_between_tests=not args.no_cleanup
    )
    
    # Handle clean-only mode
    if args.clean_only:
        runner.cleanup_test_artifacts()
        print("[OK] Cleanup complete!")
        return
    
    # Run tests
    runner.run_all_tests(pattern=args.pattern, specific_file=args.file)


if __name__ == "__main__":
    main()