#!/usr/bin/env python3
"""
Find which test file is causing memory issues during import.
This script renames test files one by one and checks if tests can be collected.
"""

import subprocess
import sys
from pathlib import Path


def find_test_files(test_dir):
    """Find all Python test files in the directory."""
    test_files = []
    for file in Path(test_dir).rglob("test_*.py"):
        if file.is_file():
            test_files.append(file)
    return sorted(test_files)


def can_collect_tests():
    """Try to collect tests without running them."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/suggestions"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout during collection"
    except Exception as e:
        return False, "", str(e)


def main():
    test_dir = Path("tests/suggestions")
    test_files = find_test_files(test_dir)

    print(f"Found {len(test_files)} test files to check")
    print("=" * 60)

    # First, check if collection works with all files
    print("Testing collection with all files...")
    success, stdout, stderr = can_collect_tests()
    if success:
        print("[OK] Collection works with all files! No memory issue found.")
        return
    else:
        print("[FAIL] Collection failed. Starting binary search...")
        if stderr:
            print(f"Error: {stderr}")

    # Binary search approach - disable half the files at a time
    problematic_files = []
    remaining_files = test_files.copy()

    while remaining_files:
        # Try disabling each file one by one
        for test_file in remaining_files[:]:
            # Rename to .bak
            backup_path = test_file.with_suffix(".py.bak")
            print(f"\nTesting without: {test_file.name}")

            try:
                test_file.rename(backup_path)

                # Test collection
                success, stdout, stderr = can_collect_tests()

                if success:
                    print(f"[OK] Collection works without {test_file.name}")
                    print(f"  FOUND PROBLEMATIC FILE: {test_file.name}")
                    problematic_files.append(test_file.name)
                    remaining_files.remove(test_file)
                    # Keep it disabled and continue searching for others
                else:
                    print(f"[FAIL] Still fails without {test_file.name}")
                    # Restore the file
                    backup_path.rename(test_file)

            except Exception as e:
                print(f"Error handling {test_file}: {e}")
                # Try to restore if possible
                if backup_path.exists():
                    backup_path.rename(test_file)

    print("\n" + "=" * 60)
    print("RESULTS:")
    if problematic_files:
        print(f"Found {len(problematic_files)} problematic files:")
        for f in problematic_files:
            print(f"  - {f}")
        print("\nThese files are currently renamed to .bak")
        print("You can safely delete them or investigate further.")
    else:
        print("Could not isolate the problem to specific files.")
        print("The issue might be in the interaction between multiple files.")

    # Also check for large fixture files
    print("\nChecking for suspiciously large files...")
    for test_file in test_dir.rglob("*.py"):
        if test_file.is_file():
            size_mb = test_file.stat().st_size / (1024 * 1024)
            if size_mb > 1:
                print(f"  - {test_file.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
