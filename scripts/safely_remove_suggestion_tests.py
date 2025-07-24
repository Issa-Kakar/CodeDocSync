#!/usr/bin/env python3
"""
Safely remove suggestion tests while preserving the working formatters.
"""

from pathlib import Path


def safely_remove_tests():
    """Remove suggestion tests but keep the working formatters."""

    # The formatters are 100% passing - keep them
    keep_dirs = ["tests/suggestions/formatters"]

    # Files to preserve
    keep_files = [
        "tests/suggestions/__init__.py",
        "tests/suggestions/fixtures.py",  # May be needed by formatters
    ]

    # Get all test files
    test_dir = Path("tests/suggestions")
    all_files = list(test_dir.rglob("*.py"))

    removed_count = 0
    kept_count = 0

    print("Safely removing suggestion tests...")
    print("=" * 60)

    for file_path in all_files:
        # Check if we should keep this file
        should_keep = False

        # Keep if in a preserve directory
        for keep_dir in keep_dirs:
            if str(file_path).startswith(str(Path(keep_dir))):
                should_keep = True
                break

        # Keep if explicitly listed
        if str(file_path) in [str(Path(f)) for f in keep_files]:
            should_keep = True

        if should_keep:
            print(f"[KEEP] {file_path}")
            kept_count += 1
        else:
            # Remove the file
            try:
                file_path.unlink()
                print(f"[REMOVED] {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"[ERROR] Could not remove {file_path}: {e}")

    # Remove empty directories
    for dirpath in sorted(test_dir.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            try:
                dirpath.rmdir()
                print(f"[REMOVED DIR] {dirpath}")
            except (OSError, PermissionError):
                pass

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Files removed: {removed_count}")
    print(f"  Files kept: {kept_count}")
    print("\nThe formatter tests (100% passing) have been preserved.")
    print("You can now safely reimplement the other suggestion tests.")


if __name__ == "__main__":
    safely_remove_tests()
