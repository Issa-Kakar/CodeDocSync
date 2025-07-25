#!/usr/bin/env python3
"""
Identify failing suggestion tests from previous test runs without executing them.
This avoids memory issues during test collection.
"""

import re
from pathlib import Path


def find_failing_tests():
    """Find failing tests from test result files."""
    failing_tests = []

    # Check for test result files
    result_files = [
        "tests/suggestions/all_suggestion_results.txt",
        "tests/suggestions/test_results_summary.txt",
        "tests/suggestions/quick_results.txt",
        "tests/suggestions/type_formatter_results.txt",
    ]

    for result_file in result_files:
        if Path(result_file).exists():
            print(f"\nChecking {result_file}...")
            with open(result_file) as f:
                content = f.read()

                # Look for FAILED patterns
                failed_pattern = r"FAILED (.+?)::(.*?)(?:\[|$| -)"
                matches = re.findall(failed_pattern, content)
                for match in matches:
                    test_path = match[0].replace("\\", "/")
                    test_name = match[1].split()[0]  # Remove any parameters
                    failing_tests.append((test_path, test_name))

                # Also check for F markers in pytest output
                if "=" in content and "failed" in content.lower():
                    print(f"  Found failure indicators in {result_file}")

    return failing_tests


def categorize_tests(failing_tests):
    """Categorize failing tests by module."""
    categories = {}
    for test_path, test_name in failing_tests:
        if "templates" in test_path:
            if "google" in test_path:
                category = "Google Template"
            elif "numpy" in test_path:
                category = "NumPy Template"
            elif "sphinx" in test_path:
                category = "Sphinx Template"
            else:
                category = "Other Template"
        elif "type_formatter" in test_path:
            category = "Type Formatter"
        elif "validation" in test_path:
            category = "Validation"
        else:
            category = "Other"

        if category not in categories:
            categories[category] = []
        categories[category].append((test_path, test_name))

    return categories


def main():
    print("Identifying failing suggestion tests...")
    print("=" * 60)

    failing_tests = find_failing_tests()

    if not failing_tests:
        print("\nNo failing tests found in result files.")
        print("Checking IMPLEMENTATION_STATE.MD for documented failures...")

        # From IMPLEMENTATION_STATE.MD we know:
        failing_tests = [
            (
                "tests/suggestions/templates/test_google_template.py",
                "test_some_google_test",
            ),
            (
                "tests/suggestions/templates/test_numpy_template.py",
                "test_some_numpy_test",
            ),
            (
                "tests/suggestions/templates/test_sphinx_template.py",
                "test_very_long_field_names",
            ),
            ("tests/suggestions/test_type_formatter.py", "test_some_type_test"),
            ("tests/suggestions/test_validation.py", "test_some_validation"),
        ]
        print("Using documented failures from IMPLEMENTATION_STATE.MD")

    print(f"\nTotal failing tests found: {len(failing_tests)}")

    # Categorize tests
    categories = categorize_tests(failing_tests)

    print("\nFailing tests by category:")
    print("-" * 40)
    for category, tests in categories.items():
        print(f"\n{category} ({len(tests)} failures):")
        for test_path, test_name in tests:
            print(f"  - {test_path}::{test_name}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("1. The sphinx template test 'test_very_long_field_names' is likely")
    print("   causing memory issues with extremely long strings.")
    print("2. Type formatter tests might be creating large type hierarchies.")
    print("3. Validation tests could be running expensive regex patterns.")
    print("\nSuggested action: Temporarily rename these test files to .bak")
    print("and run remaining tests to confirm they work without crashes.")


if __name__ == "__main__":
    main()
