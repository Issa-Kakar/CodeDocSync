#!/usr/bin/env python3
"""Analyze test results from pytest output."""

import re
from collections import Counter


def analyze_test_results(filename):
    with open(filename) as f:
        content = f.read()

    # Find all test results
    passed = re.findall(r"(\S+::test_\S+) PASSED", content)
    failed = re.findall(r"(\S+::test_\S+) FAILED", content)
    errors = re.findall(r"(\S+::test_\S+) ERROR", content)

    # Extract module names
    def get_module(test_path):
        return test_path.split("::")[0].replace("tests/", "")

    passed_modules = Counter(get_module(t) for t in passed)
    failed_modules = Counter(get_module(t) for t in failed)
    error_modules = Counter(get_module(t) for t in errors)

    print("=== TEST RESULTS ANALYSIS ===")
    print(f"\nTotal: {len(passed) + len(failed) + len(errors)} tests")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")
    print(
        f"Pass Rate: {len(passed) / (len(passed) + len(failed) + len(errors)) * 100:.1f}%"
    )

    print("\n=== BY MODULE ===")
    all_modules = (
        set(passed_modules.keys())
        | set(failed_modules.keys())
        | set(error_modules.keys())
    )

    for module in sorted(all_modules):
        p = passed_modules.get(module, 0)
        f = failed_modules.get(module, 0)
        e = error_modules.get(module, 0)
        total = p + f + e
        if total > 0:
            print(f"\n{module}:")
            print(f"  Total: {total}, Passed: {p}, Failed: {f}, Errors: {e}")
            print(f"  Pass Rate: {p / total * 100:.1f}%")

    print("\n=== MODULES WITH ISSUES ===")
    for module in sorted(all_modules):
        f = failed_modules.get(module, 0)
        e = error_modules.get(module, 0)
        if f > 0 or e > 0:
            print(f"{module}: {f} failed, {e} errors")


if __name__ == "__main__":
    analyze_test_results("all_tests_baseline.txt")
