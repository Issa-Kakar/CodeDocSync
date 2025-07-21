"""Verify that style_detector.py can be type-checked successfully."""

import subprocess
import sys

print("Checking style_detector.py with mypy...")
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "mypy",
        "codedocsync/suggestions/style_detector.py",
        "--show-error-codes",
    ],
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print("[PASS] style_detector.py passes mypy type checking!")
    print("All mypy errors have been successfully fixed.")
else:
    print("[FAIL] style_detector.py still has mypy errors:")
    # Filter to show only errors from style_detector.py
    lines = result.stdout.split("\n")
    for line in lines:
        if "style_detector.py:" in line:
            print(line)
