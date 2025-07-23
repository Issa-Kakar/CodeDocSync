#!/usr/bin/env python3
"""Fix all syntax errors in test files where imports are mixed."""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """Fix a single file with mixed imports."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Find multi-line imports with embedded imports
        # Pattern: from X import (\nfrom Y import Z\n    items...\n)
        pattern = r"(from\s+[\w\.]+\s+import\s+\(\s*\n)((?:from\s+[\w\.]+\s+import\s+[\w\s,]+\n)+)((?:\s+[\w,\s]+\n)*\))"

        matches = list(re.finditer(pattern, content, re.MULTILINE))

        if matches:
            # Process from end to start to maintain positions
            for match in reversed(matches):
                start_import = match.group(1)
                embedded_imports = match.group(2)
                rest_of_import = match.group(3)

                # Extract individual embedded imports
                embedded_lines = [
                    line.strip()
                    for line in embedded_imports.strip().split("\n")
                    if line.strip()
                ]

                # Reconstruct: embedded imports first, then the original import
                new_section = (
                    "\n".join(embedded_lines) + "\n\n" + start_import + rest_of_import
                )

                # Replace in content
                content = (
                    content[: match.start()] + new_section + content[match.end() :]
                )

            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False


def main():
    """Fix all Python files in tests directory."""
    test_dir = Path("tests")
    fixed_count = 0

    for py_file in test_dir.rglob("*.py"):
        if fix_file(py_file):
            print(f"Fixed: {py_file}")
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
