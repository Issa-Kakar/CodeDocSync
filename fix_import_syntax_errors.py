#!/usr/bin/env python3
"""Fix syntax errors in test files where imports are mixed into other imports."""

import re
from pathlib import Path


def fix_mixed_imports(file_path: Path) -> bool:
    """Fix files where imports are incorrectly mixed into multi-line imports."""
    content = file_path.read_text(encoding="utf-8")

    # Pattern to find cases where an import statement is inside a multi-line import
    # Looking for patterns like:
    # from X import (
    # from Y import Z
    #     A,
    #     B,
    # )
    pattern = (
        r"(from\s+[\w\.]+\s+import\s+\(\s*\n)(from\s+[\w\.]+\s+import\s+[\w\s,]+\n)"
    )

    if re.search(pattern, content):
        print(f"Found mixed imports in {file_path}")

        # Extract the misplaced import
        match = re.search(pattern, content)
        if match:
            # Find the complete multi-line import block
            start_pos = match.start()
            paren_count = 1
            end_pos = start_pos + len(match.group(1))

            # Find the closing parenthesis
            for i in range(end_pos, len(content)):
                if content[i] == "(":
                    paren_count += 1
                elif content[i] == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos = i + 1
                        break

            # Extract the complete import block
            import_block = content[start_pos:end_pos]

            # Find all misplaced imports within the block
            misplaced_imports = []
            lines = import_block.split("\n")
            clean_lines = []

            for line in lines:
                if line.strip().startswith("from ") and "(" not in line:
                    misplaced_imports.append(line)
                else:
                    clean_lines.append(line)

            # Reconstruct the clean import
            clean_import = "\n".join(clean_lines)

            # Place misplaced imports before the multi-line import
            new_imports = "\n".join(misplaced_imports) + "\n\n" + clean_import

            # Replace in content
            new_content = content[:start_pos] + new_imports + content[end_pos:]

            file_path.write_text(new_content, encoding="utf-8")
            return True

    return False


def main():
    """Find and fix all test files with syntax errors."""
    test_dir = Path("tests")
    fixed_files = []

    for py_file in test_dir.rglob("*.py"):
        if fix_mixed_imports(py_file):
            fixed_files.append(py_file)

    print(f"\nFixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
