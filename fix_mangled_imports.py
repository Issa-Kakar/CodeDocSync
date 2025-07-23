#!/usr/bin/env python3
"""Fix all mangled imports in test files."""

from pathlib import Path

# List of files that were mangled by the previous fix
files_to_fix = [
    "tests/suggestions/test_integration.py",
    "tests/suggestions/test_models.py",
    "tests/suggestions/test_performance.py",
    "tests/suggestions/test_style_detector.py",
    "tests/suggestions/test_suggestion_generator.py",
    "tests/suggestions/test_validation.py",
]


def fix_mangled_imports(file_path: Path) -> bool:
    """Fix imports that got mangled."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Look for patterns where items are on their own lines between import statements
        # Pattern: lines that look like module names/items without 'from' or 'import'
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a "from X import (" line
            if line.strip().startswith("from ") and line.strip().endswith("("):
                fixed_lines.append(line)
                i += 1

                # Collect all the import items until we find the closing )
                import_items = []
                while i < len(lines) and ")" not in lines[i]:
                    item_line = lines[i].strip()
                    if item_line and not item_line.startswith("from "):
                        import_items.append(item_line.rstrip(","))
                    i += 1

                # Add the collected items with proper indentation
                for item in import_items:
                    fixed_lines.append(f"    {item},")

                # Add the closing parenthesis
                if i < len(lines):
                    fixed_lines.append(lines[i])
                    i += 1
            else:
                # Check if this looks like a standalone import item
                if (
                    line.strip()
                    and not line.strip().startswith(
                        ("from ", "import ", "#", '"""', "'''")
                    )
                    and line.strip().endswith(",")
                    and i > 0
                    and "from" in lines[i - 1]
                ):
                    # Skip this line - it will be handled by the previous block
                    i += 1
                    continue
                else:
                    fixed_lines.append(line)
                    i += 1

        fixed_content = "\n".join(fixed_lines)

        if fixed_content != original:
            file_path.write_text(fixed_content, encoding="utf-8")
            return True

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")

    return False


def main():
    """Fix all specified files."""
    fixed_count = 0

    for file_str in files_to_fix:
        file_path = Path(file_str)
        if file_path.exists():
            if fix_mangled_imports(file_path):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
