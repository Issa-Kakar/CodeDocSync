"""Fix missing typing imports in the codebase."""

import re
from pathlib import Path


def fix_missing_imports(file_path: Path) -> bool:
    """Fix missing typing imports in a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Track what's needed
        needs_optional = False
        needs_list = False
        needs_dict = False
        needs_set = False
        needs_tuple = False
        needs_union = False
        needs_any = False

        # Check what's used but not imported
        if (
            re.search(r"\bOptional\[", content)
            and "from typing import" in content
            and "Optional" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_optional = True
        if (
            re.search(r"\bList\[", content)
            and "from typing import" in content
            and "List" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_list = True
        if (
            re.search(r"\bDict\[", content)
            and "from typing import" in content
            and "Dict" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_dict = True
        if (
            re.search(r"\bSet\[", content)
            and "from typing import" in content
            and "Set" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_set = True
        if (
            re.search(r"\bTuple\[", content)
            and "from typing import" in content
            and "Tuple" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_tuple = True
        if (
            re.search(r"\bUnion\[", content)
            and "from typing import" in content
            and "Union" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_union = True
        if (
            re.search(r"\bAny\b", content)
            and "from typing import" in content
            and "Any" not in content.split("from typing import")[1].split("\n")[0]
        ):
            needs_any = True

        # Add missing imports
        if any(
            [
                needs_optional,
                needs_list,
                needs_dict,
                needs_set,
                needs_tuple,
                needs_union,
                needs_any,
            ]
        ):
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("from typing import"):
                    imports = []
                    # Parse existing imports
                    import_match = re.match(r"from typing import (.+)", line)
                    if import_match:
                        existing = [
                            imp.strip() for imp in import_match.group(1).split(",")
                        ]
                        imports.extend(existing)

                    # Add missing imports
                    if needs_any and "Any" not in imports:
                        imports.append("Any")
                    if needs_dict and "Dict" not in imports:
                        imports.append("Dict")
                    if needs_list and "List" not in imports:
                        imports.append("List")
                    if needs_optional and "Optional" not in imports:
                        imports.append("Optional")
                    if needs_set and "Set" not in imports:
                        imports.append("Set")
                    if needs_tuple and "Tuple" not in imports:
                        imports.append("Tuple")
                    if needs_union and "Union" not in imports:
                        imports.append("Union")

                    # Sort imports
                    imports = sorted(set(imports))

                    # Update the line
                    lines[i] = f"from typing import {', '.join(imports)}"
                    break

            content = "\n".join(lines)

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            print(f"[FIXED] Missing imports in {file_path}")
            return True
        return False

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return False


# Fix specific files we know have issues
files_to_fix = [
    "codedocsync/parser/docstring_models.py",
    "codedocsync/analyzer/config.py",
    "codedocsync/suggestions/merging.py",
    "codedocsync/suggestions/config_manager.py",
]

print("Fixing missing typing imports...\n")
fixed_count = 0
for file_path in files_to_fix:
    path = Path(file_path)
    if path.exists():
        if fix_missing_imports(path):
            fixed_count += 1
    else:
        print(f"[WARNING] File not found: {file_path}")

print(f"\nFixed {fixed_count} files with missing imports")
