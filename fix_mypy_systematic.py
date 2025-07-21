#!/usr/bin/env python3
"""Systematic mypy fixer with specific patterns."""

import re
import subprocess
from pathlib import Path
from typing import Any


class MypySystematicFixer:
    def __init__(self) -> None:
        self.python_path = r"C:\Users\issak\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe"
        self.fixed_count = 0

    def fix_post_init_return_types(self, file_path: Path) -> int:
        """Fix missing return type annotations for __post_init__ methods."""
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Pattern for __post_init__ without return type
        pattern = r"(\s*)def __post_init__\(self([^)]*)\)(\s*):"
        replacement = r"\1def __post_init__(self\2) -> None\3:"

        content = re.sub(pattern, replacement, content)

        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return content.count("def __post_init__(self") - original.count(
                "def __post_init__(self"
            )
        return 0

    def fix_optional_parameters(self, file_path: Path) -> int:
        """Fix optional parameters missing | None annotation."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        # Pattern for parameters with None default but no Optional
        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Match: param: type = None
            match = re.match(r"(\s*)([\w_]+):\s*([^=\s]+)\s*=\s*None", line)
            if match and "|" not in match.group(3) and "Optional" not in line:
                indent, param, type_str = match.groups()
                lines[i] = f"{indent}{param}: {type_str} | None = None"
                fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_init_return_types(self, file_path: Path) -> int:
        """Fix missing return type annotations for __init__ methods."""
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Pattern for __init__ without return type
        pattern = r"(\s*)def __init__\(self([^)]*)\)(\s*):"
        replacement = r"\1def __init__(self\2) -> None\3:"

        content = re.sub(pattern, replacement, content)

        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return content.count("def __init__(self") - original.count(
                "def __init__(self"
            )
        return 0

    def fix_void_method_return_types(self, file_path: Path) -> int:
        """Fix missing return type annotations for void methods."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        # Pattern for methods without return type (excluding special methods)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Match: def method_name(...) without return type
            match = re.match(r"(\s*)def (\w+)\(([^)]*)\)(\s*):", line)
            if match and not match.group(2).startswith("__"):
                # Check if it doesn't already have a return type
                if "->" not in line:
                    indent, method_name, params, space = match.groups()

                    # Heuristic: methods that likely return None
                    void_patterns = [
                        "set_",
                        "update_",
                        "clear_",
                        "reset_",
                        "save_",
                        "load_",
                        "print_",
                        "show_",
                        "mark_",
                        "log_",
                        "write_",
                        "add_",
                        "remove_",
                        "delete_",
                        "register_",
                        "configure_",
                        "initialize_",
                        "cleanup_",
                        "close_",
                    ]

                    if any(method_name.startswith(p) for p in void_patterns):
                        lines[i] = (
                            f"{indent}def {method_name}({params}) -> None{space}:"
                        )
                        fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_any_return_types(self, file_path: Path) -> int:
        """Fix functions that should have specific return types instead of Any."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Look for common patterns that return specific types
            if "-> Any:" in line:
                # dict patterns
                if any(
                    p in line
                    for p in ["get_config", "to_dict", "as_dict", "get_settings"]
                ):
                    lines[i] = line.replace("-> Any:", "-> dict[str, Any]:")
                    fixes += 1
                # list patterns
                elif any(p in line for p in ["get_list", "to_list", "get_items"]):
                    lines[i] = line.replace("-> Any:", "-> list[Any]:")
                    fixes += 1
                # str patterns
                elif any(p in line for p in ["get_name", "get_text", "to_string"]):
                    lines[i] = line.replace("-> Any:", "-> str:")
                    fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_attribute_definitions(self, file_path: Path) -> int:
        """Fix common attribute name mistakes."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        # Common attribute replacements based on IMPLEMENTATION_STATE.MD
        replacements = [
            (r"\.type_annotation\b", ".type_str"),
            (r"\.enhanced_issues\b", ".issues"),
            (r"\.suggestion_object\b", ".rich_suggestion"),
            (r"documentation=", "docstring="),  # MatchedPair parameter fix
        ]

        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                fixes += len(re.findall(pattern, content))
                content = new_content

        if fixes > 0:
            file_path.write_text(content, encoding="utf-8")
        return fixes

    def fix_union_type_handling(self, file_path: Path) -> int:
        """Add isinstance checks for Union type handling."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for Union[RawDocstring, ParsedDocstring] access patterns
            if "docstring" in line and ("summary" in line or "description" in line):
                # Check if there's already an isinstance check
                if i > 0 and "isinstance" not in lines[i - 1]:
                    # Add isinstance check
                    indent = len(line) - len(line.lstrip())
                    indent_str = " " * indent

                    if ".summary" in line:
                        lines.insert(
                            i, f"{indent_str}if isinstance(docstring, ParsedDocstring):"
                        )
                        lines[i + 1] = f"    {lines[i + 1]}"
                        lines.insert(i + 2, f"{indent_str}else:")
                        lines.insert(
                            i + 3,
                            f"{indent_str}    summary = docstring.content if hasattr(docstring, 'content') else ''",
                        )
                        fixes += 1
                        i += 3
            i += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def get_mypy_errors_for_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Get mypy errors for a specific file."""
        cmd = [
            self.python_path,
            "-m",
            "mypy",
            str(file_path),
            "--no-error-summary",
            "--show-error-codes",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        errors = []
        for line in result.stdout.splitlines():
            if ": error:" in line:
                match = re.match(r".+?:(\d+):\s*error:\s*(.+?)\s*\[(.+?)\]", line)
                if match:
                    errors.append(
                        {
                            "line": int(match.group(1)),
                            "message": match.group(2),
                            "code": match.group(3),
                        }
                    )
        return errors

    def fix_file(self, file_path: Path) -> tuple[int, list[str]]:
        """Fix a single file and return (fixes_count, remaining_errors)."""
        if not file_path.exists():
            return 0, []

        fixes = 0

        # Apply automated fixes
        fixes += self.fix_post_init_return_types(file_path)
        fixes += self.fix_init_return_types(file_path)
        fixes += self.fix_optional_parameters(file_path)
        fixes += self.fix_void_method_return_types(file_path)
        fixes += self.fix_any_return_types(file_path)
        fixes += self.fix_attribute_definitions(file_path)
        # Temporarily disabled due to syntax issues
        # fixes += self.fix_union_type_handling(file_path)

        # Get remaining errors
        errors = self.get_mypy_errors_for_file(file_path)
        remaining = [f"Line {e['line']}: [{e['code']}] {e['message']}" for e in errors]

        return fixes, remaining

    def fix_module(self, module_path: str) -> dict[str, Any]:
        """Fix all files in a module."""
        module_dir = Path(module_path)
        results = {}
        total_fixes = 0

        for py_file in module_dir.glob("**/*.py"):
            if "__pycache__" not in str(py_file):
                fixes, remaining = self.fix_file(py_file)
                if fixes > 0 or remaining:
                    results[str(py_file)] = {
                        "fixes": fixes,
                        "remaining_errors": len(remaining),
                        "errors": remaining[:5],  # Show first 5 errors
                    }
                    total_fixes += fixes

        return {"total_fixes": total_fixes, "files": results}

    def get_status(self) -> dict[str, Any]:
        """Get current mypy error status."""
        cmd = [
            self.python_path,
            "-m",
            "mypy",
            "codedocsync",
            "--no-error-summary",
            "--show-error-codes",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        error_counts: dict[str, int] = {}
        file_errors: dict[str, int] = {}

        for line in result.stdout.splitlines():
            if ": error:" in line:
                match = re.match(r"(.+?):(\d+):\s*error:\s*(.+?)\s*\[(.+?)\]", line)
                if match:
                    file_path = match.group(1)
                    error_code = match.group(4)

                    # Count by error type
                    error_counts[error_code] = error_counts.get(error_code, 0) + 1

                    # Count by file
                    file_errors[file_path] = file_errors.get(file_path, 0) + 1

        # Sort by count
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_files = sorted(file_errors.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_errors": sum(error_counts.values()),
            "top_error_types": top_errors,
            "top_files": top_files,
        }


def main() -> None:
    """Main entry point."""
    import sys

    fixer = MypySystematicFixer()

    if len(sys.argv) < 2:
        print(
            "Usage: python fix_mypy_systematic.py [--status|--fix-file FILE|--fix-module MODULE]"
        )
        return

    command = sys.argv[1]

    if command == "--status":
        status = fixer.get_status()
        print("\n=== Mypy Status ===")
        print(f"Total errors: {status['total_errors']}")

        print("\n=== Top Error Types ===")
        for code, count in status["top_error_types"]:
            print(f"{code}: {count} errors")

        print("\n=== Top Files with Errors ===")
        for file, count in status["top_files"]:
            print(f"{file}: {count} errors")

    elif command == "--fix-file" and len(sys.argv) > 2:
        file_path = Path(sys.argv[2])
        print(f"\n=== Fixing {file_path} ===")

        fixes, remaining = fixer.fix_file(file_path)
        print(f"Applied {fixes} automated fixes")

        if remaining:
            print(f"\n{len(remaining)} errors remaining:")
            for error in remaining[:10]:  # Show first 10
                print(f"  {error}")
        else:
            print("No errors remaining!")

    elif command == "--fix-module" and len(sys.argv) > 2:
        module_path = sys.argv[2]
        print(f"\n=== Fixing module {module_path} ===")

        results = fixer.fix_module(module_path)
        print(f"Total fixes applied: {results['total_fixes']}")

        if results["files"]:
            print("\n=== Files Modified ===")
            for file, info in results["files"].items():
                print(f"\n{file}:")
                print(f"  Fixes applied: {info['fixes']}")
                print(f"  Errors remaining: {info['remaining_errors']}")
                if info["errors"]:
                    print("  Sample errors:")
                    for error in info["errors"]:
                        print(f"    {error}")

    else:
        print("Invalid command. Use --status, --fix-file FILE, or --fix-module MODULE")


if __name__ == "__main__":
    main()
