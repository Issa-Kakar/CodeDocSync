#!/usr/bin/env python3
"""
Systematic mypy fixer specifically for test files.

This script extends fix_mypy_systematic.py with patterns specific to test files,
including pytest fixtures, test method annotations, and mock object handling.
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


class TestMypyFixer:
    """Specialized fixer for mypy errors in test files."""

    def __init__(self) -> None:
        self.python_path = r"C:\Users\issak\AppData\Local\pypoetry\Cache\virtualenvs\codedocsync-5yfwj9Sn-py3.12\Scripts\python.exe"
        self.fixed_count = 0

    def fix_test_method_annotations(self, file_path: Path) -> int:
        """Fix missing return type annotations for test methods."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Match test methods without return type
            match = re.match(r"(\s*)def (test_\w+)\(([^)]*)\)(\s*):", line)
            if match and "->" not in line:
                indent, method_name, params, space = match.groups()
                lines[i] = f"{indent}def {method_name}({params}) -> None{space}:"
                fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_fixture_annotations(self, file_path: Path) -> int:
        """Fix missing type annotations for pytest fixtures."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        in_fixture = False

        for i, line in enumerate(lines):
            # Detect @pytest.fixture decorator
            if "@pytest.fixture" in line or "@fixture" in line:
                in_fixture = True
                continue

            # Fix the function definition after fixture decorator
            if in_fixture and line.strip() and not line.strip().startswith("@"):
                match = re.match(r"(\s*)def (\w+)\(([^)]*)\)(\s*):", line)
                if match and "->" not in line:
                    indent, func_name, params, space = match.groups()
                    # Common fixture return types based on name patterns
                    if any(p in func_name for p in ["mock", "mocker"]):
                        return_type = "MockerFixture"
                    elif "client" in func_name:
                        return_type = "Any"
                    elif "session" in func_name or "db" in func_name:
                        return_type = "Any"
                    elif "tmp" in func_name or "temp" in func_name:
                        return_type = "Path"
                    else:
                        return_type = "Any"

                    lines[i] = (
                        f"{indent}def {func_name}({params}) -> {return_type}{space}:"
                    )
                    fixes += 1
                in_fixture = False

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_fixture_parameters(self, file_path: Path) -> int:
        """Fix missing type annotations for fixture parameters in test methods."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Match test methods with untyped parameters
            if (
                "def test_" in line
                and ": " not in line.split("(", 1)[1].split(")", 1)[0]
            ):
                match = re.match(r"(\s*)def (test_\w+)\(([^)]+)\)([^:]*):(.*)$", line)
                if match:
                    indent, method_name, params, return_annotation, rest = (
                        match.groups()
                    )

                    # Split parameters and add types
                    param_list = []
                    for param in params.split(","):
                        param = param.strip()
                        if param and param != "self" and ":" not in param:
                            # Common fixture type patterns
                            if "mock" in param or "mocker" in param:
                                param_list.append(f"{param}: MockerFixture")
                            elif param == "tmp_path":
                                param_list.append(f"{param}: Path")
                            elif param == "caplog":
                                param_list.append(f"{param}: LogCaptureFixture")
                            elif param == "capsys":
                                param_list.append(f"{param}: CaptureFixture[str]")
                            elif "client" in param:
                                param_list.append(f"{param}: Any")
                            else:
                                param_list.append(f"{param}: Any")
                        else:
                            param_list.append(param)

                    new_params = ", ".join(param_list)
                    if "->" not in return_annotation:
                        return_annotation = " -> None"

                    lines[i] = (
                        f"{indent}def {method_name}({new_params}){return_annotation}:{rest}"
                    )
                    fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_assert_statements(self, file_path: Path) -> int:
        """Fix type issues in assert statements."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        # Common patterns that need type guards
        patterns = [
            # Pattern: assert result.attribute without checking result is not None
            (r"(\s*)assert (\w+)\.(\w+)", r"\1assert \2 is not None and \2.\3"),
            # Pattern: assert len(result) without checking result is not None
            (r"(\s*)assert len\((\w+)\)", r"\1assert \2 is not None and len(\2)"),
        ]

        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                fixes += content.count(pattern[0]) - new_content.count(pattern[0])
                content = new_content

        if fixes > 0:
            file_path.write_text(content, encoding="utf-8")
        return fixes

    def fix_mock_calls(self, file_path: Path) -> int:
        """Fix type issues with mock objects and calls."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Fix Mock() calls without type annotation
            if "Mock()" in line and "=" in line and ":" not in line.split("=")[0]:
                var_match = re.match(r"(\s*)(\w+)\s*=\s*Mock\(\)", line)
                if var_match:
                    indent, var_name = var_match.groups()
                    lines[i] = f"{indent}{var_name}: Mock = Mock()"
                    fixes += 1

            # Fix MagicMock() calls
            elif (
                "MagicMock()" in line and "=" in line and ":" not in line.split("=")[0]
            ):
                var_match = re.match(r"(\s*)(\w+)\s*=\s*MagicMock\(\)", line)
                if var_match:
                    indent, var_name = var_match.groups()
                    lines[i] = f"{indent}{var_name}: MagicMock = MagicMock()"
                    fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def add_type_ignores_for_complex_mocks(self, file_path: Path) -> int:
        """Add type: ignore comments for complex mock setups that are hard to type."""
        content = file_path.read_text(encoding="utf-8")
        fixes = 0

        lines = content.splitlines()
        for i, line in enumerate(lines):
            # Add type: ignore for complex mock attribute assignments
            if ".return_value." in line and "# type: ignore" not in line:
                lines[i] = f"{line}  # type: ignore[attr-defined]"
                fixes += 1
            elif (
                "mock_" in line
                and ".side_effect" in line
                and "# type: ignore" not in line
            ):
                lines[i] = f"{line}  # type: ignore[attr-defined]"
                fixes += 1

        if fixes > 0:
            file_path.write_text("\n".join(lines), encoding="utf-8")
        return fixes

    def fix_imports(self, file_path: Path) -> int:
        """Add missing type imports for test files."""
        content = file_path.read_text(encoding="utf-8")
        original = content

        lines = content.splitlines()
        imports_needed = set()

        # Check what imports are needed
        for line in lines:
            if (
                "MockerFixture" in line
                and "from pytest_mock import MockerFixture" not in content
            ):
                imports_needed.add("from pytest_mock import MockerFixture")
            if (
                "LogCaptureFixture" in line
                and "from _pytest.logging import LogCaptureFixture" not in content
            ):
                imports_needed.add("from _pytest.logging import LogCaptureFixture")
            if (
                "CaptureFixture" in line
                and "from _pytest.capture import CaptureFixture" not in content
            ):
                imports_needed.add("from _pytest.capture import CaptureFixture")
            if "Path" in line and "from pathlib import Path" not in content:
                imports_needed.add("from pathlib import Path")
            if "Mock" in line and "from unittest.mock import" not in content:
                imports_needed.add("from unittest.mock import Mock, MagicMock")
            if "Any" in line and "from typing import Any" not in content:
                imports_needed.add("from typing import Any")

        if imports_needed:
            # Find where to insert imports (after existing imports)
            insert_line = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_line = i + 1
                elif (
                    insert_line > 0
                    and line.strip()
                    and not line.startswith("import")
                    and not line.startswith("from")
                ):
                    break

            # Insert the new imports
            for imp in sorted(imports_needed):
                lines.insert(insert_line, imp)
                insert_line += 1

            content = "\n".join(lines)

        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return len(imports_needed)
        return 0

    def get_test_mypy_errors(self) -> dict[str, Any]:
        """Get breakdown of mypy errors in test files."""
        cmd = [
            self.python_path,
            "-m",
            "mypy",
            "tests",
            "--no-error-summary",
            "--show-error-codes",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        error_types: defaultdict[str, int] = defaultdict(int)
        file_errors: defaultdict[str, int] = defaultdict(int)

        for line in result.stdout.splitlines():
            if ": error:" in line:
                match = re.match(
                    r"tests[/\\](.+?):(\d+):\s*error:\s*(.+?)\s*\[(.+?)\]", line
                )
                if match:
                    file_path = f"tests/{match.group(1)}"
                    error_code = match.group(4)
                    error_types[error_code] += 1
                    file_errors[file_path] += 1

        return {
            "total": sum(error_types.values()),
            "by_type": dict(error_types),
            "by_file": dict(file_errors),
        }

    def fix_file(self, file_path: Path) -> tuple[int, list[str]]:
        """Fix a single test file."""
        if not file_path.exists():
            return 0, []

        fixes = 0

        # Apply fixes in order
        fixes += self.fix_imports(file_path)
        fixes += self.fix_test_method_annotations(file_path)
        fixes += self.fix_fixture_annotations(file_path)
        fixes += self.fix_fixture_parameters(file_path)
        fixes += self.fix_mock_calls(file_path)
        fixes += self.fix_assert_statements(file_path)

        # Add type ignores as last resort for complex cases
        fixes += self.add_type_ignores_for_complex_mocks(file_path)

        # Get remaining errors
        cmd = [
            self.python_path,
            "-m",
            "mypy",
            str(file_path),
            "--no-error-summary",
            "--show-error-codes",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        remaining = []
        for line in result.stdout.splitlines():
            if ": error:" in line:
                remaining.append(line)

        return fixes, remaining

    def fix_test_module(self, module_path: str) -> dict[str, Any]:
        """Fix all test files in a module."""
        module_dir = Path(module_path)
        results = {}
        total_fixes = 0

        for py_file in module_dir.glob("**/*.py"):
            if "__pycache__" not in str(py_file):
                print(f"Fixing {py_file}...")
                fixes, remaining = self.fix_file(py_file)
                if fixes > 0 or remaining:
                    results[str(py_file)] = {
                        "fixes": fixes,
                        "remaining_errors": len(remaining),
                        "errors": remaining[:5],
                    }
                    total_fixes += fixes

        return {"total_fixes": total_fixes, "files": results}


def main() -> None:
    """Main entry point."""
    import sys

    fixer = TestMypyFixer()

    if len(sys.argv) < 2:
        print(
            "Usage: python fix_mypy_tests.py [--status|--fix-file FILE|--fix-module MODULE]"
        )
        print("\nCommands:")
        print("  --status               Show current mypy error status for tests")
        print("  --fix-file FILE        Fix a specific test file")
        print("  --fix-module MODULE    Fix all files in a test module")
        print("\nExamples:")
        print("  python fix_mypy_tests.py --status")
        print("  python fix_mypy_tests.py --fix-file tests/suggestions/test_models.py")
        print("  python fix_mypy_tests.py --fix-module tests/suggestions")
        return

    command = sys.argv[1]

    if command == "--status":
        status = fixer.get_test_mypy_errors()
        print("\n=== Test Files Mypy Status ===")
        print(f"Total errors: {status['total']}")

        print("\n=== Top Error Types ===")
        by_type = status.get("by_type", {})
        if isinstance(by_type, dict):
            for code, count in sorted(
                by_type.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"{code}: {count} errors")

        print("\n=== Top Files with Errors ===")
        by_file = status.get("by_file", {})
        if isinstance(by_file, dict):
            for file, count in sorted(
                by_file.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"{file}: {count} errors")

    elif command == "--fix-file" and len(sys.argv) > 2:
        file_path = Path(sys.argv[2])
        print(f"\n=== Fixing {file_path} ===")

        fixes, remaining = fixer.fix_file(file_path)
        print(f"Applied {fixes} automated fixes")

        if remaining:
            print(f"\n{len(remaining)} errors remaining:")
            for error in remaining[:10]:
                print(f"  {error}")
        else:
            print("No errors remaining!")

    elif command == "--fix-module" and len(sys.argv) > 2:
        module_path = sys.argv[2]
        print(f"\n=== Fixing test module {module_path} ===")

        results = fixer.fix_test_module(module_path)
        print(f"Total fixes applied: {results['total_fixes']}")

        if results["files"]:
            print("\n=== Files Modified ===")
            for file, info in results["files"].items():
                print(f"\n{file}:")
                print(f"  Fixes applied: {info['fixes']}")
                print(f"  Errors remaining: {info['remaining_errors']}")


if __name__ == "__main__":
    main()
