import tempfile
from pathlib import Path

from codedocsync.matcher.module_resolver import ModuleResolver
from codedocsync.matcher.contextual_models import (
    ImportStatement,
    ImportType,
    ModuleInfo,
)


class TestModuleResolver:
    """Test cases for ModuleResolver class."""

    def test_init_with_project_root(self):
        """Test initialization with project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)
            assert resolver.project_root == Path(temp_dir).resolve()
            assert len(resolver._python_paths) >= 1
            assert resolver.project_root in resolver._python_paths

    def test_calculate_python_paths_with_init_files(self):
        """Test Python path calculation with __init__.py files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested package structure
            (temp_path / "src").mkdir()
            (temp_path / "src" / "__init__.py").touch()
            (temp_path / "src" / "package").mkdir()
            (temp_path / "src" / "package" / "__init__.py").touch()

            # Test from deepest directory
            resolver = ModuleResolver(str(temp_path / "src" / "package"))

            # Should include parent directories with __init__.py
            assert temp_path / "src" / "package" in resolver._python_paths
            assert temp_path / "src" in resolver._python_paths

    def test_resolve_module_path_regular_file(self):
        """Test module path resolution for regular Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a Python file
            (temp_path / "src").mkdir()
            test_file = temp_path / "src" / "utils.py"
            test_file.touch()

            resolver = ModuleResolver(temp_dir)
            result = resolver.resolve_module_path(str(test_file))
            assert result == "src.utils"

    def test_resolve_module_path_init_file(self):
        """Test module path resolution for __init__.py files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a package
            (temp_path / "src" / "package").mkdir(parents=True)
            init_file = temp_path / "src" / "package" / "__init__.py"
            init_file.touch()

            resolver = ModuleResolver(temp_dir)
            result = resolver.resolve_module_path(str(init_file))
            assert result == "src.package"

    def test_resolve_module_path_outside_project(self):
        """Test module path resolution for files outside project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as other_dir:
                # Create file outside project
                other_file = Path(other_dir) / "external.py"
                other_file.touch()

                resolver = ModuleResolver(temp_dir)
                result = resolver.resolve_module_path(str(other_file))
                assert result is None

    def test_resolve_import_absolute(self):
        """Test resolution of absolute imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            import_stmt = ImportStatement(
                import_type=ImportType.STANDARD,
                module_path="os.path",
                imported_names=[],
                aliases={},
                line_number=1,
            )

            result = resolver.resolve_import(import_stmt, "current.module")
            assert result == "os.path"

    def test_resolve_import_relative_single_dot(self):
        """Test resolution of single-dot relative imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            import_stmt = ImportStatement(
                import_type=ImportType.RELATIVE,
                module_path="utils",
                imported_names=[],
                aliases={},
                line_number=1,
                level=1,
            )

            result = resolver.resolve_import(import_stmt, "package.submodule")
            assert result == "package.utils"

    def test_resolve_import_relative_double_dot(self):
        """Test resolution of double-dot relative imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            import_stmt = ImportStatement(
                import_type=ImportType.RELATIVE,
                module_path="other",
                imported_names=[],
                aliases={},
                line_number=1,
                level=2,
            )

            result = resolver.resolve_import(import_stmt, "package.sub.module")
            assert result == "package.other"

    def test_resolve_import_relative_beyond_top_level(self):
        """Test resolution of relative imports beyond top-level package."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            import_stmt = ImportStatement(
                import_type=ImportType.RELATIVE,
                module_path="other",
                imported_names=[],
                aliases={},
                line_number=1,
                level=3,
            )

            # Only 2 levels in current module
            result = resolver.resolve_import(import_stmt, "package.module")
            assert result is None

    def test_resolve_import_relative_dot_only(self):
        """Test resolution of dot-only relative imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            import_stmt = ImportStatement(
                import_type=ImportType.RELATIVE,
                module_path=".",
                imported_names=[],
                aliases={},
                line_number=1,
                level=1,
            )

            result = resolver.resolve_import(import_stmt, "package.submodule")
            assert result == "package"

    def test_build_import_chain_same_module(self):
        """Test building import chain for same module."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="package.module",
                to_module="package.module",
            )
            assert result == ["test_func"]

    def test_build_import_chain_direct_import(self):
        """Test building import chain for direct imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            # Mock module info with import
            mock_module = ModuleInfo(
                module_path="package.module",
                file_path="/path/to/module.py",
                imports=[
                    ImportStatement(
                        import_type=ImportType.FROM,
                        module_path="utils.helpers",
                        imported_names=["test_func"],
                        aliases={},
                        line_number=1,
                    )
                ],
            )
            resolver.module_cache["package.module"] = mock_module

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="package.module",
                to_module="utils.helpers",
            )
            assert result == ["test_func"]

    def test_build_import_chain_with_alias(self):
        """Test building import chain with aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            # Mock module info with aliased import
            mock_module = ModuleInfo(
                module_path="package.module",
                file_path="/path/to/module.py",
                imports=[
                    ImportStatement(
                        import_type=ImportType.FROM,
                        module_path="utils.helpers",
                        imported_names=["test_func"],
                        aliases={"alias_func": "test_func"},
                        line_number=1,
                    )
                ],
            )
            resolver.module_cache["package.module"] = mock_module

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="package.module",
                to_module="utils.helpers",
            )
            assert result == ["alias_func"]

    def test_build_import_chain_wildcard(self):
        """Test building import chain for wildcard imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            # Mock module info with wildcard import
            mock_module = ModuleInfo(
                module_path="package.module",
                file_path="/path/to/module.py",
                imports=[
                    ImportStatement(
                        import_type=ImportType.WILDCARD,
                        module_path="utils.helpers",
                        imported_names=["*"],
                        aliases={},
                        line_number=1,
                    )
                ],
            )
            resolver.module_cache["package.module"] = mock_module

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="package.module",
                to_module="utils.helpers",
            )
            assert result == ["test_func"]

    def test_build_import_chain_module_import(self):
        """Test building import chain for module imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            # Mock module info with module import
            mock_module = ModuleInfo(
                module_path="package.module",
                file_path="/path/to/module.py",
                imports=[
                    ImportStatement(
                        import_type=ImportType.STANDARD,
                        module_path="utils.helpers",
                        imported_names=[],
                        aliases={},
                        line_number=1,
                    )
                ],
            )
            resolver.module_cache["package.module"] = mock_module

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="package.module",
                to_module="utils.helpers",
            )
            assert result == ["utils.helpers.test_func"]

    def test_build_import_chain_no_module_info(self):
        """Test building import chain when module info is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)

            result = resolver.build_import_chain(
                target_function="test_func",
                from_module="missing.module",
                to_module="utils.helpers",
            )
            assert result is None

    def test_find_module_file_regular_module(self):
        """Test finding regular module file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create module file
            (temp_path / "src").mkdir()
            module_file = temp_path / "src" / "utils.py"
            module_file.touch()

            resolver = ModuleResolver(temp_dir)
            result = resolver.find_module_file("src.utils")
            assert result == str(module_file)

    def test_find_module_file_package(self):
        """Test finding package module file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create package
            (temp_path / "src" / "package").mkdir(parents=True)
            init_file = temp_path / "src" / "package" / "__init__.py"
            init_file.touch()

            resolver = ModuleResolver(temp_dir)
            result = resolver.find_module_file("src.package")
            assert result == str(init_file)

    def test_find_module_file_nonexistent(self):
        """Test finding nonexistent module file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModuleResolver(temp_dir)
            result = resolver.find_module_file("nonexistent.module")
            assert result is None

    def test_find_module_file_prefers_regular_over_package(self):
        """Test that regular module file is preferred over package."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create both regular module and package with same name
            module_file = temp_path / "utils.py"
            module_file.touch()

            (temp_path / "utils").mkdir()
            (temp_path / "utils" / "__init__.py").touch()

            resolver = ModuleResolver(temp_dir)
            result = resolver.find_module_file("utils")
            assert result == str(module_file)  # Regular module preferred
