from pathlib import Path
from typing import Dict, List, Optional
import logging

from .contextual_models import ModuleInfo, ImportStatement, ImportType
from .import_parser import ImportParser

logger = logging.getLogger(__name__)


class ModuleResolver:
    """Resolves module paths and import chains."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.import_parser = ImportParser()
        self.module_cache: Dict[str, ModuleInfo] = {}
        self._python_paths = self._calculate_python_paths()

    def _calculate_python_paths(self) -> List[Path]:
        """Calculate Python module search paths."""
        paths = [self.project_root]

        # Add parent directories that contain __init__.py
        current = self.project_root
        while current.parent != current:
            if (current.parent / "__init__.py").exists():
                paths.append(current.parent)
                current = current.parent
            else:
                break

        return paths

    def resolve_module_path(self, file_path: str) -> Optional[str]:
        """
        Convert file path to module path.

        Example:
            /project/src/utils/helpers.py -> src.utils.helpers
            /project/src/utils/__init__.py -> src.utils
        """
        file_path = Path(file_path).resolve()

        # Find the base path this file is relative to
        for base_path in self._python_paths:
            try:
                relative_path = file_path.relative_to(base_path)

                # Convert to module path
                parts = list(relative_path.parts)

                # Handle __init__.py
                if parts[-1] == "__init__.py":
                    parts = parts[:-1]
                elif parts[-1].endswith(".py"):
                    parts[-1] = parts[-1][:-3]
                else:
                    continue  # Not a Python file

                module_path = ".".join(parts)
                return module_path

            except ValueError:
                # file_path is not relative to this base_path
                continue

        return None

    def resolve_import(
        self, import_stmt: ImportStatement, current_module: str
    ) -> Optional[str]:
        """
        Resolve an import statement to absolute module path.

        Args:
            import_stmt: The import statement to resolve
            current_module: The module containing the import

        Returns:
            Absolute module path or None if can't resolve
        """
        if import_stmt.import_type == ImportType.RELATIVE:
            # Handle relative imports
            if import_stmt.level == 0:
                return import_stmt.module_path

            # Go up 'level' packages from current module
            parts = current_module.split(".")
            if import_stmt.level > len(parts):
                logger.warning(
                    f"Relative import goes beyond top-level package: "
                    f"{import_stmt} in {current_module}"
                )
                return None

            # Go up the required levels
            base_parts = parts[: -import_stmt.level]

            if import_stmt.module_path and import_stmt.module_path != ".":
                # from ..package import something
                base_parts.append(import_stmt.module_path.lstrip("."))

            return ".".join(base_parts) if base_parts else None

        else:
            # Absolute import
            return import_stmt.module_path

    def build_import_chain(
        self, target_function: str, from_module: str, to_module: str
    ) -> Optional[List[str]]:
        """
        Build the import chain to access a function from another module.

        Returns:
            List of import steps, or None if not accessible
        """
        if from_module == to_module:
            return [target_function]  # Same module, direct access

        # Check direct imports
        from_info = self.module_cache.get(from_module)
        if not from_info:
            return None

        for import_stmt in from_info.imports:
            resolved = self.resolve_import(import_stmt, from_module)
            if resolved == to_module:
                # Direct import found
                if import_stmt.import_type == ImportType.WILDCARD:
                    return [target_function]
                elif target_function in import_stmt.imported_names:
                    # Check for alias
                    for alias, real_name in import_stmt.aliases.items():
                        if real_name == target_function:
                            return [alias]
                    return [target_function]
                elif not import_stmt.imported_names:
                    # import module
                    alias = import_stmt.aliases.get(to_module, to_module)
                    return [f"{alias}.{target_function}"]

        # TODO: Check indirect imports (through parent packages)
        return None

    def find_module_file(self, module_path: str) -> Optional[str]:
        """Find the file path for a module."""
        # Convert module path to possible file paths
        path_parts = module_path.split(".")

        for base_path in self._python_paths:
            # Try as regular module
            file_path = base_path / Path(*path_parts).with_suffix(".py")
            if file_path.exists():
                return str(file_path)

            # Try as package
            package_path = base_path / Path(*path_parts) / "__init__.py"
            if package_path.exists():
                return str(package_path)

        return None
