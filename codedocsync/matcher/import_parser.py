import ast
import logging
from pathlib import Path

from .contextual_models import ImportStatement, ImportType, ModuleInfo

logger = logging.getLogger(__name__)


class ImportParser:
    """Parses Python imports and module structure."""

    def parse_imports(self, file_path: str) -> tuple[list[ImportStatement], set[str]]:
        """
        Parse all imports and exports from a Python file.

        Returns:
            Tuple of (imports, exports)
            - imports: List of ImportStatement objects
            - exports: Set of exported names (from __all__ or public names)
        """
        imports = []
        exports = set()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

        except UnicodeDecodeError:
            try:
                with open(file_path, encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read {file_path} with fallback encoding: {e}")
                return [], set()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return [], set()

        try:
            tree = ast.parse(content, filename=file_path)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(self._parse_import(node))
                elif isinstance(node, ast.ImportFrom):
                    imports.extend(self._parse_import_from(node))

            # Extract exports (__all__)
            exports = self._extract_exports(tree)

            # If no explicit exports, find all public names
            if not exports:
                exports = self._extract_public_names(tree)

            return imports, exports

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return [], set()
        except Exception as e:
            logger.error(f"Failed to parse imports from {file_path}: {e}")
            return [], set()

    def _parse_import(self, node: ast.Import) -> list[ImportStatement]:
        """Parse 'import' statement."""
        imports = []
        for alias in node.names:
            imports.append(
                ImportStatement(
                    import_type=ImportType.STANDARD,
                    module_path=alias.name,
                    imported_names=[],
                    aliases={alias.asname: alias.name} if alias.asname else {},
                    line_number=node.lineno,
                )
            )
        return imports

    def _parse_import_from(self, node: ast.ImportFrom) -> list[ImportStatement]:
        """Parse 'from ... import ...' statement."""
        if node.module is None:
            # Relative import like 'from . import something'
            module_path = "." * node.level
            import_type = ImportType.RELATIVE
        else:
            module_path = node.module
            import_type = ImportType.RELATIVE if node.level > 0 else ImportType.FROM

        # Handle wildcards
        imported_names = []
        aliases = {}

        for alias in node.names:
            if alias.name == "*":
                import_type = ImportType.WILDCARD
                imported_names = ["*"]
                break
            else:
                imported_names.append(alias.name)
                if alias.asname:
                    aliases[alias.asname] = alias.name

        return [
            ImportStatement(
                import_type=import_type,
                module_path=module_path,
                imported_names=imported_names,
                aliases=aliases,
                line_number=node.lineno,
                level=node.level,
            )
        ]

    def _extract_exports(self, tree: ast.AST) -> set[str]:
        """Extract __all__ exports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        # Extract list of strings
                        if isinstance(node.value, ast.List):
                            return {
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            }
        return set()

    def _extract_public_names(self, tree: ast.AST) -> set[str]:
        """Extract all public function and class names."""
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    names.add(node.name)
        return names

    def build_module_info(self, file_path: str, module_path: str) -> ModuleInfo:
        """Build complete module information."""
        imports, exports = self.parse_imports(file_path)

        # Determine if it's a package
        path = Path(file_path)
        is_package = path.name == "__init__.py"

        return ModuleInfo(
            module_path=module_path,
            file_path=file_path,
            imports=imports,
            exports=exports,
            is_package=is_package,
        )
