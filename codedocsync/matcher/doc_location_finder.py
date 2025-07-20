import ast
import logging
from pathlib import Path

from ..parser import DocstringParser, ParsedDocstring

logger = logging.getLogger(__name__)


class DocLocationFinder:
    """Finds documentation in non-standard locations."""

    def __init__(self):
        self.docstring_parser = DocstringParser()
        self._cache: dict[str, dict[str, ParsedDocstring]] = {}

    def find_module_docs(self, module_path: str) -> dict[str, ParsedDocstring]:
        """
        Find all docstrings in a module that might document functions.

        Returns:
            Dict mapping potential function names to their docs
        """
        if module_path in self._cache:
            return self._cache[module_path]

        docs = {}

        try:
            with open(module_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for encoding issues
            try:
                with open(module_path, encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read {module_path}: {e}")
                return {}

        try:
            tree = ast.parse(content, filename=module_path)

            # Check module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                # Parse for function documentation sections
                func_docs = self._extract_function_docs_from_module(module_doc)
                docs.update(func_docs)

            # Check class docstrings for method documentation
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        method_docs = self._extract_method_docs_from_class(
                            class_doc, node.name
                        )
                        docs.update(method_docs)

            self._cache[module_path] = docs
            return docs

        except SyntaxError as e:
            logger.error(f"Syntax error in {module_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to find docs in {module_path}: {e}")
            return {}

    def find_package_docs(self, package_path: str) -> dict[str, ParsedDocstring]:
        """
        Find documentation in package __init__.py files.

        Some projects document all module functions in __init__.py
        """
        init_file = Path(package_path) / "__init__.py"
        if not init_file.exists():
            return {}

        return self.find_module_docs(str(init_file))

    def _extract_function_docs_from_module(
        self, module_docstring: str
    ) -> dict[str, ParsedDocstring]:
        """
        Extract function documentation from module docstring.

        Looks for sections like:
        Functions:
        ----------
        function_name: Description
            Full documentation here
        """
        docs = {}

        # Try to parse the module docstring and extract function information
        try:
            parsed_doc = self.docstring_parser.parse(module_docstring)

            # Look for function documentation in the description or examples
            if parsed_doc and parsed_doc.description:
                # Simple heuristic: look for function-like patterns
                lines = parsed_doc.description.split("\n")
                current_function = None
                current_doc_lines = []

                for line in lines:
                    # Look for function name patterns: "function_name:"
                    if ":" in line and not line.strip().startswith(" "):
                        # Save previous function if any
                        if current_function and current_doc_lines:
                            func_doc_text = "\n".join(current_doc_lines).strip()
                            if func_doc_text:
                                func_parsed = self.docstring_parser.parse(func_doc_text)
                                if func_parsed:
                                    docs[current_function] = func_parsed

                        # Extract function name
                        potential_func = line.split(":")[0].strip()
                        if potential_func.isidentifier():
                            current_function = potential_func
                            current_doc_lines = [line.split(":", 1)[1].strip()]
                        else:
                            current_function = None
                            current_doc_lines = []
                    elif current_function:
                        current_doc_lines.append(line)

                # Save last function
                if current_function and current_doc_lines:
                    func_doc_text = "\n".join(current_doc_lines).strip()
                    if func_doc_text:
                        func_parsed = self.docstring_parser.parse(func_doc_text)
                        if func_parsed:
                            docs[current_function] = func_parsed

        except Exception as e:
            logger.debug(f"Failed to extract function docs from module docstring: {e}")

        return docs

    def _extract_method_docs_from_class(
        self, class_docstring: str, class_name: str
    ) -> dict[str, ParsedDocstring]:
        """
        Extract method documentation from class docstring.

        Some styles document all methods in the class docstring.
        """
        docs = {}

        try:
            parsed_doc = self.docstring_parser.parse(class_docstring)

            # Look for method documentation in the class docstring
            if parsed_doc and parsed_doc.description:
                # Simple heuristic: look for method-like patterns
                lines = parsed_doc.description.split("\n")
                current_method = None
                current_doc_lines = []

                for line in lines:
                    # Look for method name patterns: "method_name:"
                    if ":" in line and not line.strip().startswith(" "):
                        # Save previous method if any
                        if current_method and current_doc_lines:
                            method_doc_text = "\n".join(current_doc_lines).strip()
                            if method_doc_text:
                                method_parsed = self.docstring_parser.parse(
                                    method_doc_text
                                )
                                if method_parsed:
                                    # Use class.method format for method names
                                    docs[f"{class_name}.{current_method}"] = (
                                        method_parsed
                                    )

                        # Extract method name
                        potential_method = line.split(":")[0].strip()
                        if potential_method.isidentifier():
                            current_method = potential_method
                            current_doc_lines = [line.split(":", 1)[1].strip()]
                        else:
                            current_method = None
                            current_doc_lines = []
                    elif current_method:
                        current_doc_lines.append(line)

                # Save last method
                if current_method and current_doc_lines:
                    method_doc_text = "\n".join(current_doc_lines).strip()
                    if method_doc_text:
                        method_parsed = self.docstring_parser.parse(method_doc_text)
                        if method_parsed:
                            docs[f"{class_name}.{current_method}"] = method_parsed

        except Exception as e:
            logger.debug(f"Failed to extract method docs from class docstring: {e}")

        return docs

    def find_related_docs(
        self, function_name: str, module_path: str, search_radius: int = 2
    ) -> list[tuple[str, ParsedDocstring]]:
        """
        Search for documentation in related files.

        Args:
            function_name: Name of the function
            module_path: Current module path
            search_radius: How many directories up to search

        Returns:
            List of (file_path, docstring) tuples
        """
        related_docs = []

        # Search in parent directories up to search_radius
        current_path = Path(module_path).parent

        for _ in range(search_radius):
            # Check __init__.py in current directory
            init_file = current_path / "__init__.py"
            if init_file.exists():
                docs = self.find_package_docs(str(current_path))
                if function_name in docs:
                    related_docs.append((str(init_file), docs[function_name]))

            # Check README or docs files
            for doc_file in ["README.md", "DOCS.md", f"{function_name}.md"]:
                doc_path = current_path / doc_file
                if doc_path.exists():
                    # For now, we don't parse markdown files
                    # This would need a markdown parser in practice
                    pass

            # Move up one directory
            if current_path.parent == current_path:
                break
            current_path = current_path.parent

        return related_docs

    def clear_cache(self) -> None:
        """Clear the documentation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_files": len(self._cache),
            "total_docs": sum(len(docs) for docs in self._cache.values()),
        }
