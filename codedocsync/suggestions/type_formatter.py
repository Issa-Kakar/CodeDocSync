"""
Type annotation formatter for docstring generation.

This module provides utilities for formatting Python type annotations
in different docstring styles, handling complex types like Union, Optional,
generics, and modern Python typing constructs.
"""

from __future__ import annotations

import ast
import re
from enum import Enum

from .models import DocstringStyle


class TypeComplexity(Enum):
    """Complexity levels for type annotations."""

    SIMPLE = "simple"  # str, int, bool
    GENERIC = "generic"  # List[str], Dict[str, Any]
    UNION = "union"  # Union[str, int], Optional[str]
    COMPLEX = "complex"  # Callable[[int, str], bool]


class TypeAnnotationFormatter:
    """Format Python type annotations for docstrings."""

    def __init__(self, style: DocstringStyle = DocstringStyle.GOOGLE) -> None:
        """Initialize type formatter for specific docstring style."""
        self.style = style
        self._type_mappings = self._get_style_mappings()
        self._complexity_cache: dict[str, TypeComplexity] = {}

    def format_for_docstring(self, type_annotation: str) -> str:
        """
        Convert type annotation to docstring format.

        Args:
            type_annotation: Raw type annotation string

        Returns:
            Formatted type string appropriate for the docstring style
        """
        if not type_annotation:
            return ""

        # Normalize and clean the type string
        normalized = self._normalize_type_string(type_annotation)

        # Determine complexity
        complexity = self._assess_complexity(normalized)

        # Apply style-specific formatting
        if complexity == TypeComplexity.SIMPLE:
            return self._format_simple_type(normalized)
        elif complexity == TypeComplexity.GENERIC:
            return self._format_generic_type(normalized)
        elif complexity == TypeComplexity.UNION:
            return self._format_union_type(normalized)
        else:  # COMPLEX
            return self._format_complex_type(normalized)

    def extract_from_ast(self, node: ast.AST | None) -> str:
        """
        Extract type annotation from AST node.

        Args:
            node: AST node containing type annotation

        Returns:
            String representation of the type
        """
        if node is None:
            return ""

        try:
            # Handle different AST node types
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Attribute):
                value = self.extract_from_ast(node.value)
                return f"{value}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = self.extract_from_ast(node.value)
                slice_value = self.extract_from_ast(node.slice)
                return f"{value}[{slice_value}]"
            elif isinstance(node, ast.Tuple):
                elements = [self.extract_from_ast(elt) for elt in node.elts]
                return ", ".join(elements)
            elif isinstance(node, ast.List):
                elements = [self.extract_from_ast(elt) for elt in node.elts]
                return f"[{', '.join(elements)}]"
            elif hasattr(ast, "unparse"):  # Python 3.9+
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                return str(node)
        except Exception:
            # Fallback to string representation
            return str(node)

    def simplify_for_style(self, type_annotation: str) -> str:
        """
        Simplify complex types for better readability in docstrings.

        Args:
            type_annotation: Type annotation to simplify

        Returns:
            Simplified type string
        """
        simplified = type_annotation

        # Apply style-specific simplifications
        if self.style == DocstringStyle.NUMPY:
            simplified = self._simplify_for_numpy(simplified)
        elif self.style == DocstringStyle.SPHINX:
            simplified = self._simplify_for_sphinx(simplified)
        elif self.style == DocstringStyle.GOOGLE:
            simplified = self._simplify_for_google(simplified)

        return simplified

    def _normalize_type_string(self, type_str: str) -> str:
        """Normalize type string by removing extra whitespace and quotes."""
        # Remove quotes if the entire string is quoted
        type_str = type_str.strip()
        if (type_str.startswith('"') and type_str.endswith('"')) or (
            type_str.startswith("'") and type_str.endswith("'")
        ):
            type_str = type_str[1:-1]

        # Normalize whitespace around brackets and commas
        type_str = re.sub(r"\s*\[\s*", "[", type_str)
        type_str = re.sub(r"\s*\]\s*", "]", type_str)
        type_str = re.sub(r"\s*,\s*", ", ", type_str)
        type_str = re.sub(r"\s+", " ", type_str)

        return type_str.strip()

    def _assess_complexity(self, type_str: str) -> TypeComplexity:
        """Assess the complexity of a type annotation."""
        if type_str in self._complexity_cache:
            return self._complexity_cache[type_str]

        complexity = TypeComplexity.SIMPLE

        # Check for unions and optionals
        if any(pattern in type_str for pattern in ["Union[", "Optional[", " | "]):
            complexity = TypeComplexity.UNION
        # Check for generics
        elif any(
            pattern in type_str for pattern in ["[", "List", "Dict", "Tuple", "Set"]
        ):
            complexity = TypeComplexity.GENERIC
        # Check for complex types
        elif any(
            pattern in type_str for pattern in ["Callable", "Protocol", "TypeVar"]
        ):
            complexity = TypeComplexity.COMPLEX
        # Simple types
        elif type_str in {"str", "int", "float", "bool", "None", "object", "Any"}:
            complexity = TypeComplexity.SIMPLE

        self._complexity_cache[type_str] = complexity
        return complexity

    def _format_simple_type(self, type_str: str) -> str:
        """Format simple types."""
        return self._type_mappings.get(type_str, type_str)

    def _format_generic_type(self, type_str: str) -> str:
        """Format generic types like List[str], Dict[str, Any]."""
        # Handle List types
        list_match = re.match(r"List\[(.+)\]", type_str)
        if list_match:
            inner_type = list_match.group(1)
            if self.style == DocstringStyle.NUMPY:
                return f"list of {self.format_for_docstring(inner_type)}"
            elif self.style == DocstringStyle.SPHINX:
                return f"list of {self.format_for_docstring(inner_type)}"
            else:  # Google
                return f"List[{self.format_for_docstring(inner_type)}]"

        # Handle Dict types
        dict_match = re.match(r"Dict\[(.+)\]", type_str)
        if dict_match:
            if self.style in {DocstringStyle.NUMPY, DocstringStyle.SPHINX}:
                return "dict"
            else:  # Google
                return type_str

        # Handle Tuple types
        tuple_match = re.match(r"Tuple\[(.+)\]", type_str)
        if tuple_match:
            if self.style == DocstringStyle.NUMPY:
                return "tuple"
            else:
                return type_str

        # Handle Set types
        set_match = re.match(r"Set\[(.+)\]", type_str)
        if set_match:
            inner_type = set_match.group(1)
            if self.style == DocstringStyle.NUMPY:
                return f"set of {self.format_for_docstring(inner_type)}"
            else:
                return type_str

        return type_str

    def _format_union_type(self, type_str: str) -> str:
        """Format Union and Optional types."""
        # Handle Optional[T] -> T, optional
        optional_match = re.match(r"Optional\[(.+)\]", type_str)
        if optional_match:
            inner_type = optional_match.group(1)
            formatted_inner = self.format_for_docstring(inner_type)
            return f"{formatted_inner}, optional"

        # Handle Optional[T] -> T, optional
        union_none_match = re.match(r"Union\[(.+), None\]", type_str)
        if union_none_match:
            inner_type = union_none_match.group(1)
            formatted_inner = self.format_for_docstring(inner_type)
            return f"{formatted_inner}, optional"

        # Handle Union[A, B, C] -> A or B or C
        union_match = re.match(r"Union\[(.+)\]", type_str)
        if union_match:
            types = self._split_union_types(union_match.group(1))
            formatted_types = [self.format_for_docstring(t.strip()) for t in types]
            return " or ".join(formatted_types)

        # Handle new-style unions (Python 3.10+): Union[A, B]
        if " | " in type_str:
            types = type_str.split(" | ")
            formatted_types = [self.format_for_docstring(t.strip()) for t in types]
            return " or ".join(formatted_types)

        return type_str

    def _format_complex_type(self, type_str: str) -> str:
        """Format complex types by simplifying them."""
        # Callable types -> just "callable"
        if type_str.startswith("Callable"):
            return "callable"

        # Protocol types -> just the protocol name
        protocol_match = re.match(r"(\w+Protocol)", type_str)
        if protocol_match:
            return protocol_match.group(1)

        # TypeVar -> just the name
        typevar_match = re.match(r'TypeVar\([\'"](\w+)[\'"].*\)', type_str)
        if typevar_match:
            return typevar_match.group(1)

        # For very complex types, just return a simplified version
        if len(type_str) > 50:
            return "complex type"

        return type_str

    def _split_union_types(self, union_content: str) -> list[str]:
        """Split Union type content, handling nested brackets."""
        types = []
        current_type = ""
        bracket_depth = 0

        for char in union_content:
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "," and bracket_depth == 0:
                types.append(current_type.strip())
                current_type = ""
                continue

            current_type += char

        if current_type.strip():
            types.append(current_type.strip())

        return types

    def _get_style_mappings(self) -> dict[str, str]:
        """Get type mappings for the current style."""
        common_mappings = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "None": "None",
            "Any": "Any",
            "object": "object",
        }

        if self.style == DocstringStyle.NUMPY:
            return {
                **common_mappings,
                "List": "array_like",
                "np.ndarray": "array_like",
                "ndarray": "array_like",
                "numpy.ndarray": "array_like",
            }
        elif self.style == DocstringStyle.SPHINX:
            return {
                **common_mappings,
                "List": "list",
                "Dict": "dict",
                "Tuple": "tuple",
                "Set": "set",
            }
        else:  # Google and others
            return common_mappings

    def _simplify_for_numpy(self, type_str: str) -> str:
        """Apply NumPy-specific type simplifications."""
        # Convert array types
        numpy_arrays = ["np.ndarray", "numpy.ndarray", "ndarray"]
        for array_type in numpy_arrays:
            if array_type in type_str:
                return "array_like"

        # Simplify generic collections
        if type_str.startswith("List["):
            return "array_like"
        if type_str.startswith("Dict["):
            return "dict"

        return type_str

    def _simplify_for_sphinx(self, type_str: str) -> str:
        """Apply Sphinx-specific type simplifications."""
        # Sphinx can handle cross-references, so be more specific
        return type_str

    def _simplify_for_google(self, type_str: str) -> str:
        """Apply Google-style specific simplifications."""
        # Google style is generally more verbose, so less simplification
        return type_str


def format_type_for_style(
    type_annotation: str, style: DocstringStyle = DocstringStyle.GOOGLE
) -> str:
    """
    Convenience function to format a type annotation for a specific style.

    Args:
        type_annotation: Type annotation to format
        style: Target docstring style

    Returns:
        Formatted type string
    """
    formatter = TypeAnnotationFormatter(style)
    return formatter.format_for_docstring(type_annotation)


def extract_type_from_ast(node: ast.AST) -> str:
    """
    Convenience function to extract type from AST node.

    Args:
        node: AST node containing type information

    Returns:
        String representation of the type
    """
    formatter = TypeAnnotationFormatter()
    return formatter.extract_from_ast(node)
