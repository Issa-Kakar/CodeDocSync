"""
NumPy-style docstring template implementation.

This module provides template functionality for generating NumPy-style docstrings,
which uses underlined section headers and specific formatting conventions.
"""

import re

from ...parser.docstring_models import (
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
)
from ..models import DocstringStyle
from .base import DocstringTemplate


class NumpyStyleTemplate(DocstringTemplate):
    """Generate NumPy-style docstrings."""

    def __init__(
        self, style: DocstringStyle = DocstringStyle.NUMPY, max_line_length: int = 88
    ):
        """Initialize NumPy template."""
        super().__init__(style, max_line_length)

    def render_parameters(self, parameters: list[DocstringParameter]) -> list[str]:
        """
        Render parameters in NumPy style.

        NumPy style uses underlined headers and specific spacing:

        Parameters
        ----------
        param_name : type
            Description of parameter that can span
            multiple lines with consistent indent.
        """
        if not parameters:
            return []

        lines = ["Parameters", "-" * 10]

        for i, param in enumerate(parameters):
            # Add blank line between parameters (except first)
            if i > 0:
                lines.append("")

            # Format parameter line: name : type
            param_line = self._format_parameter_line(param)
            lines.append(param_line)

            # Add description with 4-space indent
            if param.description:
                desc_lines = self._wrap_text(
                    param.description.strip(), subsequent_indent="    "
                )
                for desc_line in desc_lines:
                    lines.append(f"    {desc_line}")

        return lines

    def render_returns(self, returns: DocstringReturns) -> list[str]:
        """
        Render returns in NumPy style.

        Returns
        -------
        return_name : type
            Description of return value.
        """
        if not returns or not returns.description:
            return []

        lines = ["Returns", "-" * 7]

        # Format return line
        return_line = self._format_return_line(returns)
        lines.append(return_line)

        # Add description with 4-space indent
        if returns.description:
            desc_lines = self._wrap_text(
                returns.description.strip(), subsequent_indent="    "
            )
            for desc_line in desc_lines:
                lines.append(f"    {desc_line}")

        return lines

    def render_raises(self, raises: list[DocstringRaises]) -> list[str]:
        """
        Render raises in NumPy style.

        Raises
        ------
        ExceptionType
            Description of when this exception is raised.
        """
        if not raises:
            return []

        lines = ["Raises", "-" * 6]

        for i, exception in enumerate(raises):
            # Add blank line between exceptions (except first)
            if i > 0:
                lines.append("")

            # Exception type
            if exception.exception_type:
                lines.append(exception.exception_type)
            else:
                lines.append("Exception")

            # Add description with 4-space indent
            if exception.description:
                desc_lines = self._wrap_text(
                    exception.description.strip(), subsequent_indent="    "
                )
                for desc_line in desc_lines:
                    lines.append(f"    {desc_line}")

        return lines

    def _format_parameter_line(self, param: DocstringParameter) -> str:
        """Format parameter line for NumPy style."""
        if param.type_annotation:
            type_str = self._format_type_annotation(param.type_annotation)
            return f"{param.name} : {type_str}"
        else:
            return f"{param.name}"

    def _format_return_line(self, returns: DocstringReturns) -> str:
        """Format return line for NumPy style."""
        # NumPy style often uses generic names like 'result' or 'output'
        return_name = getattr(returns, "name", None) or "result"

        if returns.type_annotation:
            type_str = self._format_type_annotation(returns.type_annotation)
            return f"{return_name} : {type_str}"
        else:
            return return_name

    def _match_parameter_line(self, line: str) -> str | None:
        """Match NumPy parameter definition and extract parameter name."""
        # NumPy format: "param_name : type" or just "param_name"
        match = re.match(r"^(\w+)\s*(?::\s*.+)?$", line.strip())
        if match:
            return match.group(1)
        return None

    def _merge_descriptions(
        self, new_lines: list[str], descriptions: dict[str, str]
    ) -> list[str]:
        """Merge preserved descriptions into NumPy structure."""
        merged_lines = []
        i = 0

        while i < len(new_lines):
            line = new_lines[i]
            merged_lines.append(line)

            # Check if this is a parameter line
            param_name = self._match_parameter_line(line)
            if param_name and param_name in descriptions:
                # Skip the generated description and use preserved one
                i += 1
                while i < len(new_lines) and new_lines[i].startswith("    "):
                    i += 1  # Skip generated description

                # Add preserved description
                preserved_desc = descriptions[param_name]
                desc_lines = self._wrap_text(preserved_desc, subsequent_indent="    ")
                for desc_line in desc_lines:
                    merged_lines.append(f"    {desc_line}")
                continue

            i += 1

        return merged_lines

    def _render_examples(self, examples: list[str]) -> list[str]:
        """Render examples section in NumPy style."""
        if not examples:
            return []

        lines = ["Examples", "-" * 8, ""]

        for example in examples:
            # NumPy style examples often use doctest format
            example_lines = example.strip().split("\n")
            for example_line in example_lines:
                lines.append(example_line)
            lines.append("")

        return lines[:-1]  # Remove trailing empty line

    def _format_type_annotation(self, type_str: str | None) -> str:
        """Format type annotation for NumPy style."""
        if not type_str:
            return ""

        # Call parent method for basic formatting
        formatted = super()._format_type_annotation(type_str)

        # NumPy-specific type formatting
        # Handle array types
        if "ndarray" in formatted or "np.ndarray" in formatted:
            return "array_like"

        # Simplify complex types for readability
        if "List[" in formatted:
            formatted = formatted.replace("List[", "list of ").replace("]", "")
        if "Dict[" in formatted:
            formatted = formatted.replace("Dict[", "dict of ").replace("]", "")

        return formatted
