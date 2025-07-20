"""
Sphinx-style docstring template implementation.

This module provides template functionality for generating Sphinx-style docstrings,
which uses field lists with colon-prefixed directives.
"""

import re
from typing import List, Dict, Optional

from .base import DocstringTemplate
from ..models import DocstringStyle
from ...parser.docstring_models import (
    DocstringParameter,
    DocstringReturns,
    DocstringRaises,
)


class SphinxStyleTemplate(DocstringTemplate):
    """Generate Sphinx-style docstrings."""

    def __init__(
        self, style: DocstringStyle = DocstringStyle.SPHINX, max_line_length: int = 88
    ):
        """Initialize Sphinx template."""
        super().__init__(style, max_line_length)

    def render_parameters(self, parameters: List[DocstringParameter]) -> List[str]:
        """
        Render parameters in Sphinx style.

        Sphinx uses field lists with colons:

        :param param_name: Description of parameter
        :type param_name: str
        :param another: Description
        :type another: int, optional
        """
        if not parameters:
            return []

        lines = []

        for param in parameters:
            # :param name: description
            if param.description:
                param_line = f":param {param.name}: {param.description}"
                # Wrap long parameter descriptions
                wrapped_lines = self._wrap_sphinx_field(param_line, ":param")
                lines.extend(wrapped_lines)
            else:
                lines.append(f":param {param.name}:")

            # :type name: type annotation
            if param.type_annotation:
                type_str = self._format_type_annotation(param.type_annotation)
                type_line = f":type {param.name}: {type_str}"
                wrapped_lines = self._wrap_sphinx_field(type_line, ":type")
                lines.extend(wrapped_lines)

        return lines

    def render_returns(self, returns: DocstringReturns) -> List[str]:
        """
        Render returns in Sphinx style.

        :returns: Description of return value
        :rtype: type
        """
        if not returns:
            return []

        lines = []

        # :returns: description
        if returns.description:
            return_line = f":returns: {returns.description}"
            wrapped_lines = self._wrap_sphinx_field(return_line, ":returns")
            lines.extend(wrapped_lines)

        # :rtype: type
        if returns.type_annotation:
            type_str = self._format_type_annotation(returns.type_annotation)
            rtype_line = f":rtype: {type_str}"
            wrapped_lines = self._wrap_sphinx_field(rtype_line, ":rtype")
            lines.extend(wrapped_lines)

        return lines

    def render_raises(self, raises: List[DocstringRaises]) -> List[str]:
        """
        Render raises in Sphinx style.

        :raises ExceptionType: Description of when this exception is raised
        """
        if not raises:
            return []

        lines = []

        for exception in raises:
            if exception.exception_type and exception.description:
                raises_line = (
                    f":raises {exception.exception_type}: {exception.description}"
                )
                wrapped_lines = self._wrap_sphinx_field(raises_line, ":raises")
                lines.extend(wrapped_lines)
            elif exception.exception_type:
                lines.append(f":raises {exception.exception_type}:")
            elif exception.description:
                raises_line = f":raises Exception: {exception.description}"
                wrapped_lines = self._wrap_sphinx_field(raises_line, ":raises")
                lines.extend(wrapped_lines)

        return lines

    def _wrap_sphinx_field(self, field_line: str, field_prefix: str) -> List[str]:
        """Wrap Sphinx field line with proper continuation indentation."""
        if len(field_line) <= self.max_line_length:
            return [field_line]

        # Find the description part after the field prefix
        colon_pos = field_line.find(": ", len(field_prefix))
        if colon_pos == -1:
            return [field_line]

        field_part = field_line[: colon_pos + 2]
        description_part = field_line[colon_pos + 2 :]

        # Calculate continuation indent (align with description start)
        continuation_indent = " " * len(field_part)

        # Wrap the description part
        available_width = self.max_line_length - len(field_part)

        if len(description_part) <= available_width:
            return [field_line]

        # Split description and wrap
        lines = [field_part + description_part[:available_width]]
        remaining = description_part[available_width:]

        while remaining:
            chunk_size = self.max_line_length - len(continuation_indent)
            if len(remaining) <= chunk_size:
                lines.append(continuation_indent + remaining)
                break
            else:
                # Find a good break point
                chunk = remaining[:chunk_size]
                last_space = chunk.rfind(" ")
                if last_space > chunk_size * 0.7:  # If space is reasonably close to end
                    lines.append(continuation_indent + remaining[:last_space])
                    remaining = remaining[last_space + 1 :]
                else:
                    lines.append(continuation_indent + chunk)
                    remaining = remaining[chunk_size:]

        return lines

    def _match_parameter_line(self, line: str) -> Optional[str]:
        """Match Sphinx parameter definition and extract parameter name."""
        # Sphinx format: ":param name:" or ":param name: description"
        match = re.match(r"^:param\s+(\w+):", line.strip())
        if match:
            return match.group(1)
        return None

    def _merge_descriptions(
        self, new_lines: List[str], descriptions: Dict[str, str]
    ) -> List[str]:
        """Merge preserved descriptions into Sphinx structure."""
        merged_lines = []
        i = 0

        while i < len(new_lines):
            line = new_lines[i]

            # Check if this is a parameter line
            param_match = re.match(r"^:param\s+(\w+):", line.strip())
            if param_match:
                param_name = param_match.group(1)
                if param_name in descriptions:
                    # Replace with preserved description
                    preserved_desc = descriptions[param_name]
                    new_param_line = f":param {param_name}: {preserved_desc}"
                    wrapped_lines = self._wrap_sphinx_field(new_param_line, ":param")
                    merged_lines.extend(wrapped_lines)
                else:
                    merged_lines.append(line)
            else:
                merged_lines.append(line)

            i += 1

        return merged_lines

    def _render_examples(self, examples: List[str]) -> List[str]:
        """Render examples section in Sphinx style."""
        if not examples:
            return []

        lines = []

        for i, example in enumerate(examples):
            if i == 0:
                lines.append(".. rubric:: Examples")
                lines.append("")

            # Sphinx examples often use code-block directive
            lines.append(".. code-block:: python")
            lines.append("")

            example_lines = example.strip().split("\n")
            for example_line in example_lines:
                lines.append(f"   {example_line}")
            lines.append("")

        return lines[:-1] if lines else []  # Remove trailing empty line

    def _format_type_annotation(self, type_str: Optional[str]) -> str:
        """Format type annotation for Sphinx style."""
        if not type_str:
            return ""

        # Call parent method for basic formatting
        formatted = super()._format_type_annotation(type_str)

        # Sphinx-specific type formatting
        # Sphinx supports cross-references, so we can use more precise types
        type_mappings = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "list",
            "dict": "dict",
            "tuple": "tuple",
            "set": "set",
        }

        # Apply mappings for common types
        for original, sphinx_type in type_mappings.items():
            if formatted.lower() == original:
                return sphinx_type

        # Handle complex types
        if "List[" in formatted:
            # Convert List[str] to list of str
            inner_match = re.match(r"List\[([^\]]+)\]", formatted)
            if inner_match:
                inner_type = inner_match.group(1)
                return f"list of {inner_type}"

        if "Dict[" in formatted:
            # Convert Dict[str, Any] to dict
            return "dict"

        if "Optional[" in formatted:
            # Already handled by parent class
            pass

        return formatted

    def render_complete_docstring(
        self,
        summary: str,
        description: Optional[str] = None,
        parameters: Optional[List[DocstringParameter]] = None,
        returns: Optional[DocstringReturns] = None,
        raises: Optional[List[DocstringRaises]] = None,
        examples: Optional[List[str]] = None,
    ) -> str:
        """Render a complete Sphinx-style docstring from components."""
        lines = ['"""']

        # Add summary
        if summary:
            summary_lines = self._wrap_text(summary.strip())
            lines.extend(summary_lines)

        # Add description with blank line separator
        if description and description.strip():
            lines.append("")
            desc_lines = self._wrap_text(description.strip())
            lines.extend(desc_lines)

        # Add parameters section (no blank line before in Sphinx)
        if parameters:
            param_lines = self.render_parameters(parameters)
            if param_lines:
                lines.append("")
                lines.extend(param_lines)

        # Add returns section
        if returns:
            return_lines = self.render_returns(returns)
            if return_lines:
                if not parameters:  # Add blank line if no parameters before
                    lines.append("")
                lines.extend(return_lines)

        # Add raises section
        if raises:
            raises_lines = self.render_raises(raises)
            if raises_lines:
                if not parameters and not returns:  # Add blank line if needed
                    lines.append("")
                lines.extend(raises_lines)

        # Add examples section
        if examples:
            example_lines = self._render_examples(examples)
            if example_lines:
                lines.append("")
                lines.extend(example_lines)

        lines.append('"""')
        return "\n".join(lines)
