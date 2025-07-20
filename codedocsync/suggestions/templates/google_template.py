"""
Google-style docstring template implementation.

This module provides template functionality for generating Google-style docstrings,
which is one of the most popular Python docstring formats.
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


class GoogleStyleTemplate(DocstringTemplate):
    """Generate Google-style docstrings."""

    def __init__(
        self, style: DocstringStyle = DocstringStyle.GOOGLE, max_line_length: int = 88
    ):
        """Initialize Google template."""
        super().__init__(style, max_line_length)

    def render_parameters(self, parameters: List[DocstringParameter]) -> List[str]:
        """
        Render parameters in Google style.

        Example output:
            Args:
                param_name (type): Description of parameter.
                    Can span multiple lines with proper indentation.
                another_param: Description without type.
        """
        if not parameters:
            return []

        lines = ["Args:"]

        for param in parameters:
            # Format parameter line
            param_line = self._format_parameter_line(param)
            lines.append(f"    {param_line}")

            # Add wrapped description if it spans multiple lines
            if param.description:
                desc_lines = self._wrap_parameter_description(param.description)
                lines.extend(desc_lines)

        return lines

    def render_returns(self, returns: DocstringReturns) -> List[str]:
        """
        Render return documentation in Google style.

        Example output:
            Returns:
                type: Description of return value.
        """
        if not returns or not returns.description:
            return []

        lines = ["Returns:"]

        # Format return line
        if returns.type_str:
            type_formatted = self._format_type_annotation(returns.type_str)
            return_line = f"{type_formatted}: {returns.description}"
        else:
            return_line = returns.description

        # Wrap and indent description
        wrapped_lines = self._wrap_text(return_line, subsequent_indent="    ")
        for line in wrapped_lines:
            lines.append(f"    {line}")

        return lines

    def render_raises(self, raises: List[DocstringRaises]) -> List[str]:
        """
        Render exception documentation in Google style.

        Example output:
            Raises:
                ValueError: When input is invalid.
                TypeError: When type is wrong.
        """
        if not raises:
            return []

        lines = ["Raises:"]

        for exc in raises:
            # Format exception line
            exc_line = f"{exc.exception_type}: {exc.description}"

            # Wrap and indent description
            wrapped_lines = self._wrap_text(exc_line, subsequent_indent="    ")
            for line in wrapped_lines:
                lines.append(f"    {line}")

        return lines

    def _format_parameter_line(self, param: DocstringParameter) -> str:
        """Format a single parameter line."""
        # Handle special parameter names (*args, **kwargs)
        name = param.name

        # Format type annotation
        type_part = ""
        if param.type_str:
            formatted_type = self._format_type_annotation(param.type_str)
            type_part = f" ({formatted_type})"

        # Add description
        description = param.description or ""

        # Combine parts
        if description:
            return f"{name}{type_part}: {description}"
        else:
            return f"{name}{type_part}:"

    def _wrap_parameter_description(self, description: str) -> List[str]:
        """Wrap parameter description with proper indentation."""
        if not description:
            return []

        # Calculate indentation for continuation lines (8 spaces for Google style)
        continuation_indent = "        "

        # Check if description fits on first line
        first_line_max = self.max_line_length - 8  # Account for initial indentation

        if len(description) <= first_line_max:
            return []  # Description fits on parameter line

        # Wrap with continuation indentation
        wrapped_lines = self._wrap_text(
            description, subsequent_indent=continuation_indent
        )

        # Add proper indentation to all lines
        result = []
        for line in wrapped_lines:
            if line.strip():  # Only indent non-empty lines
                result.append(continuation_indent + line.strip())
            else:
                result.append("")

        return result

    def _match_parameter_line(self, line: str) -> Optional[str]:
        """Match Google-style parameter line and extract parameter name."""
        # Google style: "    param_name (type): description"
        # or: "    param_name: description"
        match = re.match(r"^\s+([a-zA-Z_*][a-zA-Z0-9_]*)\s*(?:\([^)]+\))?\s*:", line)
        if match:
            return match.group(1)
        return None

    def _merge_descriptions(
        self, new_lines: List[str], descriptions: Dict[str, str]
    ) -> List[str]:
        """Merge preserved descriptions into new Google-style structure."""
        if not descriptions:
            return new_lines

        result = []
        current_param = None

        for line in new_lines:
            # Check if this is a parameter line
            param_name = self._match_parameter_line(line)
            if param_name:
                current_param = param_name

                # If we have a preserved description, use it
                if param_name in descriptions:
                    # Replace description in parameter line
                    base_line = re.sub(
                        r":\s*.*$", f": {descriptions[param_name]}", line
                    )
                    result.append(base_line)
                else:
                    result.append(line)
            else:
                # For non-parameter lines, add as-is unless we're preserving a description
                if current_param and current_param in descriptions:
                    # Skip continuation lines for parameters we're replacing
                    if line.strip() and not line.strip().startswith(
                        ("Args:", "Returns:", "Raises:")
                    ):
                        continue

                result.append(line)

                # Reset current_param on section boundaries
                if line.strip() in ["Args:", "Returns:", "Raises:", "Examples:"]:
                    current_param = None

        return result

    def _render_examples(self, examples: List[str]) -> List[str]:
        """Render examples section in Google style."""
        if not examples:
            return []

        lines = ["Examples:"]

        for i, example in enumerate(examples):
            if i > 0:
                lines.append("")  # Blank line between examples

            # Add example content with proper indentation
            example_lines = example.strip().split("\n")
            for example_line in example_lines:
                if example_line.strip():
                    lines.append(f"    {example_line}")
                else:
                    lines.append("")

        return lines

    def render_section_update(
        self,
        section_type: str,
        content: List[DocstringParameter],
        existing_docstring: Optional[str] = None,
    ) -> List[str]:
        """Render just a specific section for partial updates."""
        if section_type == "parameters":
            return self.render_parameters(content)
        elif section_type == "returns" and isinstance(content, DocstringReturns):
            return self.render_returns(content)
        elif section_type == "raises":
            return self.render_raises(content)
        else:
            raise ValueError(f"Unsupported section type: {section_type}")

    def extract_section_boundaries(
        self, docstring_lines: List[str]
    ) -> Dict[str, tuple]:
        """Extract section boundaries for precise section replacement."""
        boundaries = {}
        current_section = None
        start_line = None

        for i, line in enumerate(docstring_lines):
            line_stripped = line.strip()

            # Check for section headers
            if line_stripped == "Args:":
                if current_section and start_line is not None:
                    boundaries[current_section] = (start_line, i - 1)
                current_section = "parameters"
                start_line = i
            elif line_stripped == "Returns:":
                if current_section and start_line is not None:
                    boundaries[current_section] = (start_line, i - 1)
                current_section = "returns"
                start_line = i
            elif line_stripped == "Raises:":
                if current_section and start_line is not None:
                    boundaries[current_section] = (start_line, i - 1)
                current_section = "raises"
                start_line = i
            elif line_stripped == "Examples:":
                if current_section and start_line is not None:
                    boundaries[current_section] = (start_line, i - 1)
                current_section = "examples"
                start_line = i
            elif (
                line_stripped.endswith('"""')
                and current_section
                and start_line is not None
            ):
                # End of docstring
                boundaries[current_section] = (start_line, i - 1)
                break

        # Handle case where docstring ends without closing quotes
        if current_section and start_line is not None:
            boundaries[current_section] = (start_line, len(docstring_lines) - 1)

        return boundaries

    def replace_section(
        self, original_lines: List[str], section_type: str, new_section_lines: List[str]
    ) -> List[str]:
        """Replace a specific section in the docstring."""
        boundaries = self.extract_section_boundaries(original_lines)

        if section_type not in boundaries:
            # Section doesn't exist, append it
            return self._append_new_section(
                original_lines, section_type, new_section_lines
            )

        start, end = boundaries[section_type]

        # Replace the section
        result = original_lines[:start]
        result.extend(new_section_lines)
        result.extend(original_lines[end + 1 :])

        return result

    def _append_new_section(
        self, original_lines: List[str], section_type: str, new_section_lines: List[str]
    ) -> List[str]:
        """Append a new section to the docstring."""
        # Find insertion point (before closing quotes or at end)
        insert_point = len(original_lines)

        for i in range(len(original_lines) - 1, -1, -1):
            if original_lines[i].strip().endswith('"""'):
                insert_point = i
                break

        # Insert new section
        result = original_lines[:insert_point]

        # Add blank line if needed
        if result and result[-1].strip():
            result.append("")

        result.extend(new_section_lines)
        result.extend(original_lines[insert_point:])

        return result
