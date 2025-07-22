"""
Base template system for docstring generation.

This module provides the foundation for all docstring templates, defining
the interface for rendering different sections and merging with existing content.
"""

import re
import textwrap
from abc import ABC, abstractmethod

from ...parser.docstring_models import (
    DocstringParameter,
    DocstringRaises,
    DocstringReturns,
    ParsedDocstring,
)
from ..models import DocstringStyle


class DocstringTemplate(ABC):
    """Base template for docstring generation."""

    def __init__(self, style: DocstringStyle, max_line_length: int = 88) -> None:
        """Initialize template with style and formatting options."""
        self.style = style
        self.max_line_length = max_line_length
        self.indent_size = 4

    @abstractmethod
    def render_parameters(self, parameters: list[DocstringParameter]) -> list[str]:
        """Render parameter documentation."""
        pass

    @abstractmethod
    def render_returns(self, returns: DocstringReturns) -> list[str]:
        """Render return documentation."""
        pass

    @abstractmethod
    def render_raises(self, raises: list[DocstringRaises]) -> list[str]:
        """Render exception documentation."""
        pass

    def render_complete_docstring(
        self,
        summary: str,
        description: str | None = None,
        parameters: list[DocstringParameter] | None = None,
        returns: DocstringReturns | None = None,
        raises: list[DocstringRaises] | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Render a complete docstring from components."""
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

        # Add parameters section
        if parameters:
            param_lines = self.render_parameters(parameters)
            if param_lines:
                lines.append("")
                lines.extend(param_lines)

        # Add returns section
        if returns:
            return_lines = self.render_returns(returns)
            if return_lines:
                lines.append("")
                lines.extend(return_lines)

        # Add raises section
        if raises:
            raises_lines = self.render_raises(raises)
            if raises_lines:
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

    def merge_with_existing(
        self,
        new_content: list[str],
        existing: list[str],
        preserve_descriptions: bool = True,
    ) -> list[str]:
        """Intelligently merge new content with existing."""
        if not existing or not preserve_descriptions:
            return new_content

        # Parse existing to extract descriptions
        existing_descriptions = self._extract_descriptions(existing)

        # Merge descriptions with new structure
        return self._merge_descriptions(new_content, existing_descriptions)

    def _wrap_text(self, text: str, subsequent_indent: str = "") -> list[str]:
        """Wrap text to max line length."""
        if not text:
            return []

        # Calculate available width (accounting for indentation)
        available_width = self.max_line_length - len(subsequent_indent)

        # Use textwrap for intelligent wrapping
        wrapped = textwrap.fill(
            text,
            width=available_width,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return wrapped.split("\n")

    def _format_type_annotation(self, type_str: str | None) -> str:
        """Format type annotation for docstring style."""
        if not type_str:
            return ""

        # Normalize common type representations
        type_str = type_str.strip()

        # Handle Optional types
        if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner_type = type_str[9:-1]
            return f"{inner_type}, optional"

        # Handle Union types with None
        if "Optional[" in type_str and "]" in type_str:
            # Extract non-None types
            union_match = re.match(r"Union\[(.*), None\]", type_str)
            if union_match:
                return f"{union_match.group(1)}, optional"

        return type_str

    def _extract_descriptions(self, lines: list[str]) -> dict[str, str]:
        """Extract existing descriptions from docstring lines."""
        descriptions = {}
        current_param = None
        current_desc_lines: list[str] = []

        for line in lines:
            line = line.strip()

            # Check for parameter definition
            param_match = self._match_parameter_line(line)
            if param_match:
                # Save previous parameter description
                if current_param and current_desc_lines:
                    descriptions[current_param] = " ".join(current_desc_lines).strip()

                current_param = param_match
                current_desc_lines = []
            elif (
                current_param
                and line
                and not line.startswith(("Args:", "Parameters", "Returns", "Raises"))
            ):
                # Continuation of description
                current_desc_lines.append(line)

        # Save last parameter
        if current_param and current_desc_lines:
            descriptions[current_param] = " ".join(current_desc_lines).strip()

        return descriptions

    def _match_parameter_line(self, line: str) -> str | None:
        """Match parameter definition line and extract parameter name."""
        # This is style-specific and should be overridden
        return None

    def _merge_descriptions(
        self, new_lines: list[str], descriptions: dict[str, str]
    ) -> list[str]:
        """Merge preserved descriptions into new structure."""
        # This is style-specific and should be overridden
        return new_lines

    def _render_examples(self, examples: list[str]) -> list[str]:
        """Render examples section (style-specific)."""
        if not examples:
            return []

        lines = ["Examples:", ""]
        for example in examples:
            # Simple example formatting
            example_lines = example.strip().split("\n")
            for example_line in example_lines:
                lines.append(f"    {example_line}")
            lines.append("")

        return lines[:-1]  # Remove trailing empty line

    def _calculate_indent(self, line: str) -> int:
        """Calculate indentation level of a line."""
        return len(line) - len(line.lstrip())

    def _add_indent(self, lines: list[str], indent: int) -> list[str]:
        """Add indentation to lines."""
        indent_str = " " * indent
        return [f"{indent_str}{line}" if line.strip() else line for line in lines]


class TemplateRegistry:
    """Registry for managing docstring templates."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._templates: dict[DocstringStyle, type[DocstringTemplate]] = {}

    def register(
        self, style: DocstringStyle, template_class: type[DocstringTemplate]
    ) -> None:
        """Register a template class for a style."""
        if not issubclass(template_class, DocstringTemplate):
            raise ValueError("Template class must inherit from DocstringTemplate")

        self._templates[style] = template_class

    def get_template(self, style: DocstringStyle, **kwargs: int) -> DocstringTemplate:
        """Get template instance for style."""
        if style not in self._templates:
            raise ValueError(f"No template registered for style: {style}")

        template_class = self._templates[style]
        return template_class(style, **kwargs)

    def available_styles(self) -> list[DocstringStyle]:
        """Get list of available styles."""
        return list(self._templates.keys())


# Global template registry
template_registry = TemplateRegistry()


def get_template(style: DocstringStyle, **kwargs: int) -> DocstringTemplate:
    """Convenience function to get template from global registry."""
    return template_registry.get_template(style, **kwargs)


class TemplateMerger:
    """Utility class for intelligent docstring merging."""

    @staticmethod
    def merge_sections(
        original_docstring: ParsedDocstring | None,
        new_sections: dict[str, list[str]],
        preserve_original: bool = True,
    ) -> list[str]:
        """Merge new sections with original docstring content."""
        if not original_docstring or not preserve_original:
            # Return new content only
            result = []
            for _, lines in new_sections.items():
                result.extend(lines)
                result.append("")  # Blank line between sections
            return result[:-1] if result else []  # Remove trailing blank line

        # Complex merging logic would go here
        # For now, return new sections
        return TemplateMerger.merge_sections(None, new_sections, False)

    @staticmethod
    def preserve_custom_sections(
        original_lines: list[str], standard_sections: set[str]
    ) -> list[str]:
        """Extract custom sections not in standard set."""
        custom_lines = []
        in_custom_section = False

        for line in original_lines:
            line_stripped = line.strip()

            # Check if this is a standard section header
            is_standard = any(
                line_stripped.startswith(section) for section in standard_sections
            )

            if is_standard:
                in_custom_section = False
            elif (
                line_stripped and line_stripped.endswith(":") and not in_custom_section
            ):
                # Potential custom section
                in_custom_section = True
                custom_lines.append(line)
            elif in_custom_section:
                custom_lines.append(line)

        return custom_lines
