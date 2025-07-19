"""
Utility functions for rule engine implementation.

This file contains shared utilities for rule implementation including type parsing,
parameter extraction, and suggestion generation.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from codedocsync.parser import (
    ParsedFunction,
    FunctionParameter,
    ParsedDocstring,
    DocstringParameter,
)


# TYPE PARSING UTILITIES


def normalize_type_string(type_str: str) -> str:
    """
    Convert type string variations to canonical form.

    Args:
        type_str: Type string to normalize

    Returns:
        Normalized type string
    """
    if not type_str:
        return ""

    # Remove whitespace and normalize case
    normalized = type_str.strip()

    # Common type mappings
    type_mappings = {
        "string": "str",
        "integer": "int",
        "boolean": "bool",
        "float": "float",
        "list": "List",
        "dict": "Dict",
        "tuple": "Tuple",
        "set": "Set",
    }

    # Apply mappings (case insensitive)
    for old, new in type_mappings.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized, flags=re.IGNORECASE)

    # Handle union types (Python 3.10+ style vs typing style)
    normalized = re.sub(r"\bUnion\[([^,]+),\s*None\]", r"Optional[\1]", normalized)
    normalized = re.sub(r"\b(\w+)\s*\|\s*None\b", r"Optional[\1]", normalized)

    # Normalize List/list variations
    normalized = re.sub(r"\blist\[", "List[", normalized)
    normalized = re.sub(r"\bdict\[", "Dict[", normalized)
    normalized = re.sub(r"\btuple\[", "Tuple[", normalized)
    normalized = re.sub(r"\bset\[", "Set[", normalized)

    return normalized


def compare_types(type1: str, type2: str) -> bool:
    """
    Check if two type strings are equivalent.

    Args:
        type1: First type string
        type2: Second type string

    Returns:
        True if types are equivalent
    """
    if not type1 or not type2:
        return type1 == type2

    # Normalize both types
    norm1 = normalize_type_string(type1)
    norm2 = normalize_type_string(type2)

    # Direct comparison
    if norm1 == norm2:
        return True

    # Check for equivalent forms
    equivalents = [
        (norm1.lower(), norm2.lower()),
        # Handle Optional variations
        (norm1.replace("Optional[", "").replace("]", "") + " | None", norm2),
        (norm1, norm2.replace("Optional[", "").replace("]", "") + " | None"),
    ]

    return any(eq1 == eq2 for eq1, eq2 in equivalents)


def extract_base_type(type_str: str) -> str:
    """
    Get base type from Optional, List, etc.

    Args:
        type_str: Type string to extract from

    Returns:
        Base type string
    """
    if not type_str:
        return ""

    normalized = normalize_type_string(type_str)

    # Extract from Optional[T]
    optional_match = re.match(r"Optional\[(.+)\]", normalized)
    if optional_match:
        return optional_match.group(1)

    # Extract from List[T], Dict[K,V] etc
    generic_match = re.match(r"(\w+)\[(.+)\]", normalized)
    if generic_match:
        container = generic_match.group(1)
        inner = generic_match.group(2)

        # For List[T], return T
        if container in ["List", "Set", "Tuple"]:
            return inner.split(",")[0].strip()
        # For Dict[K,V], return K,V
        elif container == "Dict":
            return inner

    return normalized


# PARAMETER EXTRACTION HELPERS


def extract_function_params(function: ParsedFunction) -> List[FunctionParameter]:
    """
    Get parameters from FunctionSignature, excluding self/cls.

    Args:
        function: Parsed function to extract from

    Returns:
        List of function parameters (excluding self/cls)
    """
    params = []
    for param in function.signature.parameters:
        if not should_ignore_param(param.name):
            params.append(param)
    return params


def extract_doc_params(
    docstring: Optional[ParsedDocstring],
) -> List[DocstringParameter]:
    """
    Get parameters from ParsedDocstring.

    Args:
        docstring: Parsed docstring to extract from

    Returns:
        List of documented parameters
    """
    if not docstring:
        return []
    return docstring.parameters


def should_ignore_param(param_name: str) -> bool:
    """
    Check if parameter should be skipped in analysis.

    Args:
        param_name: Parameter name to check

    Returns:
        True if parameter should be ignored
    """
    # Ignore self and cls parameters
    return param_name in ("self", "cls")


def get_param_mapping(
    func_params: List[FunctionParameter], doc_params: List[DocstringParameter]
) -> Tuple[Dict[str, FunctionParameter], Dict[str, DocstringParameter]]:
    """
    Create name-to-parameter mappings for comparison.

    Args:
        func_params: Function parameters
        doc_params: Documentation parameters

    Returns:
        Tuple of (func_mapping, doc_mapping)
    """
    func_mapping = {p.name.lstrip("*"): p for p in func_params}
    doc_mapping = {p.name.lstrip("*"): p for p in doc_params}
    return func_mapping, doc_mapping


# SUGGESTION GENERATORS


def generate_parameter_suggestion(
    issue_type: str,
    param_name: str,
    expected_value: str = "",
    actual_value: str = "",
    **kwargs,
) -> str:
    """
    Create fix suggestion for parameter issues.

    Args:
        issue_type: Type of issue (from ISSUE_TYPES)
        param_name: Parameter name
        expected_value: Expected/correct value
        actual_value: Current/incorrect value
        **kwargs: Additional context

    Returns:
        Formatted suggestion string
    """
    if issue_type == "parameter_name_mismatch":
        return (
            f"Update the docstring parameter name from '{actual_value}' to '{expected_value}':\n\n"
            f"Args:\n"
            f"    {expected_value}: [Keep existing description]  # Changed from '{actual_value}'"
        )

    elif issue_type == "parameter_type_mismatch":
        return (
            f"Update the docstring type for '{param_name}' to match function annotation:\n\n"
            f"Args:\n"
            f"    {param_name} ({expected_value}): [Keep existing description]  # Changed from '{actual_value}'"
        )

    elif issue_type == "missing_params":
        param_type = kwargs.get("param_type", "")
        type_hint = f" ({param_type})" if param_type else ""
        return (
            f"Add documentation for parameter '{param_name}':\n\n"
            f"Args:\n"
            f"    {param_name}{type_hint}: [TODO: Add description]"
        )

    elif issue_type == "default_mismatches":
        return (
            f"Update the default value for '{param_name}' in docstring:\n\n"
            f"Args:\n"
            f"    {param_name}: Description. Defaults to {expected_value}.  # Changed from {actual_value}"
        )

    elif issue_type == "parameter_order_different":
        correct_order = kwargs.get("correct_order", [])
        if correct_order:
            return (
                "Reorder docstring parameters to match function signature:\n\n"
                "Args:\n"
                + "\n".join(
                    f"    {param}: [Keep existing description]"
                    for param in correct_order
                )
            )

    return f"Fix the {issue_type} for parameter '{param_name}'"


def generate_docstring_template(
    missing_params: List[FunctionParameter],
    missing_return_type: Optional[str] = None,
    missing_raises: Optional[List[str]] = None,
) -> str:
    """
    Create template for missing documentation sections.

    Args:
        missing_params: Parameters that need documentation
        missing_return_type: Return type that needs documentation
        missing_raises: Exception types that need documentation

    Returns:
        Formatted docstring template
    """
    template_parts = []

    if missing_params:
        template_parts.append("Args:")
        for param in missing_params:
            param_name = param.name.lstrip("*")
            type_hint = f" ({param.type_annotation})" if param.type_annotation else ""
            default_hint = (
                f" Defaults to {param.default_value}." if param.default_value else ""
            )
            template_parts.append(
                f"    {param_name}{type_hint}: [TODO: Add description]{default_hint}"
            )

    if missing_return_type:
        template_parts.append("\nReturns:")
        template_parts.append(
            f"    {missing_return_type}: [TODO: Describe return value]"
        )

    if missing_raises:
        template_parts.append("\nRaises:")
        for exc_type in missing_raises:
            template_parts.append(
                f"    {exc_type}: [TODO: Describe when this is raised]"
            )

    return "\n".join(template_parts)


def format_code_suggestion(suggestion: str, language: str = "python") -> str:
    """
    Format suggestions as markdown code blocks.

    Args:
        suggestion: Raw suggestion text
        language: Programming language for syntax highlighting

    Returns:
        Markdown-formatted suggestion
    """
    if "```" in suggestion:
        # Already formatted
        return suggestion

    # Check if it contains code-like content
    if any(
        keyword in suggestion.lower()
        for keyword in ["args:", "returns:", "raises:", "def ", "class "]
    ):
        lines = suggestion.split("\n")
        formatted_lines = []

        for line in lines:
            if any(
                keyword in line.lower() for keyword in ["args:", "returns:", "raises:"]
            ):
                # This is a docstring section - format as markdown
                formatted_lines.append(line)
            elif line.strip().startswith(("def ", "class ", "import ", "from ")):
                # This is Python code - wrap in code block
                if not any("```" in prev_line for prev_line in formatted_lines[-3:]):
                    formatted_lines.append(f"```{language}")
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        # Close any open code blocks
        if "```" in "\n".join(formatted_lines) and not suggestion.rstrip().endswith(
            "```"
        ):
            formatted_lines.append("```")

        return "\n".join(formatted_lines)

    return suggestion


def get_parameter_statistics(
    func_params: List[FunctionParameter], doc_params: List[DocstringParameter]
) -> Dict[str, Any]:
    """
    Generate statistics about parameter documentation coverage.

    Args:
        func_params: Function parameters
        doc_params: Documentation parameters

    Returns:
        Dictionary with coverage statistics
    """
    func_names = {p.name.lstrip("*") for p in func_params}
    doc_names = {p.name.lstrip("*") for p in doc_params}

    total_params = len(func_params)
    documented_params = len(func_names & doc_names)
    missing_params = len(func_names - doc_names)
    extra_params = len(doc_names - func_names)

    coverage_percent = (
        (documented_params / total_params * 100) if total_params > 0 else 100
    )

    return {
        "total_function_params": total_params,
        "documented_params": documented_params,
        "missing_params": missing_params,
        "extra_params": extra_params,
        "coverage_percent": round(coverage_percent, 1),
        "is_complete": missing_params == 0 and extra_params == 0,
    }


def validate_special_parameters(
    func_params: List[FunctionParameter],
) -> List[Dict[str, Any]]:
    """
    Validate special parameter patterns (*args, **kwargs, Optional types).

    Args:
        func_params: Function parameters to validate

    Returns:
        List of validation issues found
    """
    issues = []

    for param in func_params:
        # Check for *args/**kwargs without documentation
        if param.name.startswith("*"):
            if not param.name.startswith("**") and param.name != "*":
                issues.append(
                    {
                        "type": "undocumented_args",
                        "param": param.name,
                        "suggestion": f"Document {param.name} parameter for variable arguments",
                    }
                )
            elif param.name.startswith("**"):
                issues.append(
                    {
                        "type": "undocumented_kwargs",
                        "param": param.name,
                        "suggestion": f"Document {param.name} parameter for keyword arguments",
                    }
                )

        # Check for Optional type consistency
        if param.default_value and param.type_annotation:
            if (
                param.default_value.lower() in ("none", "null")
                and "optional" not in param.type_annotation.lower()
            ):
                issues.append(
                    {
                        "type": "missing_optional",
                        "param": param.name,
                        "suggestion": f"Parameter '{param.name}' has None default but type is not Optional",
                    }
                )

    return issues
