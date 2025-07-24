"""Utility functions for data processing."""

import os
from pathlib import Path


def validate_input(file_path: str) -> bool:
    """Check if input file is valid.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and is readable
    """
    path = Path(file_path)
    return path.exists() and path.is_file() and os.access(file_path, os.R_OK)


def format_output(data: dict) -> str:
    """Format data dictionary as string output.

    Args:
        data: Data to format

    Returns:
        Formatted string representation
    """
    lines = ["=== Processed Data ==="]

    # Format main data
    if "data" in data:
        lines.append(f"Data: {data['data']}")

    # Format metadata
    if "metadata" in data:
        lines.append("\nMetadata:")
        for key, value in data["metadata"].items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def clean_data(text: str) -> str:
    """Clean and normalize text data.

    Args:
        text: Raw text to clean

    Returns:
        str: Cleaned text  # Redundant type in return
    """
    # Remove extra whitespace
    cleaned = " ".join(text.split())

    # Remove special characters
    cleaned = cleaned.replace("\t", " ").replace("\r", "")

    return cleaned.strip()


def merge_configs(*configs: dict) -> dict:
    """Merge multiple configuration dictionaries.

    Args:
        configs: Variable number of config dicts  # MISSING: *args not properly documented

    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
