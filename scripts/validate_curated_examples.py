#!/usr/bin/env python3
"""Validate curated examples JSON structure and content."""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def validate_example_structure(example: dict[str, Any], index: int) -> list[str]:
    """Validate a single example's structure and return any errors."""
    errors = []

    # Required fields
    required_fields = {
        "function_name",
        "module_path",
        "function_signature",
        "docstring_format",
        "docstring_content",
        "has_params",
        "has_returns",
        "has_examples",
        "complexity_score",
        "quality_score",
        "source",
    }

    # Check for missing fields
    missing_fields = required_fields - set(example.keys())
    if missing_fields:
        errors.append(f"Example {index}: Missing fields: {missing_fields}")

    # Validate field types
    if "function_name" in example and not isinstance(example["function_name"], str):
        errors.append(f"Example {index}: function_name must be string")

    if "module_path" in example and not isinstance(example["module_path"], str):
        errors.append(f"Example {index}: module_path must be string")

    if "function_signature" in example and not isinstance(
        example["function_signature"], str
    ):
        errors.append(f"Example {index}: function_signature must be string")

    if "docstring_format" in example:
        if example["docstring_format"] not in ["google", "numpy", "sphinx"]:
            errors.append(
                f"Example {index}: Invalid docstring_format: {example['docstring_format']}"
            )

    if "complexity_score" in example:
        if (
            not isinstance(example["complexity_score"], int)
            or not 1 <= example["complexity_score"] <= 5
        ):
            errors.append(f"Example {index}: complexity_score must be int between 1-5")

    if "quality_score" in example:
        if (
            not isinstance(example["quality_score"], int)
            or not 1 <= example["quality_score"] <= 5
        ):
            errors.append(f"Example {index}: quality_score must be int between 1-5")

    # Curated examples should have quality_score = 5
    if example.get("source") == "curated" and example.get("quality_score") != 5:
        errors.append(f"Example {index}: Curated example must have quality_score=5")

    # Validate boolean fields
    for field in ["has_params", "has_returns", "has_examples"]:
        if field in example and not isinstance(example[field], bool):
            errors.append(f"Example {index}: {field} must be boolean")

    # Content validation
    if "docstring_content" in example:
        content = example["docstring_content"]
        if not content or not content.strip():
            errors.append(f"Example {index}: Empty docstring_content")

        # Check consistency with flags
        content_lower = content.lower()

        if example.get("has_params", False):
            # Different formats use different keywords
            if example.get("docstring_format") == "numpy":
                param_keywords = ["parameters", "params", "arguments"]
            else:
                param_keywords = ["args:", "parameters:", "params:", "arguments:"]
            if not any(kw in content_lower for kw in param_keywords):
                errors.append(
                    f"Example {index}: has_params=True but no parameter section found"
                )

        if example.get("has_returns", False):
            # Different formats use different keywords
            if example.get("docstring_format") == "numpy":
                return_keywords = ["returns", "return", "yields", "yield"]
            else:
                return_keywords = ["returns:", "return:", "yields:", "yield:"]
            if not any(kw in content_lower for kw in return_keywords):
                errors.append(
                    f"Example {index}: has_returns=True but no return section found"
                )

        if example.get("has_examples", False):
            example_keywords = ["example:", "examples:", ">>>", ".. code-block::"]
            if not any(kw in content_lower for kw in example_keywords):
                errors.append(
                    f"Example {index}: has_examples=True but no example section found"
                )

    return errors


def validate_curated_examples(
    file_path: Path,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate the curated examples file and return status, errors, and stats."""
    errors = []
    stats: dict[str, Any] = {
        "total_examples": 0,
        "curated_count": 0,
        "categories": Counter(),
        "formats": Counter(),
        "complexity_scores": Counter(),
        "quality_scores": Counter(),
        "duplicates": [],
    }

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], stats
    except Exception as e:
        return False, [f"Error reading file: {e}"], stats

    if "examples" not in data:
        errors.append("Missing 'examples' key in JSON")
        return False, errors, stats

    if not isinstance(data["examples"], list):
        errors.append("'examples' must be a list")
        return False, errors, stats

    examples = data["examples"]
    stats["total_examples"] = len(examples)

    # Track function names for duplicate detection
    function_names = []

    for i, example in enumerate(examples):
        # Validate structure
        example_errors = validate_example_structure(example, i)
        errors.extend(example_errors)

        # Collect stats
        if example.get("source") == "curated":
            stats["curated_count"] += 1

        if "category" in example:
            stats["categories"][example["category"]] += 1

        if "docstring_format" in example:
            stats["formats"][example["docstring_format"]] += 1

        if "complexity_score" in example:
            stats["complexity_scores"][example["complexity_score"]] += 1

        if "quality_score" in example:
            stats["quality_scores"][example["quality_score"]] += 1

        # Check for duplicates
        if "function_name" in example:
            func_name = example["function_name"]
            if func_name in function_names:
                stats["duplicates"].append(func_name)
            function_names.append(func_name)

    # Additional validation
    if stats["curated_count"] < 75:
        errors.append(
            f"Expected at least 75 curated examples, found {stats['curated_count']}"
        )

    # Check category distribution
    expected_categories = {
        "async_patterns": 20,
        "rest_api_patterns": 15,
        "data_science_patterns": 10,
        "advanced_patterns": 10,
    }

    for category, min_count in expected_categories.items():
        actual_count = stats["categories"].get(category, 0)
        if actual_count < min_count:
            errors.append(
                f"Category '{category}' has {actual_count} examples, expected at least {min_count}"
            )

    # Check for duplicates
    if stats["duplicates"]:
        unique_duplicates = list(set(stats["duplicates"]))
        errors.append(f"Duplicate function names found: {unique_duplicates}")

    return len(errors) == 0, errors, stats


def print_validation_report(
    is_valid: bool, errors: list[str], stats: dict[str, Any]
) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("CURATED EXAMPLES VALIDATION REPORT")
    print("=" * 60)

    print(f"\nStatus: {'VALID' if is_valid else 'INVALID'}")
    print(f"Total Examples: {stats['total_examples']}")
    print(f"Curated Examples: {stats['curated_count']}")

    if stats["categories"]:
        print("\nCategory Distribution:")
        for category, count in sorted(stats["categories"].items()):
            print(f"  - {category}: {count}")

    print("\nDocstring Format Distribution:")
    for format_name, count in sorted(stats["formats"].items()):
        print(f"  - {format_name}: {count}")

    print("\nComplexity Score Distribution:")
    for score in sorted(stats["complexity_scores"].keys()):
        count = stats["complexity_scores"][score]
        print(f"  - Score {score}: {count}")

    print("\nQuality Score Distribution:")
    for score in sorted(stats["quality_scores"].keys()):
        count = stats["quality_scores"][score]
        print(f"  - Score {score}: {count}")

    if errors:
        print(f"\nERROR: Found {len(errors)} validation errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\nSUCCESS: No validation errors found!")

    print("\n" + "=" * 60)


def main():
    """Main validation function."""
    # Find the curated examples file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    curated_file = project_root / "data" / "curated_examples.json"

    if not curated_file.exists():
        print(f"ERROR: Curated examples file not found at {curated_file}")
        sys.exit(1)

    print(f"Validating: {curated_file}")

    # Run validation
    is_valid, errors, stats = validate_curated_examples(curated_file)

    # Print report
    print_validation_report(is_valid, errors, stats)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
