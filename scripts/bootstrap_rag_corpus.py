#!/usr/bin/env python3
"""
Bootstrap RAG corpus by extracting high-quality docstring examples from CodeDocSync.

This script analyzes CodeDocSync's own codebase to build an initial corpus of
well-documented functions that can be used as examples for generating suggestions.
"""

import json
import logging

# Add parent directory to path for imports
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codedocsync.parser import (
    DocstringParser,
    ParsedDocstring,
    ParsedFunction,
    parse_python_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DocstringExample:
    """Represents a high-quality docstring example for the corpus."""

    function_name: str
    module_path: str
    function_signature: str
    docstring_format: str
    docstring_content: str
    has_params: bool
    has_returns: bool
    has_examples: bool
    complexity_score: int  # 1-5, based on function complexity
    quality_score: int  # 1-5, based on docstring completeness


def calculate_complexity_score(function: ParsedFunction) -> int:
    """Calculate complexity score based on function characteristics."""
    score = 1

    # Add points for parameters
    param_count = len(function.signature.parameters)
    if param_count >= 3:
        score += 1
    if param_count >= 5:
        score += 1

    # Add points for type annotations
    if all(p.type_annotation for p in function.signature.parameters):
        score += 1

    # Add point for return type
    if function.signature.return_type:
        score += 1

    return min(score, 5)


def calculate_quality_score(docstring: ParsedDocstring) -> int:
    """Calculate quality score based on docstring completeness."""
    score = 1

    # Add points for different sections
    if docstring.description and len(docstring.description) > 20:
        score += 1

    if docstring.parameters:
        score += 1
        # Extra point for all params having descriptions
        if all(p.description for p in docstring.parameters):
            score += 1

    if docstring.returns and docstring.returns.description:
        score += 1

    if hasattr(docstring, "examples") and docstring.examples:
        score += 1

    return min(score, 5)


def extract_examples_from_file(file_path: Path) -> list[DocstringExample]:
    """Extract high-quality docstring examples from a Python file."""
    examples = []

    try:
        docstring_parser = DocstringParser()

        functions = parse_python_file(str(file_path))

        for func in functions:
            # Skip functions without docstrings
            if not func.docstring or not func.docstring.raw_text:
                continue

            # Parse the docstring
            parsed_doc = docstring_parser.parse(func.docstring.raw_text)
            if not parsed_doc:
                continue

            # Calculate scores
            complexity = calculate_complexity_score(func)
            quality = calculate_quality_score(parsed_doc)

            # Only include high-quality examples (quality >= 3)
            if quality >= 3:
                example = DocstringExample(
                    function_name=func.signature.name,
                    module_path=str(file_path.relative_to(Path.cwd())),
                    function_signature=str(func.signature),
                    docstring_format=(
                        parsed_doc.format.value
                        if hasattr(parsed_doc.format, "value")
                        else str(parsed_doc.format)
                    ),
                    docstring_content=func.docstring.raw_text,
                    has_params=bool(parsed_doc.parameters),
                    has_returns=bool(parsed_doc.returns),
                    has_examples=bool(
                        hasattr(parsed_doc, "examples") and parsed_doc.examples
                    ),
                    complexity_score=complexity,
                    quality_score=quality,
                )
                examples.append(example)

    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")

    return examples


def main():
    """Main function to bootstrap the RAG corpus."""
    # Paths
    project_root = Path(__file__).parent.parent
    codedocsync_dir = project_root / "codedocsync"
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # Collect all Python files
    python_files = list(codedocsync_dir.rglob("*.py"))
    logger.info(f"Found {len(python_files)} Python files to analyze")

    # Extract examples
    all_examples = []
    for file_path in python_files:
        # Skip test files and __pycache__
        if "test" in file_path.parts or "__pycache__" in str(file_path):
            continue

        examples = extract_examples_from_file(file_path)
        if examples:
            all_examples.extend(examples)
            logger.info(f"Extracted {len(examples)} examples from {file_path.name}")

    # Sort by quality score (descending)
    all_examples.sort(key=lambda x: (x.quality_score, x.complexity_score), reverse=True)

    # Take top 150 examples
    corpus_examples = all_examples[:150]

    logger.info("\nCorpus Statistics:")
    logger.info(f"Total examples collected: {len(all_examples)}")
    logger.info(f"Examples in corpus: {len(corpus_examples)}")

    # Quality distribution
    quality_dist: dict[int, int] = {}
    for ex in corpus_examples:
        quality_dist[ex.quality_score] = quality_dist.get(ex.quality_score, 0) + 1
    logger.info(f"Quality distribution: {quality_dist}")

    # Format distribution
    format_dist: dict[str, int] = {}
    for ex in corpus_examples:
        format_dist[ex.docstring_format] = format_dist.get(ex.docstring_format, 0) + 1
    logger.info(f"Format distribution: {format_dist}")

    # Save to JSON
    output_file = data_dir / "bootstrap_corpus.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "1.0",
                "examples": [asdict(ex) for ex in corpus_examples],
                "statistics": {
                    "total_collected": len(all_examples),
                    "corpus_size": len(corpus_examples),
                    "quality_distribution": quality_dist,
                    "format_distribution": format_dist,
                },
            },
            f,
            indent=2,
        )

    logger.info(f"\nBootstrap corpus saved to: {output_file}")

    # Also create empty curated examples file
    curated_file = data_dir / "curated_examples.json"
    if not curated_file.exists():
        with open(curated_file, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "examples": []}, f, indent=2)
        logger.info(f"Created empty curated examples file: {curated_file}")


if __name__ == "__main__":
    main()
