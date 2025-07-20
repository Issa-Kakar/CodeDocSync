"""
Docstring style detection and validation.

This module provides automatic detection of docstring styles from existing
code and validates that suggestions match the detected or configured style.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

from .config import SuggestionConfig
from .models import StyleDetectionError


class DocstringStyleDetector:
    """Detect docstring style from existing code."""

    def __init__(self, config: SuggestionConfig | None = None):
        """Initialize detector with configuration."""
        self.config = config or SuggestionConfig()
        self._detection_cache: dict[str, str] = {}

        # Style indicators and their weights
        self._style_indicators = {
            "google": {
                "required": [r"Args?:", r"Returns?:", r"Raises?:"],
                "optional": [r"Yields?:", r"Note:", r"Example:"],
                "forbidden": [
                    r"Parameters\s*\n\s*-{3,}",
                    r":param\s+\w+:",
                    r"^\s*\.\.\s+",
                ],
                "weight": 1.0,
            },
            "numpy": {
                "required": [r"Parameters\s*\n\s*-{3,}", r"Returns\s*\n\s*-{3,}"],
                "optional": [
                    r"Yields\s*\n\s*-{3,}",
                    r"Raises\s*\n\s*-{3,}",
                    r"Notes\s*\n\s*-{3,}",
                    r"Examples\s*\n\s*-{3,}",
                ],
                "forbidden": [r"Args?:", r":param\s+\w+:", r"^\s*\.\.\s+"],
                "weight": 1.0,
            },
            "sphinx": {
                "required": [r":param\s+\w+:", r":returns?:", r":rtype:"],
                "optional": [r":type\s+\w+:", r":raises?\s+\w+:", r":yields?:"],
                "forbidden": [r"Args?:", r"Parameters\s*\n\s*-{3,}", r"^\s*\.\.\s+"],
                "weight": 1.0,
            },
            "rest": {
                "required": [r"^\s*\.\.\s+", r"^\s*\*\s+", r"^\s*\d+\.\s+"],
                "optional": [
                    r"^\s*::\s*$",
                    r"^\s*\.\.\s+note::",
                    r"^\s*\.\.\s+warning::",
                ],
                "forbidden": [r"Args?:", r"Parameters\s*\n\s*-{3,}", r":param\s+\w+:"],
                "weight": 0.8,  # Less specific indicators
            },
        }

    def detect_from_file(self, file_path: str) -> str:
        """Detect predominant style in a file."""
        cache_key = f"file:{file_path}"
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (FileNotFoundError, UnicodeDecodeError) as e:
            raise StyleDetectionError(
                f"Could not read file {file_path}: {e}",
                fallback_style=self.config.default_style,
            ) from e

        # Extract all docstrings from the file
        docstrings = self._extract_docstrings_from_code(content)

        if not docstrings:
            # No docstrings found, use default
            detected_style = self.config.default_style
        else:
            # Analyze all docstrings and find predominant style
            detected_style = self._analyze_multiple_docstrings(docstrings)

        self._detection_cache[cache_key] = detected_style
        return detected_style

    def detect_from_docstring(self, docstring: str) -> str:
        """Detect style from a single docstring."""
        cache_key = f"docstring:{hash(docstring)}"
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]

        if not docstring.strip():
            detected_style = self.config.default_style
        else:
            detected_style = self._analyze_single_docstring(docstring)

        self._detection_cache[cache_key] = detected_style
        return detected_style

    def detect_from_project(self, project_path: str, sample_size: int = 50) -> str:
        """Detect predominant style across a project."""
        cache_key = f"project:{project_path}:{sample_size}"
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]

        python_files = list(Path(project_path).rglob("*.py"))

        if not python_files:
            detected_style = self.config.default_style
        else:
            # Sample files to avoid processing huge projects
            if len(python_files) > sample_size:
                import random

                python_files = random.sample(python_files, sample_size)

            all_docstrings = []
            for file_path in python_files:
                try:
                    docstrings = self._extract_docstrings_from_file(str(file_path))
                    all_docstrings.extend(docstrings)
                except Exception:
                    continue  # Skip files that can't be processed

            detected_style = (
                self._analyze_multiple_docstrings(all_docstrings)
                if all_docstrings
                else self.config.default_style
            )

        self._detection_cache[cache_key] = detected_style
        return detected_style

    def _extract_docstrings_from_code(self, code: str) -> list[str]:
        """Extract all docstrings from Python code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        docstrings = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append(docstring)

        # Also check module docstring
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                if isinstance(tree.body[0].value.value, str):
                    docstrings.append(tree.body[0].value.value)

        return docstrings

    def _extract_docstrings_from_file(self, file_path: str) -> list[str]:
        """Extract docstrings from a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return self._extract_docstrings_from_code(content)
        except Exception:
            return []

    def _analyze_single_docstring(self, docstring: str) -> str:
        """Analyze a single docstring to determine its style."""
        scores = {}

        for style, indicators in self._style_indicators.items():
            score = self._calculate_style_score(docstring, indicators)
            scores[style] = score

        # Find the style with the highest score
        if not scores or max(scores.values()) == 0:
            return self.config.default_style

        best_style = max(scores, key=scores.get)

        # Require minimum confidence for detection
        if scores[best_style] < 0.3:
            return self.config.default_style

        return best_style

    def _analyze_multiple_docstrings(self, docstrings: list[str]) -> str:
        """Analyze multiple docstrings to find predominant style."""
        if not docstrings:
            return self.config.default_style

        style_votes = defaultdict(float)
        total_confidence = 0.0

        for docstring in docstrings:
            style_scores = {}
            for style, indicators in self._style_indicators.items():
                score = self._calculate_style_score(docstring, indicators)
                style_scores[style] = score

            # Vote for the best style from this docstring
            if style_scores and max(style_scores.values()) > 0:
                best_style = max(style_scores, key=style_scores.get)
                confidence = style_scores[best_style]
                style_votes[best_style] += confidence
                total_confidence += confidence

        if not style_votes or total_confidence == 0:
            return self.config.default_style

        # Find style with most votes
        winning_style = max(style_votes, key=style_votes.get)

        # Check if there's clear winner (>40% of total confidence)
        if style_votes[winning_style] / total_confidence < 0.4:
            return self.config.default_style

        return winning_style

    def _calculate_style_score(
        self, docstring: str, indicators: dict[str, list[str]]
    ) -> float:
        """Calculate how well a docstring matches a style's indicators."""
        score = 0.0
        text = docstring.lower()

        # Required indicators (high weight)
        required_matches = 0
        for pattern in indicators["required"]:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                required_matches += 1

        # Bonus for having required indicators
        if required_matches > 0:
            score += required_matches * 2.0

        # Optional indicators (medium weight)
        optional_matches = 0
        for pattern in indicators["optional"]:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                optional_matches += 1

        score += optional_matches * 0.5

        # Penalty for forbidden indicators
        forbidden_matches = 0
        for pattern in indicators["forbidden"]:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                forbidden_matches += 1

        score -= forbidden_matches * 1.5

        # Apply style weight and normalize
        score *= indicators["weight"]

        # Normalize to 0-1 range
        return max(0.0, min(1.0, score / 5.0))

    def validate_style_consistency(
        self, text: str, expected_style: str
    ) -> tuple[bool, list[str]]:
        """Validate that text matches the expected style."""
        issues = []

        if expected_style not in self._style_indicators:
            return False, [f"Unknown style: {expected_style}"]

        indicators = self._style_indicators[expected_style]
        text_lower = text.lower()

        # Check for forbidden patterns
        for pattern in indicators["forbidden"]:
            if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                issues.append(f"Found {expected_style}-incompatible pattern: {pattern}")

        # Check style-specific formatting rules
        if expected_style == "google":
            issues.extend(self._validate_google_style(text))
        elif expected_style == "numpy":
            issues.extend(self._validate_numpy_style(text))
        elif expected_style == "sphinx":
            issues.extend(self._validate_sphinx_style(text))
        elif expected_style == "rest":
            issues.extend(self._validate_rest_style(text))

        return len(issues) == 0, issues

    def _validate_google_style(self, text: str) -> list[str]:
        """Validate Google-style specific formatting."""
        issues = []
        lines = text.split("\n")

        # Check section headers end with colon
        google_sections = [
            "Args",
            "Arguments",
            "Returns",
            "Return",
            "Yields",
            "Raises",
            "Note",
            "Example",
        ]
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for section in google_sections:
                if line_stripped == section:  # Section without colon
                    issues.append(
                        f"Google style section '{section}' should end with colon"
                    )
                elif line_stripped == f"{section}:":
                    # Check next non-empty line is indented
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            if not lines[j].startswith("    "):
                                issues.append(
                                    f"Content after '{section}:' should be indented 4 spaces"
                                )
                            break

        return issues

    def _validate_numpy_style(self, text: str) -> list[str]:
        """Validate NumPy-style specific formatting."""
        issues = []
        lines = text.split("\n")

        # Check section headers have proper underlines
        numpy_sections = [
            "Parameters",
            "Returns",
            "Yields",
            "Raises",
            "See Also",
            "Notes",
            "Examples",
        ]
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped in numpy_sections:
                # Next line should be underline with dashes
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not next_line.startswith("-") or len(next_line) < 3:
                        issues.append(
                            f"NumPy section '{line_stripped}' should be followed by dashes"
                        )
                    elif len(next_line) != len(line_stripped):
                        issues.append(
                            "NumPy section underline should match header length"
                        )

        return issues

    def _validate_sphinx_style(self, text: str) -> list[str]:
        """Validate Sphinx-style specific formatting."""
        issues = []

        # Check field list formatting
        sphinx_patterns = [
            (r":param\s+(\w+)\s*:", "param"),
            (r":type\s+(\w+)\s*:", "type"),
            (r":returns?\s*:", "returns"),
            (r":rtype\s*:", "rtype"),
            (r":raises?\s+(\w+)\s*:", "raises"),
        ]

        for pattern, field_type in sphinx_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Basic validation - field should be at line start or properly indented
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_prefix = text[line_start : match.start()]
                if line_prefix and not line_prefix.isspace():
                    issues.append(
                        f"Sphinx field '{field_type}' should start at beginning of line or be properly indented"
                    )

        return issues

    def _validate_rest_style(self, text: str) -> list[str]:
        """Validate reStructuredText style specific formatting."""
        issues = []

        # Check directive formatting
        directive_pattern = r"^\s*\.\.\s+(\w+)::\s*(.*)$"
        lines = text.split("\n")

        for i, line in enumerate(lines):
            match = re.match(directive_pattern, line)
            if match:
                directive_name = match.group(1)
                # Check if content is properly indented on following lines
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        if not lines[j].startswith("   "):  # At least 3 spaces
                            issues.append(
                                f"Content after '{directive_name}::' directive should be indented"
                            )
                        break

        return issues

    def get_style_confidence(self, text: str, style: str) -> float:
        """Get confidence score for a specific style match."""
        if style not in self._style_indicators:
            return 0.0

        indicators = self._style_indicators[style]
        return self._calculate_style_score(text, indicators)

    def get_all_style_scores(self, text: str) -> dict[str, float]:
        """Get confidence scores for all supported styles."""
        scores = {}
        for style in self._style_indicators:
            scores[style] = self.get_style_confidence(text, style)
        return scores

    def clear_cache(self) -> None:
        """Clear the detection cache."""
        self._detection_cache.clear()

    def get_style_template(self, style: str) -> dict[str, str]:
        """Get formatting templates for a specific style."""
        templates = {
            "google": {
                "section_header": "{section}:",
                "parameter": "    {name} ({type}): {description}",
                "parameter_no_type": "    {name}: {description}",
                "return": "    {type}: {description}",
                "return_no_type": "    {description}",
                "raises": "    {exception}: {description}",
            },
            "numpy": {
                "section_header": "{section}\n{underline}",
                "parameter": "{name} : {type}\n    {description}",
                "parameter_no_type": "{name}\n    {description}",
                "return": "{type}\n    {description}",
                "return_no_type": "{description}",
                "raises": "{exception}\n    {description}",
            },
            "sphinx": {
                "section_header": "",
                "parameter": ":param {name}: {description}",
                "parameter_with_type": ":param {name}: {description}\n:type {name}: {type}",
                "return": ":returns: {description}",
                "return_with_type": ":returns: {description}\n:rtype: {type}",
                "raises": ":raises {exception}: {description}",
            },
            "rest": {
                "section_header": ".. {section}::",
                "parameter": "   {name} ({type}): {description}",
                "parameter_no_type": "   {name}: {description}",
                "return": "   {type}: {description}",
                "return_no_type": "   {description}",
                "raises": "   {exception}: {description}",
            },
        }

        return templates.get(style, templates["google"])


# Global detector instance
style_detector = DocstringStyleDetector()
