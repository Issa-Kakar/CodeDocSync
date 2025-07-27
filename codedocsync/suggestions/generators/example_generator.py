"""
Example generation system for creating usage examples in docstrings.

This module specializes in generating realistic, runnable examples for functions
based on their signatures, behavior, and common usage patterns.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from ...parser.ast_parser import FunctionParameter
from ..base import BaseSuggestionGenerator
from ..models import (
    DocstringStyle,
    Suggestion,
    SuggestionContext,
    SuggestionDiff,
    SuggestionMetadata,
    SuggestionType,
)
from ..templates.base import get_template

logger = logging.getLogger(__name__)

# DocstringExample not needed - examples are stored as strings


@dataclass
class ExampleTemplate:
    """Template for generating examples."""

    setup_code: list[str]
    function_call: str
    expected_output: str
    imports: list[str]
    description: str
    complexity: str  # 'basic', 'intermediate', 'advanced'


class ParameterValueGenerator:
    """Generate realistic parameter values for examples."""

    def __init__(self) -> None:
        self.type_defaults = {
            "str": '"example"',
            "int": "42",
            "float": "3.14",
            "bool": "True",
            "list": "[1, 2, 3]",
            "List": "[1, 2, 3]",
            "dict": '{"key": "value"}',
            "Dict": '{"key": "value"}',
            "tuple": "(1, 2, 3)",
            "Tuple": "(1, 2, 3)",
            "set": "{1, 2, 3}",
            "Set": "{1, 2, 3}",
            "None": "None",
            "Any": '"example"',
        }

        self.name_based_values = {
            "name": '"John Doe"',
            "filename": '"example.txt"',
            "path": '"/path/to/file"',
            "url": '"https://example.com"',
            "email": '"user@example.com"',
            "password": '"secure_password"',
            "id": "123",
            "count": "10",
            "size": "100",
            "length": "50",
            "width": "200",
            "height": "150",
            "age": "25",
            "score": "95",
            "index": "0",
            "key": '"api_key"',
            "token": '"abc123"',
            "data": "[1, 2, 3, 4, 5]",
            "items": '["a", "b", "c"]',
            "values": "[1, 2, 3]",
            "text": '"Hello, world!"',
            "message": '"Processing complete"',
            "config": '{"debug": True}',
            "options": '{"verbose": True}',
        }

    def generate_value(self, param: FunctionParameter) -> str:
        """Generate a realistic value for a parameter."""
        param_name = param.name.lower()

        # First try name-based generation
        for name_pattern, value in self.name_based_values.items():
            if name_pattern in param_name:
                return value

        # Then try type-based generation
        if param.type_annotation:
            type_str = self._normalize_type(param.type_annotation)
            for type_pattern, value in self.type_defaults.items():
                if type_pattern.lower() in type_str.lower():
                    return value

        # Handle optional parameters
        if not param.is_required and param.default_value:
            return param.default_value

        # Default fallback
        return '"example"'

    def _normalize_type(self, type_annotation: str) -> str:
        """Normalize type annotation for matching."""
        # Remove Optional, Union wrappers
        normalized = re.sub(r"Optional\[(.*?)\]", r"\1", type_annotation)
        normalized = re.sub(r"Union\[(.*?),\s*None\]", r"\1", normalized)
        normalized = re.sub(r"Union\[None,\s*(.*?)\]", r"\1", normalized)

        # Extract base type from generic types
        normalized = re.sub(r"List\[(.*?)\]", "List", normalized)
        normalized = re.sub(r"Dict\[(.*?)\]", "Dict", normalized)
        normalized = re.sub(r"Tuple\[(.*?)\]", "Tuple", normalized)
        normalized = re.sub(r"Set\[(.*?)\]", "Set", normalized)

        return normalized.strip()


class ExamplePatternAnalyzer:
    """Analyze function patterns to generate appropriate examples."""

    def analyze_function(self, function: Any, source_code: str = "") -> dict[str, Any]:
        """Analyze function to determine example characteristics."""
        analysis = {
            "is_property": False,
            "is_classmethod": False,
            "is_staticmethod": False,
            "is_async": False,
            "is_generator": False,
            "has_side_effects": False,
            "return_type": None,
            "complexity": "basic",
            "domain": "general",
        }

        # Analyze function signature
        if hasattr(function, "signature"):
            sig = function.signature

            # Check decorators
            if hasattr(sig, "decorators"):
                decorators = sig.decorators or []
                analysis["is_property"] = "property" in decorators
                analysis["is_classmethod"] = "classmethod" in decorators
                analysis["is_staticmethod"] = "staticmethod" in decorators

            # Get return type
            if hasattr(sig, "return_annotation"):
                analysis["return_type"] = sig.return_annotation

        # Analyze source code if available
        if source_code:
            try:
                tree = ast.parse(source_code)
                self._analyze_ast(tree, analysis)
            except SyntaxError:
                pass

        return analysis

    def _analyze_ast(self, tree: ast.AST, analysis: dict[str, Any]) -> None:
        """Analyze AST for function characteristics."""
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                analysis["is_async"] = True
            elif isinstance(node, ast.Yield | ast.YieldFrom):
                analysis["is_generator"] = True
            elif isinstance(node, ast.Call):
                self._analyze_function_call(node, analysis)

    def _analyze_function_call(self, node: ast.Call, analysis: dict[str, Any]) -> None:
        """Analyze function calls for side effects and domain."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check for side effects
        side_effect_functions = {
            "print",
            "open",
            "write",
            "save",
            "log",
            "debug",
            "info",
            "warning",
            "error",
        }
        if func_name in side_effect_functions:
            analysis["has_side_effects"] = True

        # Determine domain
        math_functions = {"sin", "cos", "tan", "sqrt", "abs", "max", "min", "sum"}
        file_functions = {"open", "read", "write", "close"}
        web_functions = {"get", "post", "request", "response"}

        if func_name in math_functions:
            analysis["domain"] = "math"
        elif func_name in file_functions:
            analysis["domain"] = "file"
        elif func_name in web_functions:
            analysis["domain"] = "web"


class ExampleGenerator:
    """Generate usage examples for functions."""

    def __init__(self) -> None:
        self.value_generator = ParameterValueGenerator()
        self.pattern_analyzer = ExamplePatternAnalyzer()

    def generate_examples(
        self, function: Any, source_code: str = "", count: int = 1
    ) -> list[ExampleTemplate]:
        """Generate usage examples for a function."""
        analysis = self.pattern_analyzer.analyze_function(function, source_code)
        examples = []

        # Generate basic example
        basic_example = self._generate_basic_example(function, analysis)
        if basic_example:
            examples.append(basic_example)

        # Generate additional examples if requested
        if count > 1:
            # Try to generate edge case example
            edge_example = self._generate_edge_case_example(function, analysis)
            if edge_example and len(examples) < count:
                examples.append(edge_example)

            # Try to generate advanced example
            if len(examples) < count:
                advanced_example = self._generate_advanced_example(function, analysis)
                if advanced_example:
                    examples.append(advanced_example)

        return examples[:count]

    def _generate_basic_example(
        self, function: Any, analysis: dict[str, Any]
    ) -> ExampleTemplate | None:
        """Generate a basic usage example."""
        if not hasattr(function, "signature"):
            return None

        sig = function.signature
        function_name = sig.name

        # Generate parameter values
        call_args = []
        setup_lines = []
        imports: list[str] = []

        # Handle different function types
        if analysis["is_classmethod"]:
            call_prefix = "ClassName."
        elif analysis["is_staticmethod"]:
            call_prefix = "ClassName."
        elif analysis["is_property"]:
            call_prefix = "instance."
        else:
            call_prefix = ""

        # Generate arguments
        for param in sig.parameters:
            if param.name in ("self", "cls"):
                continue

            if not param.is_required:
                # Skip optional parameters in basic example
                continue

            value = self.value_generator.generate_value(param)
            call_args.append(f"{param.name}={value}")

        # Build function call
        args_str = ", ".join(call_args)
        if analysis["is_property"]:
            function_call = f"{call_prefix}{function_name}"
        else:
            function_call = f"{call_prefix}{function_name}({args_str})"

        # Handle async functions
        if analysis["is_async"]:
            function_call = f"await {function_call}"
            setup_lines.append("# In an async context")

        # Generate expected output
        expected_output = self._generate_expected_output(analysis)

        # Add result assignment if function returns something
        if expected_output and expected_output != "None":
            full_call = f"result = {function_call}"
        else:
            full_call = function_call

        return ExampleTemplate(
            setup_code=setup_lines,
            function_call=full_call,
            expected_output=expected_output,
            imports=imports,
            description="Basic usage example",
            complexity="basic",
        )

    def _generate_edge_case_example(
        self, function: Any, analysis: dict[str, Any]
    ) -> ExampleTemplate | None:
        """Generate an edge case example."""
        if not hasattr(function, "signature"):
            return None

        sig = function.signature
        function_name = sig.name

        # Generate edge case parameter values
        call_args = []
        setup_lines = ["# Edge case with boundary values"]

        for param in sig.parameters:
            if param.name in ("self", "cls"):
                continue

            # Generate edge case values
            edge_value = self._generate_edge_case_value(param)
            call_args.append(f"{param.name}={edge_value}")

        args_str = ", ".join(call_args)
        function_call = f"{function_name}({args_str})"

        if analysis["is_async"]:
            function_call = f"await {function_call}"

        expected_output = self._generate_expected_output(analysis)

        return ExampleTemplate(
            setup_code=setup_lines,
            function_call=f"result = {function_call}",
            expected_output=expected_output,
            imports=[],
            description="Edge case example",
            complexity="intermediate",
        )

    def _generate_advanced_example(
        self, function: Any, analysis: dict[str, Any]
    ) -> ExampleTemplate | None:
        """Generate an advanced usage example."""
        if not hasattr(function, "signature"):
            return None

        sig = function.signature
        function_name = sig.name

        setup_lines = [
            "# Advanced usage with complex data",
            "data = {'key1': [1, 2, 3], 'key2': {'nested': True}}",
        ]

        # Use more complex parameter values
        call_args = []
        for param in sig.parameters:
            if param.name in ("self", "cls"):
                continue

            advanced_value = self._generate_advanced_value(param)
            call_args.append(f"{param.name}={advanced_value}")

        args_str = ", ".join(call_args)
        function_call = f"{function_name}({args_str})"

        if analysis["is_async"]:
            function_call = f"await {function_call}"

        expected_output = self._generate_expected_output(analysis)

        return ExampleTemplate(
            setup_code=setup_lines,
            function_call=f"result = {function_call}",
            expected_output=expected_output,
            imports=[],
            description="Advanced usage example",
            complexity="advanced",
        )

    def _generate_edge_case_value(self, param: FunctionParameter) -> str:
        """Generate edge case values for parameters."""
        param_name = param.name.lower()

        # Edge cases by name
        if "count" in param_name or "size" in param_name:
            return "0"  # Zero count/size
        elif "index" in param_name:
            return "-1"  # Negative index
        elif "list" in param_name or param_name.endswith("s"):
            return "[]"  # Empty list
        elif "dict" in param_name:
            return "{}"  # Empty dict
        elif "str" in param_name or "text" in param_name:
            return '""'  # Empty string

        # Edge cases by type
        if param.type_annotation:
            type_str = param.type_annotation.lower()
            if "int" in type_str:
                return "0"
            elif "float" in type_str:
                return "0.0"
            elif "str" in type_str:
                return '""'
            elif "list" in type_str:
                return "[]"
            elif "dict" in type_str:
                return "{}"

        return "None"

    def _generate_advanced_value(self, param: FunctionParameter) -> str:
        """Generate advanced/complex values for parameters."""
        param_name = param.name.lower()

        # Complex values by name
        if "config" in param_name:
            return '{"debug": True, "timeout": 30, "retries": 3}'
        elif "data" in param_name:
            return "data"  # Reference to complex data variable
        elif "options" in param_name:
            return '{"verbose": True, "format": "json"}'
        elif "callback" in param_name:
            return "lambda x: x * 2"

        # Use default complex values
        return self.value_generator.generate_value(param)

    def _generate_expected_output(self, analysis: dict[str, Any]) -> str:
        """Generate expected output description."""
        if analysis["return_type"]:
            return_type = analysis["return_type"].lower()

            if "none" in return_type:
                return "None"
            elif "bool" in return_type:
                return "True"
            elif "int" in return_type:
                return "42"
            elif "str" in return_type:
                return '"result"'
            elif "list" in return_type:
                return "[1, 2, 3]"
            elif "dict" in return_type:
                return '{"result": "value"}'

        if analysis["is_generator"]:
            return "# Generator object"

        return '"result"'


class ExampleSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for example-related issues."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the generator."""
        super().__init__(config)
        self._used_rag = False
        self._current_context: SuggestionContext | None = None

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate example documentation fixes."""
        self._current_context = context
        self._used_rag = False  # Reset for each generation
        issue = context.issue

        if issue.issue_type == "example_invalid":
            return self._fix_invalid_example(context)
        elif issue.issue_type == "missing_examples":
            return self._add_missing_examples(context)
        elif issue.issue_type == "example_outdated":
            return self._update_outdated_example(context)
        elif issue.issue_type == "example_incomplete":
            return self._complete_example(context)
        else:
            return self._generic_example_fix(context)

    def _fix_invalid_example(self, context: SuggestionContext) -> Suggestion:
        """Fix invalid example in docstring."""
        return self._generate_new_examples(context, "fix invalid example")

    def _add_missing_examples(self, context: SuggestionContext) -> Suggestion:
        """Add missing examples to docstring."""
        return self._generate_new_examples(context, "add missing examples")

    def _update_outdated_example(self, context: SuggestionContext) -> Suggestion:
        """Update outdated example in docstring."""
        return self._generate_new_examples(context, "update outdated example")

    def _complete_example(self, context: SuggestionContext) -> Suggestion:
        """Complete incomplete example in docstring."""
        return self._generate_new_examples(context, "complete example")

    def _generate_new_examples(
        self, context: SuggestionContext, action: str
    ) -> Suggestion:
        """Generate new examples for the function."""
        function = context.function

        # Try RAG-enhanced generation first
        rag_examples = self._generate_rag_enhanced_examples(function, context.issue)

        if rag_examples:
            logger.debug(f"Using RAG with {len(rag_examples)} examples")
            # Convert RAG examples to docstring format
            docstring_examples = rag_examples
        else:
            # Fallback to rule-based generation
            logger.debug("No RAG examples found, falling back to rule-based")
            generator = ExampleGenerator()
            source_code = getattr(function, "source_code", "")
            examples = generator.generate_examples(function, source_code, count=2)

            if not examples:
                return self._create_fallback_suggestion(
                    context, "Could not generate examples for this function"
                )

            # Convert to formatted example strings
            docstring_examples = []
            for example in examples:
                # Format example code
                example_code = self._format_example_code(example)
                # Combine description and code into a single example string
                if example.description:
                    example_str = f"{example.description}\n{example_code}"
                else:
                    example_str = example_code
                docstring_examples.append(example_str)

        # Update docstring with examples
        updated_docstring = self._add_examples_to_docstring(context, docstring_examples)

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Generated examples: {action}",
            confidence=0.7,  # Lower confidence for generated content
            suggestion_type=SuggestionType.EXAMPLE_UPDATE,
        )

        return suggestion

    def _format_example_code(self, example: ExampleTemplate) -> str:
        """Format example template into code string."""
        code_lines = []

        # Add imports if needed
        if example.imports:
            code_lines.extend(example.imports)
            code_lines.append("")

        # Add setup code
        if example.setup_code:
            code_lines.extend(example.setup_code)

        # Add function call
        code_lines.append(example.function_call)

        # Add expected output if it's meaningful
        if example.expected_output and example.expected_output != "None":
            code_lines.append(f"# Expected: {example.expected_output}")

        return "\n".join(code_lines)

    def _add_examples_to_docstring(
        self, context: SuggestionContext, examples: list[str]
    ) -> str:
        """Add examples to existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = DocstringStyle[style.upper()] if isinstance(style, str) else style
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        # Combine with existing examples
        existing_examples = getattr(docstring, "examples", []) if docstring else []
        all_examples = existing_examples + examples

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", "") if docstring else "",
            description=getattr(docstring, "description", None) if docstring else None,
            parameters=getattr(docstring, "parameters", []) if docstring else [],
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=all_examples,
        )

    def _detect_style(self, docstring: Any) -> str:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            # Return the string format directly
            return cast(str, docstring.format.value)

        return "google"  # Default fallback

    def _create_suggestion(
        self,
        context: SuggestionContext,
        suggested_text: str,
        description: str,
        confidence: float,
        suggestion_type: SuggestionType,
    ) -> Suggestion:
        """Create a suggestion object."""
        original_text = (
            getattr(context.docstring, "raw_text", "")
            if context.docstring
            else '"""TODO: Add docstring"""'
        )

        # Create diff
        original_lines = original_text.split("\n") if original_text else []
        suggested_lines = suggested_text.split("\n")

        diff = SuggestionDiff(
            original_lines=original_lines,
            suggested_lines=suggested_lines,
            start_line=getattr(context.function, "line_number", 1),
            end_line=getattr(context.function, "line_number", 1) + len(original_lines),
        )

        metadata = SuggestionMetadata(
            generator_type=self.__class__.__name__,
            generator_version="1.0.0",
            used_rag_examples=getattr(self, "_used_rag", False),
            # Store the description in rule_triggers for now
            rule_triggers=[description] if description else [],
        )

        return Suggestion(
            original_text=original_text,
            suggested_text=suggested_text,
            suggestion_type=suggestion_type,
            confidence=confidence,
            diff=diff,
            metadata=metadata,
            style=self._detect_style(context.docstring),
            copy_paste_ready=True,
        )

    def _create_fallback_suggestion(
        self, context: SuggestionContext, reason: str
    ) -> Suggestion:
        """Create a low-confidence fallback suggestion."""
        return self._create_suggestion(
            context,
            getattr(context.docstring, "raw_text", "") if context.docstring else "",
            f"Unable to generate specific example fix: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.EXAMPLE_UPDATE,
        )

    def _generic_example_fix(self, context: SuggestionContext) -> Suggestion:
        """Generic example fix for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown example issue type: {context.issue.issue_type}"
        )

    def _generate_rag_enhanced_examples(
        self, function: Any, issue: Any
    ) -> list[str] | None:
        """Generate examples using RAG corpus."""
        if not self._current_context or not self._current_context.related_functions:
            logger.debug("No RAG context or related functions available")
            return None

        logger.debug(
            f"RAG context available with {len(self._current_context.related_functions)} related functions"
        )

        # Extract and analyze patterns from related functions
        patterns = self._extract_example_patterns_from_corpus(
            self._current_context.related_functions
        )

        logger.debug(
            f"Extracted {len(patterns) if patterns else 0} patterns from corpus"
        )

        if not patterns:
            return None

        # Set self._used_rag = True only if patterns used
        self._used_rag = True

        # Determine complexity level based on issue type
        complexity_level = "basic"
        if issue.issue_type == "example_incomplete":
            complexity_level = "intermediate"
        elif "advanced" in issue.description.lower():
            complexity_level = "advanced"

        # Generate examples using patterns
        generated_examples: list[str] = []

        # Determine how many examples to generate based on issue type
        target_examples = 1
        if issue.issue_type == "example_incomplete":
            target_examples = 3  # Generate more examples for incomplete issues
        elif complexity_level != "basic":
            target_examples = 2

        # Generate examples at different complexity levels
        complexities = ["basic", "intermediate", "advanced"]
        for _i, complexity in enumerate(complexities):
            if len(generated_examples) >= target_examples:
                break

            example = self._synthesize_example_from_patterns(
                patterns, function, complexity
            )
            if example and example not in generated_examples:
                generated_examples.append(example)

        # Fallback to simple adaptation if we need more examples
        if len(generated_examples) < target_examples:
            logger.debug(
                f"Need {target_examples - len(generated_examples)} more examples, using simple adaptation"
            )
            for pattern in patterns[len(generated_examples) : target_examples]:
                if not pattern:
                    continue
                raw_text = pattern.get("raw_text", "")
                if raw_text:
                    adapted = self._adapt_example_code(raw_text, function)
                    if adapted and adapted not in generated_examples:
                        generated_examples.append(adapted)

        return generated_examples if generated_examples else None

    def _extract_example_patterns_from_corpus(
        self, related_functions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract example patterns from RAG corpus.

        Returns a list of patterns with metadata about their usage.
        """
        patterns = []

        for related in related_functions:
            if related.get("similarity", 0) <= 0.3:
                continue

            docstring_text = related.get("docstring", "")
            if not docstring_text:
                continue

            # Extract examples from this function
            examples = self._extract_examples_from_docstring(docstring_text)
            logger.debug(
                f"Extracted {len(examples)} examples from docstring with similarity {related.get('similarity', 0)}"
            )
            if not examples:
                continue

            # Parse each example to understand its structure
            for example in examples:
                pattern = self._parse_example_section(example)
                if pattern:
                    pattern["source_similarity"] = related.get("similarity", 0)
                    pattern["source_signature"] = related.get("signature", "")
                    patterns.append(pattern)

        # Sort by relevance
        patterns.sort(key=lambda p: p.get("relevance", 0), reverse=True)
        return patterns

    def _parse_example_section(self, example_text: str) -> dict[str, Any] | None:
        """Parse an example section to extract its components.

        Returns a dictionary with:
        - setup_code: List of setup lines
        - function_calls: List of function call lines
        - assertions: List of assertion/result lines
        - complexity: Estimated complexity level
        - features: List of features used (async, generators, etc.)
        """
        if not example_text.strip():
            return None

        lines = example_text.strip().split("\n")
        pattern: dict[str, Any] = {
            "setup_code": [],
            "function_calls": [],
            "assertions": [],
            "complexity": "basic",
            "features": [],
            "raw_text": example_text,
        }

        # Analyze each line
        for line in lines:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue

            # Detect features
            if "await" in clean_line:
                pattern["features"].append("async")
            if "yield" in clean_line:
                pattern["features"].append("generator")
            if "lambda" in clean_line:
                pattern["features"].append("lambda")

            # Categorize lines
            if any(keyword in clean_line for keyword in ["import", "from"]):
                pattern["setup_code"].append(line)
            elif (
                "=" in clean_line and "==" not in clean_line and "!=" not in clean_line
            ):
                # Assignment line - could be setup or result capture
                if "(" in clean_line and ")" in clean_line:
                    pattern["function_calls"].append(line)
                else:
                    pattern["setup_code"].append(line)
            elif clean_line.startswith(("assert", "print", "#")):
                pattern["assertions"].append(line)
            elif "(" in clean_line and ")" in clean_line:
                pattern["function_calls"].append(line)

        # Estimate complexity
        total_lines = (
            len(pattern["setup_code"])
            + len(pattern["function_calls"])
            + len(pattern["assertions"])
        )
        if total_lines > 10 or len(pattern["features"]) > 2:
            pattern["complexity"] = "advanced"
        elif total_lines > 5 or len(pattern["features"]) > 0:
            pattern["complexity"] = "intermediate"

        # Calculate relevance based on content
        pattern["relevance"] = self._calculate_example_relevance(pattern)

        return pattern

    def _calculate_example_relevance(self, pattern: dict[str, Any]) -> float:
        """Calculate relevance score for an example pattern.

        Factors considered:
        - Has function calls (required)
        - Has meaningful setup
        - Has assertions/results
        - Complexity matches target
        - Source similarity score
        """
        score = 0.0

        # Must have function calls
        if not pattern.get("function_calls"):
            return 0.0

        score += 0.3  # Base score for having function calls

        # Setup code adds value
        if pattern.get("setup_code"):
            score += 0.2

        # Assertions/results are valuable
        if pattern.get("assertions"):
            score += 0.2

        # Complexity scoring (prefer basic for most cases)
        complexity = pattern.get("complexity", "basic")
        if complexity == "basic":
            score += 0.1
        elif complexity == "intermediate":
            score += 0.05

        # Source similarity matters
        source_sim = pattern.get("source_similarity", 0)
        score += source_sim * 0.2

        return min(score, 1.0)

    def _synthesize_example_from_patterns(
        self,
        patterns: list[dict[str, Any]],
        target_function: Any,
        complexity_level: str = "basic",
    ) -> str | None:
        """Synthesize a new example from extracted patterns.

        Combines the best patterns to create an example tailored
        to the target function.
        """
        if not patterns:
            return None

        # Filter patterns by complexity
        suitable_patterns = [
            p for p in patterns if p.get("complexity", "basic") == complexity_level
        ]

        if not suitable_patterns:
            # Fallback to any pattern
            suitable_patterns = patterns

        if not suitable_patterns:
            return None

        # Use the best pattern as base
        best_pattern = suitable_patterns[0]

        # Extract components
        setup_lines = best_pattern.get("setup_code", [])
        call_lines = best_pattern.get("function_calls", [])
        result_lines = best_pattern.get("assertions", [])

        # Adapt to target function
        if hasattr(target_function, "signature"):
            sig = target_function.signature
            func_name = sig.name

            # Replace function names in calls
            adapted_calls = []
            for call in call_lines:
                # Simple name replacement - more sophisticated version would use AST
                adapted = self._adapt_function_call(call, func_name, sig)
                if adapted:
                    adapted_calls.append(adapted)

            if adapted_calls:
                call_lines = adapted_calls

        # Combine into example
        example_lines: list[str] = []

        # Add setup if meaningful
        meaningful_setup = [
            line
            for line in setup_lines
            if not line.strip().startswith("import") and line.strip()
        ]
        if meaningful_setup:
            # Add >>> prefix to setup lines
            example_lines.extend(f">>> {line}" for line in meaningful_setup)

        # Add function calls with >>> prefix
        if call_lines:
            example_lines.extend(f">>> {line}" for line in call_lines)

        # Add results if present (without >>> prefix)
        if result_lines:
            # Results typically don't have >>> prefix
            for line in result_lines[:1]:  # Limit to one result line
                if line.strip().startswith(("assert", "print")):
                    example_lines.append(f">>> {line}")
                else:
                    # Expected output (no >>> prefix)
                    example_lines.append(line)

        return "\n".join(example_lines) if example_lines else None

    def _adapt_function_call(
        self, call_line: str, target_name: str, target_signature: Any
    ) -> str | None:
        """Adapt a function call line to use the target function."""
        # Extract the pattern of the call

        # Match function calls: result = func(args) or func(args)
        match = re.search(r"(\w+)?\s*=?\s*(\w+)\s*\((.*?)\)", call_line)
        if not match:
            return None

        result_var = match.group(1)
        _old_func_name = match.group(2)  # Unused but needed for regex grouping
        args_str = match.group(3)

        # Build new call
        if result_var:
            new_call = f"{result_var} = {target_name}({args_str})"
        else:
            new_call = f"{target_name}({args_str})"

        # Preserve indentation
        indent = len(call_line) - len(call_line.lstrip())
        return " " * indent + new_call

    def _extract_examples_from_docstring(self, docstring_text: str) -> list[str]:
        """Extract example code blocks from a docstring."""
        examples = []
        logger.debug(f"Extracting examples from docstring: {docstring_text[:100]}...")

        # Try parsing with docstring_parser first
        try:
            from docstring_parser import parse as parse_docstring

            parsed = parse_docstring(docstring_text)
            if hasattr(parsed, "examples") and parsed.examples:
                logger.debug(f"docstring_parser found {len(parsed.examples)} examples")
                for example in parsed.examples:
                    if hasattr(example, "snippet") and example.snippet:
                        examples.append(example.snippet)
                        logger.debug(
                            f"Added example from parser: {example.snippet[:50]}..."
                        )
        except Exception as e:
            logger.debug(f"docstring_parser failed: {e}")

        # Fallback to regex patterns if parser fails
        if not examples:
            logger.debug("Trying regex extraction...")
            # Try multiple patterns for Examples section

            # Pattern 1: Look for Examples: section with any content after it
            example_patterns = [
                # Pattern for Examples: followed by content
                r"Examples?:?\s*\n((?:.*\n)*?)(?=\n\s*\w+:|$)",
                # Pattern for indented content after Examples:
                r"Examples?:?\s*\n((?:\s+.*\n)*)",
                # Pattern to capture everything after Examples: until end of string
                r"Examples?:?\s*\n(.*?)$",
            ]

            for pattern in example_patterns:
                example_match = re.search(
                    pattern, docstring_text, re.MULTILINE | re.DOTALL
                )
                if example_match:
                    example_text = example_match.group(1)
                    logger.debug(
                        f"Regex matched with pattern, example text: {example_text[:100]}..."
                    )

                    # Extract >>> code blocks
                    code_blocks = re.findall(
                        r">>>.*?(?=\n(?!>>>|\.\.\.)|$)", example_text, re.DOTALL
                    )
                    logger.debug(f"Found {len(code_blocks)} code blocks")

                    for i, block in enumerate(code_blocks):
                        logger.debug(f"Processing block {i}: {block[:50]}...")
                        # Clean up the code block
                        lines = []
                        for line in block.split("\n"):
                            if line.strip().startswith((">>>", "...")):
                                code = line.strip()[3:].strip()
                                if code:
                                    lines.append(code)
                        if lines:
                            example_code = "\n".join(lines)
                            examples.append(example_code)
                            logger.debug(f"Added example {i}: {example_code[:50]}...")

                    if examples:
                        break  # Stop if we found examples

        logger.debug(f"Total examples extracted: {len(examples)}")
        return examples

    def _adapt_example_code(self, example: str, function: Any) -> str | None:
        """Adapt example code from related function to current function."""
        if not hasattr(function, "signature"):
            return None

        sig = function.signature
        function_name = sig.name

        # Simple adaptation - replace function names
        # In a more sophisticated implementation, we'd parse and rewrite the AST
        adapted = example

        # Try to identify function calls in the example
        # This is a simplified approach - real implementation would be more robust
        func_call_pattern = r"\b(\w+)\s*\("
        matches = re.findall(func_call_pattern, example)

        if matches:
            # Replace the most common function name with our target function
            old_func_name = max(set(matches), key=matches.count)
            adapted = re.sub(rf"\b{old_func_name}\b", function_name, adapted)

        # Add basic setup if needed
        if "await" in adapted and "async" not in adapted:
            adapted = f"# In an async context\n{adapted}"

        # Ensure the example has >>> prefixes if it doesn't already
        if ">>>" not in adapted:
            lines = adapted.strip().split("\n")
            formatted_lines = []
            for line in lines:
                if line.strip():
                    # Don't add >>> to comments or expected output
                    if line.strip().startswith("#") or (
                        not any(c in line for c in ["(", "=", "import", "from"])
                    ):
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append(f">>> {line}")
                else:
                    formatted_lines.append(line)
            adapted = "\n".join(formatted_lines)

        return adapted
