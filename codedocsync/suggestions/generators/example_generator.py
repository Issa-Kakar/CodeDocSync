"""
Example generation system for creating usage examples in docstrings.

This module specializes in generating realistic, runnable examples for functions
based on their signatures, behavior, and common usage patterns.
"""

import ast
import re
from dataclasses import dataclass
from typing import Any

from ...parser.ast_parser import FunctionParameter
from ..base import BaseSuggestionGenerator
from ..models import (
    Suggestion,
    SuggestionContext,
    SuggestionDiff,
    SuggestionMetadata,
    SuggestionType,
)
from ..templates.base import get_template

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

    def __init__(self):
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

    def analyze_function(self, function, source_code: str = "") -> dict[str, Any]:
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
            elif isinstance(node, (ast.Yield, ast.YieldFrom)):
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

    def __init__(self):
        self.value_generator = ParameterValueGenerator()
        self.pattern_analyzer = ExamplePatternAnalyzer()

    def generate_examples(
        self, function, source_code: str = "", count: int = 1
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
        self, function, analysis: dict[str, Any]
    ) -> ExampleTemplate | None:
        """Generate a basic usage example."""
        if not hasattr(function, "signature"):
            return None

        sig = function.signature
        function_name = sig.name

        # Generate parameter values
        call_args = []
        setup_lines = []
        imports = []

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
        self, function, analysis: dict[str, Any]
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
        self, function, analysis: dict[str, Any]
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

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate example documentation fixes."""
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

        # Generate examples
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
        template = get_template(style, max_line_length=self.config.max_line_length)

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

    def _detect_style(self, docstring) -> str:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            # Return the string format directly
            return docstring.format.value

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
            getattr(context.docstring, "raw_text", "") if context.docstring else ""
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
        )

        return Suggestion(
            original_text=original_text,
            suggested_text=suggested_text,
            suggestion_type=suggestion_type,
            confidence=confidence,
            description=description,
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
