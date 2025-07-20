"""
Return type suggestion generator for handling return-related documentation issues.

This module specializes in generating suggestions for return type mismatches,
missing return documentation, and other return-specific problems.
"""

import ast
import re

from ...parser.docstring_models import DocstringReturns
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


class ReturnAnalysisResult:
    """Result of analyzing return statements in function code."""

    def __init__(self):
        self.return_types: set[str] = set()
        self.has_explicit_return: bool = False
        self.has_implicit_none: bool = False
        self.is_generator: bool = False
        self.is_async: bool = False
        self.conditional_returns: list[str] = []
        self.complexity_score: float = 0.0


class ReturnStatementAnalyzer:
    """Analyze function code to determine actual return behavior."""

    def analyze(self, source_code: str) -> ReturnAnalysisResult:
        """Analyze return statements in function source code."""
        result = ReturnAnalysisResult()

        try:
            tree = ast.parse(source_code)
            self._analyze_ast(tree, result)
        except SyntaxError:
            # If we can't parse, provide minimal analysis
            result.complexity_score = 0.1
            result.has_implicit_none = True

        return result

    def _analyze_ast(self, tree: ast.AST, result: ReturnAnalysisResult) -> None:
        """Analyze AST for return patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._analyze_function_node(node, result)
                break
            elif isinstance(node, ast.AsyncFunctionDef):
                result.is_async = True
                self._analyze_function_node(node, result)
                break

    def _analyze_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        result: ReturnAnalysisResult,
    ) -> None:
        """Analyze a function node for return patterns."""
        has_return_value = False

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                result.has_explicit_return = True
                if child.value is not None:
                    has_return_value = True
                    return_type = self._infer_return_type(child.value)
                    if return_type:
                        result.return_types.add(return_type)

            elif isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                result.is_generator = True
                if hasattr(child, "value") and child.value:
                    yield_type = self._infer_return_type(child.value)
                    if yield_type:
                        result.return_types.add(f"Generator[{yield_type}]")

        # Check for implicit None return
        if not has_return_value and not result.is_generator:
            result.has_implicit_none = True
            result.return_types.add("None")

        # Calculate complexity based on number of return paths
        result.complexity_score = min(len(result.return_types) / 5.0, 1.0)

    def _infer_return_type(self, node: ast.expr) -> str | None:
        """Infer return type from AST node."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            elif isinstance(node.value, bool):
                return "bool"
            elif isinstance(node.value, int):
                return "int"
            elif isinstance(node.value, float):
                return "float"
            elif isinstance(node.value, str):
                return "str"

        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.Set):
            return "Set"

        elif isinstance(node, ast.Name):
            # Could be a variable, try to infer from name
            if node.id in ("True", "False"):
                return "bool"
            elif node.id == "None":
                return "None"

        return None


class ReturnSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for return-related issues."""

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate return documentation fixes."""
        issue = context.issue

        if issue.issue_type == "return_type_mismatch":
            return self._fix_return_type(context)
        elif issue.issue_type == "missing_returns":
            return self._add_missing_return_documentation(context)
        elif issue.issue_type == "return_description_vague":
            return self._improve_return_description(context)
        elif issue.issue_type == "generator_return_incorrect":
            return self._fix_generator_return(context)
        else:
            return self._generic_return_fix(context)

    def _fix_return_type(self, context: SuggestionContext) -> Suggestion:
        """Fix return type mismatch between code and documentation."""
        function = context.function

        # Analyze actual return statements
        analyzer = ReturnStatementAnalyzer()
        source_code = getattr(function, "source_code", "")
        if not source_code:
            return self._create_fallback_suggestion(
                context, "Source code not available for analysis"
            )

        return_analysis = analyzer.analyze(source_code)

        # Determine the best return type to document
        suggested_type = self._determine_best_return_type(return_analysis, function)

        if not suggested_type:
            return self._create_fallback_suggestion(
                context, "Could not determine appropriate return type"
            )

        # Generate corrected docstring
        corrected_docstring = self._update_return_type_in_docstring(
            context, suggested_type, return_analysis
        )

        suggestion = self._create_suggestion(
            context,
            corrected_docstring,
            f"Fix return type documentation: {suggested_type}",
            confidence=0.85 if len(return_analysis.return_types) == 1 else 0.7,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

        return suggestion

    def _add_missing_return_documentation(
        self, context: SuggestionContext
    ) -> Suggestion:
        """Add missing return documentation."""
        function = context.function

        # Analyze function to determine return behavior
        analyzer = ReturnStatementAnalyzer()
        source_code = getattr(function, "source_code", "")

        if source_code:
            return_analysis = analyzer.analyze(source_code)
            suggested_type = self._determine_best_return_type(return_analysis, function)
        else:
            # Fallback to function signature
            suggested_type = self._get_return_type_from_signature(function)

        if not suggested_type or suggested_type == "None":
            return self._create_fallback_suggestion(
                context, "Function appears to return None - no documentation needed"
            )

        # Generate docstring with return documentation
        updated_docstring = self._add_return_to_docstring(context, suggested_type)

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Add missing return documentation: {suggested_type}",
            confidence=0.8,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

        return suggestion

    def _improve_return_description(self, context: SuggestionContext) -> Suggestion:
        """Improve vague or unclear return descriptions."""
        function = context.function
        docstring = context.docstring

        current_return = getattr(docstring, "returns", None)
        if not current_return:
            return self._add_missing_return_documentation(context)

        # Analyze the function to suggest better description
        analyzer = ReturnStatementAnalyzer()
        source_code = getattr(function, "source_code", "")

        if source_code:
            return_analysis = analyzer.analyze(source_code)
            improved_description = self._generate_improved_return_description(
                function, return_analysis, current_return
            )
        else:
            improved_description = self._generate_basic_return_description(
                function, current_return
            )

        # Update docstring with improved description
        updated_docstring = self._update_return_description_in_docstring(
            context, improved_description
        )

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            "Improve return description clarity",
            confidence=0.6,  # Lower confidence for subjective improvements
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

        return suggestion

    def _fix_generator_return(self, context: SuggestionContext) -> Suggestion:
        """Fix return documentation for generator functions."""
        function = context.function

        # Analyze generator yield patterns
        analyzer = ReturnStatementAnalyzer()
        source_code = getattr(function, "source_code", "")

        if source_code:
            return_analysis = analyzer.analyze(source_code)
            if return_analysis.is_generator:
                # Determine yielded types
                yield_types = [
                    t.replace("Generator[", "").replace("]", "")
                    for t in return_analysis.return_types
                    if t.startswith("Generator[")
                ]

                if yield_types:
                    if len(set(yield_types)) == 1:
                        generator_type = f"Generator[{yield_types[0]}, None, None]"
                    else:
                        generator_type = f"Generator[Union[{', '.join(set(yield_types))}], None, None]"
                else:
                    generator_type = "Generator[Any, None, None]"
            else:
                generator_type = "Generator[Any, None, None]"
        else:
            generator_type = "Generator[Any, None, None]"

        # Update docstring with correct generator return type
        updated_docstring = self._update_return_type_in_docstring(
            context, generator_type, None
        )

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Fix generator return documentation: {generator_type}",
            confidence=0.9,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

        return suggestion

    def _determine_best_return_type(
        self, analysis: ReturnAnalysisResult, function
    ) -> str | None:
        """Determine the best return type to document."""
        if analysis.is_generator:
            return "Generator"

        if analysis.has_implicit_none and not analysis.return_types:
            return "None"

        # Get type from function signature if available
        signature_type = self._get_return_type_from_signature(function)
        if signature_type:
            return signature_type

        # Use inferred types from analysis
        if len(analysis.return_types) == 1:
            return list(analysis.return_types)[0]
        elif len(analysis.return_types) > 1:
            # Multiple return types - suggest Union
            types = sorted(analysis.return_types)
            return f"Union[{', '.join(types)}]"

        return None

    def _get_return_type_from_signature(self, function) -> str | None:
        """Extract return type from function signature."""
        if hasattr(function, "signature") and hasattr(
            function.signature, "return_annotation"
        ):
            return function.signature.return_annotation
        return None

    def _update_return_type_in_docstring(
        self,
        context: SuggestionContext,
        return_type: str,
        analysis: ReturnAnalysisResult | None,
    ) -> str:
        """Update return type in existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=self.config.max_line_length)

        # Create new return documentation
        description = self._generate_return_description(return_type, analysis)
        new_return = DocstringReturns(type_str=return_type, description=description)

        # Generate updated docstring
        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=getattr(docstring, "parameters", []),
            returns=new_return,
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _add_return_to_docstring(
        self, context: SuggestionContext, return_type: str
    ) -> str:
        """Add return documentation to existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=self.config.max_line_length)

        # Create return documentation
        description = self._generate_return_description(return_type, None)
        new_return = DocstringReturns(type_str=return_type, description=description)

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=getattr(docstring, "parameters", []),
            returns=new_return,
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _generate_return_description(
        self, return_type: str, analysis: ReturnAnalysisResult | None
    ) -> str:
        """Generate appropriate return description based on type and analysis."""
        if return_type == "None":
            return "None"

        if return_type.startswith("Generator"):
            return "Generator yielding values"

        if return_type.startswith("Union"):
            return "The result of the operation"

        # Basic descriptions based on type
        type_descriptions = {
            "bool": "True if successful, False otherwise",
            "str": "The string result",
            "int": "The integer result",
            "float": "The numeric result",
            "List": "List of results",
            "Dict": "Dictionary containing results",
            "Tuple": "Tuple of results",
            "Set": "Set of results",
        }

        return type_descriptions.get(return_type, f"The {return_type.lower()} result")

    def _generate_improved_return_description(
        self, function, analysis: ReturnAnalysisResult, current_return
    ) -> str:
        """Generate improved return description based on function analysis."""
        current_desc = getattr(current_return, "description", "")

        # Check if current description is vague
        vague_patterns = [
            r"^(the )?result$",
            r"^return[s]?$",
            r"^output$",
            r"^value$",
        ]

        is_vague = any(
            re.match(pattern, current_desc.lower().strip())
            for pattern in vague_patterns
        )

        if not is_vague and len(current_desc) > 10:
            return current_desc  # Keep existing if it's descriptive enough

        # Generate improved description based on function name and analysis
        function_name = getattr(function.signature, "name", "function")

        if "get" in function_name.lower():
            return (
                f"The retrieved {function_name.replace('get_', '').replace('get', '')}"
            )
        elif "calculate" in function_name.lower() or "compute" in function_name.lower():
            return "The calculated result"
        elif "validate" in function_name.lower() or "check" in function_name.lower():
            return "True if validation passes, False otherwise"
        elif "create" in function_name.lower() or "build" in function_name.lower():
            return f"The created {function_name.replace('create_', '').replace('create', '')}"

        return "The function result"

    def _update_return_description_in_docstring(
        self, context: SuggestionContext, new_description: str
    ) -> str:
        """Update return description in existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=self.config.max_line_length)

        # Update existing return documentation
        current_return = getattr(docstring, "returns", None)
        if current_return:
            updated_return = DocstringReturns(
                type_str=getattr(current_return, "type_str", ""),
                description=new_description,
            )
        else:
            updated_return = DocstringReturns(type_str="", description=new_description)

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=getattr(docstring, "parameters", []),
            returns=updated_return,
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _detect_style(self, docstring) -> DocstringStyle:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            format_mapping = {
                "google": DocstringStyle.GOOGLE,
                "numpy": DocstringStyle.NUMPY,
                "sphinx": DocstringStyle.SPHINX,
                "rest": DocstringStyle.REST,
            }
            return format_mapping.get(
                str(docstring.format).lower(), DocstringStyle.GOOGLE
            )

        return DocstringStyle.GOOGLE  # Default fallback

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
            generator_name=self.__class__.__name__,
            generator_version="1.0.0",
            analysis_type=suggestion_type.value,
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
            f"Unable to generate specific return fix: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

    def _generic_return_fix(self, context: SuggestionContext) -> Suggestion:
        """Generic return fix for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown return issue type: {context.issue.issue_type}"
        )
