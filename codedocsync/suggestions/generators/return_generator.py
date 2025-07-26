"""
Return type suggestion generator for handling return-related documentation issues.

This module specializes in generating suggestions for return type mismatches,
missing return documentation, and other return-specific problems.
"""

import ast
import logging
import re
from typing import Any

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

logger = logging.getLogger(__name__)


class ReturnAnalysisResult:
    """Result of analyzing return statements in function code."""

    def __init__(self) -> None:
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

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the generator."""
        super().__init__(config)
        self._used_rag = False  # Track if RAG was used

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate return documentation fixes."""
        # Store context for RAG-enhanced description generation
        self._used_rag = False  # Reset for each generation
        self._current_context = context

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
            # Generate basic description based on function name
            improved_description = self._generate_improved_return_description(
                function, ReturnAnalysisResult(), current_return
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
        self, analysis: ReturnAnalysisResult, function: Any
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

    def _get_return_type_from_signature(self, function: Any) -> str | None:
        """Extract return type from function signature."""
        if hasattr(function, "signature") and hasattr(
            function.signature, "return_annotation"
        ):
            return str(function.signature.return_annotation)
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
        style_enum = DocstringStyle(style) if isinstance(style, str) else style
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

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
        style_enum = DocstringStyle(style) if isinstance(style, str) else style
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

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
        # Check if we have context with RAG examples
        if hasattr(self, "_current_context") and self._current_context:
            rag_desc = self._generate_rag_enhanced_return_description(return_type)
            if rag_desc:
                logger.info(
                    f"Using RAG-enhanced description for return type '{return_type}'"
                )
                self._used_rag = True  # Mark that RAG was used
                return rag_desc
            else:
                logger.debug(
                    f"No RAG enhancement available for return type '{return_type}', using basic description"
                )
        else:
            logger.debug(f"No RAG context available for return type '{return_type}'")

        # Fallback to enhanced semantic descriptions
        return self._generate_semantic_return_description(return_type, analysis)

    def _generate_improved_return_description(
        self, function: Any, analysis: ReturnAnalysisResult, current_return: Any
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
        style_enum = DocstringStyle(style) if isinstance(style, str) else style
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

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

    def _detect_style(self, docstring: Any) -> str:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            # Return the string format directly
            return str(docstring.format.value)

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
            used_rag_examples=getattr(self, "_used_rag", False),
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
            f"Unable to generate specific return fix: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

    def _generic_return_fix(self, context: SuggestionContext) -> Suggestion:
        """Generic return fix for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown return issue type: {context.issue.issue_type}"
        )

    def _generate_rag_enhanced_return_description(self, return_type: str) -> str | None:
        """Generate return description using RAG examples."""
        if not self._current_context or not self._current_context.related_functions:
            logger.debug("No RAG examples available for return description generation")
            return None

        logger.info(
            f"RAG: Generating return description for type '{return_type}' using {len(self._current_context.related_functions)} examples"
        )

        # Extract return descriptions from RAG examples
        return_patterns = self._extract_return_patterns_from_examples(
            return_type, self._current_context.related_functions
        )

        if not return_patterns:
            logger.debug(
                f"RAG: No matching return patterns found for type '{return_type}'"
            )
            return None

        logger.info(
            f"RAG: Found {len(return_patterns)} return patterns for type '{return_type}'"
        )

        # Synthesize description from patterns
        if len(return_patterns) > 1:
            # Use multiple patterns to create a better description
            description = self._synthesize_return_description_from_patterns(
                return_patterns, return_type
            )
        else:
            # Use the single best pattern
            best_pattern = return_patterns[0]
            description = best_pattern["description"]
            logger.debug(
                f"RAG: Using pattern with similarity {best_pattern.get('similarity', 0):.2f}: '{description[:100]}...'"
            )

        # Adapt the description for this specific context
        adapted_description = self._adapt_return_description_for_context(
            description, return_type
        )

        return adapted_description

    def _extract_return_patterns_from_examples(
        self, return_type: str, examples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract return description patterns from RAG examples."""
        patterns = []

        logger.debug(
            f"Extracting return patterns from {len(examples)} examples for type '{return_type}'"
        )

        for example in examples:
            docstring_content = example.get("docstring", "")
            if not docstring_content:
                logger.debug(
                    f"Skipping example with empty docstring: {example.get('signature', 'unknown')[:50]}"
                )
                continue

            # Parse the example docstring to find return descriptions
            return_desc = self._extract_return_from_docstring(docstring_content)

            if return_desc:
                logger.debug(
                    f"Found return pattern in example: {example.get('signature', '')[:50]}"
                )
                patterns.append(
                    {
                        "description": return_desc["description"],
                        "type_hint": return_desc.get("type", ""),
                        "similarity": example.get("similarity", 0.0),
                        "source": example.get("signature", ""),
                    }
                )
            else:
                logger.debug(
                    f"No return pattern found in example: {example.get('signature', 'unknown')[:50]}"
                )

        # Sort by similarity score
        patterns.sort(key=lambda x: x["similarity"], reverse=True)

        logger.debug(
            f"Extracted {len(patterns)} return patterns, top similarity: {patterns[0]['similarity'] if patterns else 0:.2f}"
        )

        return patterns

    def _extract_return_from_docstring(
        self, docstring_content: str
    ) -> dict[str, Any] | None:
        """Extract return description from a docstring."""
        # Google style pattern
        google_pattern = r"Returns?:\s*\n\s+(.+?)(?=\n\s*\n|\n\s*\w+:|\Z)"
        google_match = re.search(
            google_pattern, docstring_content, re.DOTALL | re.MULTILINE
        )
        if google_match:
            return {"description": google_match.group(1).strip()}

        # NumPy style pattern
        numpy_pattern = (
            r"Returns?\s*\n\s*-+\s*\n\s*(.+?)(?=\n\s*\n|\n\s*\w+\s*\n\s*-+|\Z)"
        )
        numpy_match = re.search(
            numpy_pattern, docstring_content, re.DOTALL | re.MULTILINE
        )
        if numpy_match:
            return {"description": numpy_match.group(1).strip()}

        # Sphinx style pattern
        sphinx_pattern = r":returns?:\s*(.+?)(?=:|\Z)"
        sphinx_match = re.search(sphinx_pattern, docstring_content, re.DOTALL)
        if sphinx_match:
            return {"description": sphinx_match.group(1).strip()}

        return None

    def _synthesize_return_description_from_patterns(
        self, patterns: list[dict[str, Any]], return_type: str
    ) -> str:
        """Synthesize a return description from multiple RAG patterns."""
        # Extract unique concepts from all patterns
        concepts = []
        for pattern in patterns[:3]:  # Use top 3 patterns
            description = pattern["description"]
            # Extract core meaning from each pattern
            core_meaning = self._extract_core_meaning(description, return_type)
            if core_meaning and core_meaning not in concepts:
                concepts.append(core_meaning)

        if not concepts:
            # Fallback to the best pattern
            return patterns[0]["description"]

        # Merge concepts intelligently
        if len(concepts) == 1:
            return concepts[0]
        else:
            # Select the most specific/informative concept
            return max(concepts, key=lambda x: len(x.split()))

    def _adapt_return_description_for_context(
        self, description: str, return_type: str
    ) -> str:
        """Adapt the return description for the specific context."""
        # Extract core meaning first
        core_description = self._extract_core_meaning(description, return_type)

        # Adapt based on semantic patterns
        adapted = self._adapt_based_on_semantics(core_description, return_type)

        # Ensure proper grammar
        final_description = self._ensure_proper_grammar(adapted, return_type)

        return final_description

    def _extract_core_meaning(self, description: str, return_type: str) -> str:
        """Extract the core meaning from a return description."""
        # Remove common boilerplate patterns
        patterns_to_remove = [
            r"^(the\s+)?return(s|ed)?\s+(value\s+)?",
            r"^(the\s+)?result(s)?\s+(of\s+)?",
            r"^(the\s+)?output\s+",
            r"^\s*None\s*$",
            r"^returns?\s+",
            r"^(the\s+)?function\s+returns?\s+",
        ]

        cleaned = description.strip()
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove redundant type mentions
        if return_type and return_type.lower() in cleaned.lower():
            # Remove patterns like "The dict result" or "A dictionary containing"
            type_patterns = [
                f"^(a|an|the)\\s+{re.escape(return_type.lower())}\\s+",
                f"\\s+{re.escape(return_type.lower())}\\s*$",
                f"\\s*\\({re.escape(return_type)}\\)\\s*$",
            ]
            for pattern in type_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Extract meaningful content after common prefixes
        extraction_patterns = [
            r"(?:containing|with|of|that contains?)\s+(.+)",
            r"(?:representing|indicating|showing)\s+(.+)",
        ]

        for pattern in extraction_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
                break

        return cleaned if cleaned else description

    def _adapt_based_on_semantics(self, description: str, return_type: str) -> str:
        """Adapt description based on semantic patterns and return type."""
        function_context = ""
        if hasattr(self, "_current_context") and self._current_context:
            function_name = getattr(
                self._current_context.function.signature, "name", ""
            )
            function_context = function_name.lower()

        # Type-specific patterns
        if return_type:
            type_lower = return_type.lower()

            # Boolean returns
            if type_lower == "bool":
                if any(
                    word in function_context
                    for word in ["validate", "check", "verify", "is_", "has_"]
                ):
                    return "True if validation passes, False otherwise"
                elif any(word in function_context for word in ["exists", "contains"]):
                    return "True if exists, False otherwise"
                elif description and len(description) > 20:
                    return description
                else:
                    return "Success status"

            # Dictionary returns
            elif "dict" in type_lower or "mapping" in type_lower:
                if "config" in function_context:
                    return "Configuration settings"
                elif "stats" in function_context or "metrics" in function_context:
                    return "Statistics and metrics"
                elif "results" in function_context:
                    return "Processing results"
                elif description and "containing" in description:
                    return description
                else:
                    return "Mapping of results"

            # List/Collection returns
            elif any(t in type_lower for t in ["list", "sequence", "collection"]):
                if "errors" in function_context:
                    return "List of errors encountered"
                elif "results" in function_context:
                    return "List of processing results"
                elif "items" in function_context or "elements" in function_context:
                    return "Collection of items"
                elif description and len(description) > 15:
                    return description
                else:
                    return "List of results"

            # String returns
            elif type_lower == "str":
                if "path" in function_context:
                    return "File or directory path"
                elif "url" in function_context:
                    return "URL string"
                elif "name" in function_context:
                    return "Name identifier"
                elif "message" in function_context:
                    return "Status or error message"
                elif description and len(description) > 10:
                    return description
                else:
                    return "String result"

            # Numeric returns
            elif type_lower in ["int", "float"]:
                if "count" in function_context:
                    return "Number of items"
                elif "size" in function_context or "length" in function_context:
                    return "Size or length value"
                elif "score" in function_context:
                    return "Calculated score"
                elif "index" in function_context:
                    return "Index position"
                elif description and len(description) > 10:
                    return description
                else:
                    return "Numeric result"

            # Generator returns
            elif "generator" in type_lower:
                if description and "yield" in description:
                    return description
                else:
                    return "Iterator of results"

            # Tuple returns
            elif "tuple" in type_lower:
                if description and ("pair" in description or "both" in description):
                    return description
                else:
                    return "Tuple containing results"

        # Function name-based patterns (when no specific type match)
        if function_context:
            if function_context.startswith("get_"):
                item = function_context[4:].replace("_", " ")
                return f"Retrieved {item}"
            elif function_context.startswith("create_"):
                item = function_context[7:].replace("_", " ")
                return f"Created {item}"
            elif function_context.startswith(
                "calculate_"
            ) or function_context.startswith("compute_"):
                return "Calculated result"
            elif function_context.startswith("fetch_"):
                item = function_context[6:].replace("_", " ")
                return f"Fetched {item}"
            elif function_context.startswith("parse_"):
                return "Parsed data"
            elif function_context.startswith("process_"):
                return "Processed result"
            elif function_context.startswith("analyze_"):
                return "Analysis results"

        # Return original if no pattern matches but it's descriptive enough
        if description and len(description) > 15:
            return description

        # Generic fallback based on type
        return self._generate_type_based_fallback(return_type)

    def _ensure_proper_grammar(self, description: str, return_type: str) -> str:
        """Ensure the return description has proper grammar."""
        if not description:
            return self._generate_type_based_fallback(return_type)

        # Capitalize first letter
        description = description[0].upper() + description[1:] if description else ""

        # Ensure it ends with proper punctuation
        if description and description[-1] not in ".!?":
            description += "."

        # Remove double spaces
        description = re.sub(r"\s+", " ", description)

        # Remove redundant "return" mentions
        description = re.sub(r"^Returns?\s+", "", description, flags=re.IGNORECASE)

        return description.strip()

    def _generate_type_based_fallback(self, return_type: str) -> str:
        """Generate a fallback description based on return type."""
        if not return_type or return_type == "None":
            return "None"

        type_fallbacks = {
            "bool": "Success status",
            "str": "Result string",
            "int": "Integer value",
            "float": "Numeric value",
            "list": "List of results",
            "dict": "Dictionary of results",
            "tuple": "Tuple of values",
            "set": "Set of unique values",
            "Any": "Result value",
        }

        # Check for generic types
        for base_type, fallback in type_fallbacks.items():
            if base_type in return_type:
                return fallback

        # For complex types, simplify
        if "[" in return_type:
            # Extract base type from generic
            base = return_type.split("[")[0]
            if base.lower() == "generator":
                return "Iterator of results"
            elif base.lower() == "optional":
                inner_type = return_type[9:-1]  # Remove "Optional[" and "]"
                return f"{self._generate_type_based_fallback(inner_type)} or None"

        return "Operation result"

    def _generate_semantic_return_description(
        self, return_type: str, analysis: ReturnAnalysisResult | None
    ) -> str:
        """Generate semantic description when RAG is not available."""
        # Use semantic patterns directly
        return self._adapt_based_on_semantics("", return_type)
