"""
Exception documentation generator for handling raises-related documentation issues.

This module specializes in generating suggestions for missing exception documentation,
incorrect exception types, and comprehensive exception analysis.
"""

import ast
import re
from dataclasses import dataclass
from typing import Any

from ...parser.docstring_models import DocstringRaises
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


@dataclass
class ExceptionInfo:
    """Information about an exception that can be raised."""

    exception_type: str
    description: str
    condition: str | None = None
    line_number: int | None = None
    is_re_raised: bool = False
    confidence: float = 1.0


class ExceptionAnalyzer:
    """Analyze function code to find all possible exceptions."""

    def __init__(self) -> None:
        self.builtin_exceptions = {
            "ValueError": "When an invalid value is provided",
            "TypeError": "When an invalid type is provided",
            "KeyError": "When a key is not found",
            "IndexError": "When an index is out of range",
            "AttributeError": "When an attribute is not found",
            "FileNotFoundError": "When a file is not found",
            "PermissionError": "When permission is denied",
            "RuntimeError": "When a runtime error occurs",
            "NotImplementedError": "When functionality is not implemented",
            "OSError": "When an OS-related error occurs",
            "IOError": "When an I/O operation fails",
        }

    def analyze_exceptions(self, source_code: str) -> list[ExceptionInfo]:
        """Find all exceptions that can be raised."""
        exceptions: list[ExceptionInfo] = []

        try:
            tree = ast.parse(source_code)
            self._analyze_ast(tree, exceptions)
        except SyntaxError:
            # If we can't parse, return basic exceptions
            exceptions.append(
                ExceptionInfo(
                    exception_type="Exception",
                    description="When an error occurs",
                    confidence=0.3,
                )
            )

        return self._deduplicate_exceptions(exceptions)

    def _analyze_ast(self, tree: ast.AST, exceptions: list[ExceptionInfo]) -> None:
        """Analyze AST for exception patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                self._analyze_function_node(node, exceptions)
                break

    def _analyze_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        exceptions: list[ExceptionInfo],
    ) -> None:
        """Analyze a function node for exception patterns."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                self._analyze_raise_statement(child, exceptions)
            elif isinstance(child, ast.Call):
                self._analyze_function_call(child, exceptions)
            elif isinstance(child, ast.Subscript):
                self._analyze_subscript(child, exceptions)
            elif isinstance(child, ast.Attribute):
                self._analyze_attribute_access(child, exceptions)

    def _analyze_raise_statement(
        self, node: ast.Raise, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze a raise statement."""
        if node.exc is None:
            # Bare raise - re-raising current exception
            exceptions.append(
                ExceptionInfo(
                    exception_type="Exception",
                    description="Re-raised exception",
                    is_re_raised=True,
                    line_number=node.lineno,
                    confidence=0.8,
                )
            )
            return

        exception_type = None
        if isinstance(node.exc, ast.Call):
            # raise ExceptionType(message)
            if isinstance(node.exc.func, ast.Name):
                exception_type = node.exc.func.id
        elif isinstance(node.exc, ast.Name):
            # raise exception_instance
            exception_type = node.exc.id

        if exception_type:
            description = self.builtin_exceptions.get(
                exception_type, f"When a {exception_type} condition occurs"
            )
            exceptions.append(
                ExceptionInfo(
                    exception_type=exception_type,
                    description=description,
                    line_number=node.lineno,
                    confidence=0.95,
                )
            )

    def _analyze_function_call(
        self, node: ast.Call, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze function calls that might raise exceptions."""
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            # Common functions that raise specific exceptions
            function_exceptions = {
                "open": ["FileNotFoundError", "PermissionError", "IOError"],
                "int": ["ValueError"],
                "float": ["ValueError"],
                "len": ["TypeError"],
                "max": ["ValueError"],
                "min": ["ValueError"],
                "next": ["StopIteration"],
                "iter": ["TypeError"],
                "getattr": ["AttributeError"],
                "setattr": ["AttributeError"],
                "delattr": ["AttributeError"],
            }

            if func_name in function_exceptions:
                for exc_type in function_exceptions[func_name]:
                    description = self.builtin_exceptions.get(
                        exc_type, f"When {func_name} fails"
                    )
                    exceptions.append(
                        ExceptionInfo(
                            exception_type=exc_type,
                            description=description,
                            condition=f"When calling {func_name}",
                            line_number=node.lineno,
                            confidence=0.6,
                        )
                    )

    def _analyze_subscript(
        self, node: ast.Subscript, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze subscript operations that might raise exceptions."""
        # Dictionary/list access can raise KeyError/IndexError
        exceptions.append(
            ExceptionInfo(
                exception_type="KeyError",
                description="When accessing a non-existent key",
                condition="When accessing dictionary keys",
                line_number=node.lineno,
                confidence=0.4,
            )
        )
        exceptions.append(
            ExceptionInfo(
                exception_type="IndexError",
                description="When accessing an invalid index",
                condition="When accessing list/tuple indices",
                line_number=node.lineno,
                confidence=0.4,
            )
        )

    def _analyze_attribute_access(
        self, node: ast.Attribute, exceptions: list[ExceptionInfo]
    ) -> None:
        """Analyze attribute access that might raise exceptions."""
        # Attribute access can raise AttributeError
        exceptions.append(
            ExceptionInfo(
                exception_type="AttributeError",
                description="When accessing a non-existent attribute",
                condition="When accessing object attributes",
                line_number=node.lineno,
                confidence=0.3,
            )
        )

    def _deduplicate_exceptions(
        self, exceptions: list[ExceptionInfo]
    ) -> list[ExceptionInfo]:
        """Remove duplicate exceptions and merge similar ones."""
        unique_exceptions: dict[str, ExceptionInfo] = {}

        for exc in exceptions:
            key = exc.exception_type
            if key in unique_exceptions:
                # Keep the one with higher confidence
                if exc.confidence > unique_exceptions[key].confidence:
                    unique_exceptions[key] = exc
            else:
                unique_exceptions[key] = exc

        # Sort by confidence and exception type
        return sorted(
            unique_exceptions.values(), key=lambda x: (-x.confidence, x.exception_type)
        )


class RaisesSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for exception documentation."""

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate exception documentation fixes."""
        issue = context.issue

        if issue.issue_type == "missing_raises":
            return self._add_missing_raises_documentation(context)
        elif issue.issue_type == "raises_type_mismatch":
            return self._fix_raises_type_mismatch(context)
        elif issue.issue_type == "raises_description_vague":
            return self._improve_raises_description(context)
        elif issue.issue_type == "undocumented_exceptions":
            return self._add_undocumented_exceptions(context)
        else:
            return self._generic_raises_fix(context)

    def _add_missing_raises_documentation(
        self, context: SuggestionContext
    ) -> Suggestion:
        """Add missing exception documentation."""
        function = context.function

        # Analyze function for exceptions
        source_code = getattr(function, "source_code", "")
        if not source_code:
            return self._create_fallback_suggestion(
                context, "Source code not available for exception analysis"
            )

        analyzer = ExceptionAnalyzer()
        exceptions = analyzer.analyze_exceptions(source_code)

        # Filter out low-confidence exceptions unless there are none
        high_confidence_exceptions = [e for e in exceptions if e.confidence >= 0.6]
        if not high_confidence_exceptions and exceptions:
            # Keep top 3 most likely exceptions
            high_confidence_exceptions = exceptions[:3]

        if not high_confidence_exceptions:
            return self._create_fallback_suggestion(
                context, "No significant exceptions detected"
            )

        # Generate docstring with exception documentation
        updated_docstring = self._add_exceptions_to_docstring(
            context, high_confidence_exceptions
        )

        exception_names = [e.exception_type for e in high_confidence_exceptions]
        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Add missing exception documentation: {', '.join(exception_names)}",
            confidence=min(0.8, max(e.confidence for e in high_confidence_exceptions)),
            suggestion_type=SuggestionType.RAISES_UPDATE,
        )

        return suggestion

    def _fix_raises_type_mismatch(self, context: SuggestionContext) -> Suggestion:
        """Fix mismatch between documented and actual exception types."""
        function = context.function
        docstring = context.docstring

        # Get documented exceptions
        documented_raises = getattr(docstring, "raises", [])
        documented_types = {
            r.exception_type for r in documented_raises if hasattr(r, "exception_type")
        }

        # Analyze actual exceptions
        source_code = getattr(function, "source_code", "")
        if source_code:
            analyzer = ExceptionAnalyzer()
            actual_exceptions = analyzer.analyze_exceptions(source_code)
            actual_types = {
                e.exception_type for e in actual_exceptions if e.confidence >= 0.6
            }
        else:
            actual_types = set()

        # Find mismatches
        missing_in_docs = actual_types - documented_types
        extra_in_docs = documented_types - actual_types

        if not missing_in_docs and not extra_in_docs:
            return self._create_fallback_suggestion(
                context, "No exception type mismatches found"
            )

        # Create corrected exception list
        corrected_exceptions = []

        # Keep documented exceptions that are still valid
        for doc_raise in documented_raises:
            if (
                hasattr(doc_raise, "exception_type")
                and doc_raise.exception_type not in extra_in_docs
            ):
                corrected_exceptions.append(doc_raise)

        # Add missing exceptions
        analyzer = ExceptionAnalyzer()
        for exc_type in missing_in_docs:
            # Find the corresponding exception info
            for exc in actual_exceptions:
                if exc.exception_type == exc_type:
                    corrected_exceptions.append(
                        DocstringRaises(
                            exception_type=exc_type, description=exc.description
                        )
                    )
                    break

        # Generate corrected docstring
        corrected_docstring = self._update_raises_in_docstring(
            context, corrected_exceptions
        )

        changes = []
        if missing_in_docs:
            changes.append(f"add {', '.join(missing_in_docs)}")
        if extra_in_docs:
            changes.append(f"remove {', '.join(extra_in_docs)}")

        suggestion = self._create_suggestion(
            context,
            corrected_docstring,
            f"Fix exception documentation: {'; '.join(changes)}",
            confidence=0.8,
            suggestion_type=SuggestionType.RAISES_UPDATE,
        )

        return suggestion

    def _improve_raises_description(self, context: SuggestionContext) -> Suggestion:
        """Improve vague or unclear exception descriptions."""
        docstring = context.docstring

        documented_raises = getattr(docstring, "raises", [])
        if not documented_raises:
            return self._add_missing_raises_documentation(context)

        # Analyze function for better exception descriptions
        function = context.function
        source_code = getattr(function, "source_code", "")

        improved_raises = []
        analyzer = ExceptionAnalyzer()

        for doc_raise in documented_raises:
            if not hasattr(doc_raise, "exception_type"):
                improved_raises.append(doc_raise)
                continue

            current_desc = getattr(doc_raise, "description", "")

            # Check if description is vague
            if self._is_vague_description(current_desc):
                improved_desc = self._generate_improved_exception_description(
                    doc_raise.exception_type, source_code, analyzer
                )
                improved_raises.append(
                    DocstringRaises(
                        exception_type=doc_raise.exception_type,
                        description=improved_desc,
                    )
                )
            else:
                improved_raises.append(doc_raise)

        # Generate updated docstring
        updated_docstring = self._update_raises_in_docstring(context, improved_raises)

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            "Improve exception description clarity",
            confidence=0.6,  # Lower confidence for subjective improvements
            suggestion_type=SuggestionType.RAISES_UPDATE,
        )

        return suggestion

    def _add_undocumented_exceptions(self, context: SuggestionContext) -> Suggestion:
        """Add documentation for exceptions found in code but not documented."""
        return self._add_missing_raises_documentation(context)

    def _is_vague_description(self, description: str) -> bool:
        """Check if exception description is vague."""
        if not description or len(description.strip()) < 10:
            return True

        vague_patterns = [
            r"^(when )?error(s)?( occur)?$",
            r"^exception$",
            r"^failure$",
            r"^problem$",
            r"^issue$",
        ]

        return any(
            re.match(pattern, description.lower().strip()) for pattern in vague_patterns
        )

    def _generate_improved_exception_description(
        self, exception_type: str, source_code: str, analyzer: ExceptionAnalyzer
    ) -> str:
        """Generate improved exception description based on analysis."""
        # Analyze code for this specific exception
        if source_code:
            exceptions = analyzer.analyze_exceptions(source_code)
            for exc in exceptions:
                if exc.exception_type == exception_type and exc.description:
                    return exc.description

        # Fallback to standard descriptions
        return analyzer.builtin_exceptions.get(
            exception_type, f"When a {exception_type} condition occurs"
        )

    def _add_exceptions_to_docstring(
        self, context: SuggestionContext, exceptions: list[ExceptionInfo]
    ) -> str:
        """Add exception documentation to existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=self.config.max_line_length)

        # Convert to DocstringRaises objects
        raises_docs = []
        for exc in exceptions:
            raises_docs.append(
                DocstringRaises(
                    exception_type=exc.exception_type, description=exc.description
                )
            )

        # Combine with existing raises
        existing_raises = getattr(docstring, "raises", [])
        all_raises = existing_raises + raises_docs

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=getattr(docstring, "parameters", []),
            returns=getattr(docstring, "returns", None),
            raises=all_raises,
            examples=getattr(docstring, "examples", []),
        )

    def _update_raises_in_docstring(
        self, context: SuggestionContext, raises_list: list[DocstringRaises]
    ) -> str:
        """Update raises documentation in existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=self.config.max_line_length)

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=getattr(docstring, "parameters", []),
            returns=getattr(docstring, "returns", None),
            raises=raises_list,
            examples=getattr(docstring, "examples", []),
        )

    def _detect_style(self, docstring: Any) -> DocstringStyle:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            # Convert string to DocstringStyle enum
            style_value = docstring.format.value
            return DocstringStyle(style_value)

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
            generator_type=self.__class__.__name__,
            generator_version="1.0.0",
        )

        return Suggestion(
            original_text=original_text,
            suggested_text=suggested_text,
            suggestion_type=suggestion_type,
            confidence=confidence,
            diff=diff,
            metadata=metadata,
            style=self._detect_style(context.docstring).value,
            copy_paste_ready=True,
        )

    def _create_fallback_suggestion(
        self, context: SuggestionContext, reason: str
    ) -> Suggestion:
        """Create a low-confidence fallback suggestion."""
        return self._create_suggestion(
            context,
            getattr(context.docstring, "raw_text", "") if context.docstring else "",
            f"Unable to generate specific raises fix: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.RAISES_UPDATE,
        )

    def _generic_raises_fix(self, context: SuggestionContext) -> Suggestion:
        """Generic raises fix for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown raises issue type: {context.issue.issue_type}"
        )
