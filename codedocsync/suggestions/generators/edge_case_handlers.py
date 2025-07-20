"""
Edge case handlers for special Python constructs and documentation scenarios.

This module handles special Python constructs like property decorators, class methods,
overloaded functions, and other edge cases that require specialized documentation approaches.
"""

import ast
from dataclasses import dataclass

from ...parser.docstring_models import DocstringParameter, DocstringReturns
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
class SpecialConstruct:
    """Information about a special Python construct."""

    construct_type: str
    requires_special_handling: bool
    documentation_style: str
    confidence: float


class SpecialConstructAnalyzer:
    """Analyze functions for special Python constructs."""

    def analyze_function(self, function) -> list[SpecialConstruct]:
        """Analyze function for special constructs that need edge case handling."""
        constructs = []

        if not hasattr(function, "signature"):
            return constructs

        sig = function.signature
        decorators = getattr(sig, "decorators", []) or []

        # Check for property decorators
        if "property" in decorators:
            constructs.append(
                SpecialConstruct(
                    construct_type="property_getter",
                    requires_special_handling=True,
                    documentation_style="property",
                    confidence=1.0,
                )
            )
        elif any(d.endswith(".setter") for d in decorators if isinstance(d, str)):
            constructs.append(
                SpecialConstruct(
                    construct_type="property_setter",
                    requires_special_handling=True,
                    documentation_style="property_setter",
                    confidence=1.0,
                )
            )
        elif any(d.endswith(".deleter") for d in decorators if isinstance(d, str)):
            constructs.append(
                SpecialConstruct(
                    construct_type="property_deleter",
                    requires_special_handling=True,
                    documentation_style="property_deleter",
                    confidence=1.0,
                )
            )

        # Check for class/static methods
        if "classmethod" in decorators:
            constructs.append(
                SpecialConstruct(
                    construct_type="classmethod",
                    requires_special_handling=True,
                    documentation_style="classmethod",
                    confidence=1.0,
                )
            )
        elif "staticmethod" in decorators:
            constructs.append(
                SpecialConstruct(
                    construct_type="staticmethod",
                    requires_special_handling=True,
                    documentation_style="staticmethod",
                    confidence=1.0,
                )
            )

        # Check for overloaded functions
        if "overload" in decorators:
            constructs.append(
                SpecialConstruct(
                    construct_type="overload",
                    requires_special_handling=True,
                    documentation_style="overload",
                    confidence=1.0,
                )
            )

        # Check for async functions
        if getattr(sig, "is_async", False):
            constructs.append(
                SpecialConstruct(
                    construct_type="async_function",
                    requires_special_handling=True,
                    documentation_style="async",
                    confidence=1.0,
                )
            )

        # Check for generator functions
        source_code = getattr(function, "source_code", "")
        if self._is_generator_function(source_code):
            constructs.append(
                SpecialConstruct(
                    construct_type="generator",
                    requires_special_handling=True,
                    documentation_style="generator",
                    confidence=0.9,
                )
            )

        # Check for context managers
        if self._is_context_manager(function):
            constructs.append(
                SpecialConstruct(
                    construct_type="context_manager",
                    requires_special_handling=True,
                    documentation_style="context_manager",
                    confidence=0.8,
                )
            )

        # Check for magic methods
        function_name = getattr(sig, "name", "")
        if self._is_magic_method(function_name):
            constructs.append(
                SpecialConstruct(
                    construct_type="magic_method",
                    requires_special_handling=True,
                    documentation_style="magic_method",
                    confidence=1.0,
                )
            )

        return constructs

    def _is_generator_function(self, source_code: str) -> bool:
        """Check if function is a generator."""
        if not source_code:
            return False

        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Yield, ast.YieldFrom)):
                    return True
        except SyntaxError:
            pass

        return False

    def _is_context_manager(self, function) -> bool:
        """Check if function is designed as a context manager."""
        if not hasattr(function, "signature"):
            return False

        function_name = getattr(function.signature, "name", "")
        return function_name in ("__enter__", "__exit__", "__aenter__", "__aexit__")

    def _is_magic_method(self, function_name: str) -> bool:
        """Check if function is a magic method."""
        return function_name.startswith("__") and function_name.endswith("__")


class PropertyMethodHandler:
    """Handle property methods (getter, setter, deleter)."""

    def handle_property_getter(self, context: SuggestionContext) -> Suggestion:
        """Handle property getter documentation."""
        function = context.function
        docstring = context.docstring

        # Property getters should not document parameters (except self)
        # They should focus on return value
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Generate appropriate return documentation
        return_type = self._infer_property_type(function)
        property_desc = self._generate_property_description(function)

        return_doc = None
        if return_type or property_desc:
            return_doc = DocstringReturns(
                type_str=return_type or "",
                description=property_desc or "The property value",
            )

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", property_desc)
                if docstring
                else property_desc
            ),
            description=None,  # Keep it concise for properties
            parameters=[],  # No parameters for property getters
            returns=return_doc,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=[],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Format as property getter documentation",
            confidence=0.9,
            suggestion_type=SuggestionType.FULL_DOCSTRING,
        )

    def handle_property_setter(self, context: SuggestionContext) -> Suggestion:
        """Handle property setter documentation."""
        function = context.function
        docstring = context.docstring

        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Property setters typically have one parameter (value)
        parameters = []
        if hasattr(function, "signature") and hasattr(function.signature, "parameters"):
            for param in function.signature.parameters:
                if param.name not in ("self", "cls"):
                    parameters.append(
                        DocstringParameter(
                            name=param.name,
                            type_str=param.type_annotation or "",
                            description=f"The new {param.name} value",
                            is_optional=not param.is_required,
                        )
                    )

        setter_desc = self._generate_setter_description(function)

        suggested_docstring = template.render_complete_docstring(
            summary=setter_desc,
            description=None,
            parameters=parameters,
            returns=None,  # Setters typically don't return values
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=[],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Format as property setter documentation",
            confidence=0.9,
            suggestion_type=SuggestionType.FULL_DOCSTRING,
        )

    def handle_property_deleter(self, context: SuggestionContext) -> Suggestion:
        """Handle property deleter documentation."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        deleter_desc = "Delete the property value."

        suggested_docstring = template.render_complete_docstring(
            summary=deleter_desc,
            description=None,
            parameters=[],  # No parameters for deleters
            returns=None,  # Deleters don't return values
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=[],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Format as property deleter documentation",
            confidence=0.9,
            suggestion_type=SuggestionType.FULL_DOCSTRING,
        )

    def _infer_property_type(self, function) -> str | None:
        """Infer property type from function signature or name."""
        if hasattr(function, "signature") and hasattr(
            function.signature, "return_annotation"
        ):
            return function.signature.return_annotation

        # Try to infer from name
        function_name = (
            getattr(function.signature, "name", "")
            if hasattr(function, "signature")
            else ""
        )
        if "count" in function_name.lower():
            return "int"
        elif "name" in function_name.lower():
            return "str"
        elif "is_" in function_name.lower() or function_name.lower().startswith("has_"):
            return "bool"

        return None

    def _generate_property_description(self, function) -> str:
        """Generate description for property based on function name."""
        function_name = (
            getattr(function.signature, "name", "property")
            if hasattr(function, "signature")
            else "property"
        )

        # Remove common prefixes
        clean_name = function_name
        if clean_name.startswith("get_"):
            clean_name = clean_name[4:]

        # Convert snake_case to words
        words = clean_name.replace("_", " ")

        return f"Get the {words}"

    def _generate_setter_description(self, function) -> str:
        """Generate description for setter based on function name."""
        function_name = (
            getattr(function.signature, "name", "property")
            if hasattr(function, "signature")
            else "property"
        )

        # Remove common prefixes/suffixes
        clean_name = function_name
        if clean_name.startswith("set_"):
            clean_name = clean_name[4:]

        # Convert snake_case to words
        words = clean_name.replace("_", " ")

        return f"Set the {words}"

    def _detect_style(self, docstring) -> DocstringStyle:
        """Detect docstring style."""
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
        return DocstringStyle.GOOGLE

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


class ClassMethodHandler:
    """Handle class and static methods."""

    def handle_classmethod(self, context: SuggestionContext) -> Suggestion:
        """Handle classmethod documentation."""
        function = context.function
        docstring = context.docstring

        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Filter out 'cls' parameter from documentation
        parameters = []
        if hasattr(function, "signature") and hasattr(function.signature, "parameters"):
            for param in function.signature.parameters:
                if param.name != "cls":
                    parameters.append(
                        DocstringParameter(
                            name=param.name,
                            type_str=param.type_annotation or "",
                            description=f"Description for {param.name}",
                            is_optional=not param.is_required,
                        )
                    )

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", "Class method")
                if docstring
                else "Class method"
            ),
            description=getattr(docstring, "description", None) if docstring else None,
            parameters=parameters,
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Format as classmethod documentation (exclude 'cls' parameter)",
            confidence=0.9,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

    def handle_staticmethod(self, context: SuggestionContext) -> Suggestion:
        """Handle staticmethod documentation."""
        function = context.function
        docstring = context.docstring

        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Static methods document all parameters normally
        parameters = []
        if hasattr(function, "signature") and hasattr(function.signature, "parameters"):
            for param in function.signature.parameters:
                parameters.append(
                    DocstringParameter(
                        name=param.name,
                        type_str=param.type_annotation or "",
                        description=f"Description for {param.name}",
                        is_optional=not param.is_required,
                    )
                )

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", "Static method")
                if docstring
                else "Static method"
            ),
            description=getattr(docstring, "description", None) if docstring else None,
            parameters=parameters,
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Format as staticmethod documentation",
            confidence=0.9,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

    def _detect_style(self, docstring) -> DocstringStyle:
        """Detect docstring style."""
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
        return DocstringStyle.GOOGLE

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


class EdgeCaseSuggestionGenerator(BaseSuggestionGenerator):
    """Main edge case handler that delegates to specialized handlers."""

    def __init__(self, config):
        super().__init__(config)
        self.analyzer = SpecialConstructAnalyzer()
        self.property_handler = PropertyMethodHandler()
        self.classmethod_handler = ClassMethodHandler()

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate suggestions for edge cases."""
        # Analyze function for special constructs
        constructs = self.analyzer.analyze_function(context.function)

        # Handle each construct type
        for construct in constructs:
            if construct.construct_type == "property_getter":
                return self.property_handler.handle_property_getter(context)
            elif construct.construct_type == "property_setter":
                return self.property_handler.handle_property_setter(context)
            elif construct.construct_type == "property_deleter":
                return self.property_handler.handle_property_deleter(context)
            elif construct.construct_type == "classmethod":
                return self.classmethod_handler.handle_classmethod(context)
            elif construct.construct_type == "staticmethod":
                return self.classmethod_handler.handle_staticmethod(context)
            elif construct.construct_type == "async_function":
                return self._handle_async_function(context)
            elif construct.construct_type == "generator":
                return self._handle_generator_function(context)
            elif construct.construct_type == "magic_method":
                return self._handle_magic_method(context)
            elif construct.construct_type == "overload":
                return self._handle_overloaded_function(context)

        # No special constructs found - return generic improvement
        return self._handle_generic_edge_case(context)

    def _handle_async_function(self, context: SuggestionContext) -> Suggestion:
        """Handle async function documentation."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Add note about async nature to description
        current_desc = getattr(docstring, "description", None) if docstring else None
        async_note = "This is an async function and should be awaited."

        if current_desc:
            enhanced_desc = f"{current_desc}\n\nNote: {async_note}"
        else:
            enhanced_desc = async_note

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", "Async function")
                if docstring
                else "Async function"
            ),
            description=enhanced_desc,
            parameters=getattr(docstring, "parameters", []) if docstring else [],
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Add async function documentation note",
            confidence=0.8,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

    def _handle_generator_function(self, context: SuggestionContext) -> Suggestion:
        """Handle generator function documentation."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        # Update return documentation for generators
        generator_return = DocstringReturns(
            type_str="Generator", description="Generator yielding values"
        )

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", "Generator function")
                if docstring
                else "Generator function"
            ),
            description=getattr(docstring, "description", None) if docstring else None,
            parameters=getattr(docstring, "parameters", []) if docstring else [],
            returns=generator_return,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Update generator function documentation",
            confidence=0.9,
            suggestion_type=SuggestionType.RETURN_UPDATE,
        )

    def _handle_magic_method(self, context: SuggestionContext) -> Suggestion:
        """Handle magic method documentation."""
        function = context.function
        function_name = (
            getattr(function.signature, "name", "")
            if hasattr(function, "signature")
            else ""
        )

        # Special descriptions for common magic methods
        magic_descriptions = {
            "__init__": "Initialize a new instance",
            "__str__": "Return string representation",
            "__repr__": "Return detailed string representation",
            "__len__": "Return length of the object",
            "__eq__": "Check equality with another object",
            "__lt__": "Check if less than another object",
            "__gt__": "Check if greater than another object",
            "__call__": "Make the object callable",
            "__enter__": "Enter the context manager",
            "__exit__": "Exit the context manager",
            "__iter__": "Return an iterator",
            "__next__": "Return the next item from iterator",
            "__getitem__": "Get item by key/index",
            "__setitem__": "Set item by key/index",
            "__delitem__": "Delete item by key/index",
        }

        description = magic_descriptions.get(
            function_name, f"Magic method {function_name}"
        )

        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        suggested_docstring = template.render_complete_docstring(
            summary=description,
            description=getattr(docstring, "description", None) if docstring else None,
            parameters=getattr(docstring, "parameters", []) if docstring else [],
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            f"Document magic method {function_name}",
            confidence=0.8,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

    def _handle_overloaded_function(self, context: SuggestionContext) -> Suggestion:
        """Handle overloaded function documentation."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        template = get_template(style, max_line_length=88)

        overload_note = "This function is overloaded. See individual overload signatures for specific parameter types."

        suggested_docstring = template.render_complete_docstring(
            summary=(
                getattr(docstring, "summary", "Overloaded function")
                if docstring
                else "Overloaded function"
            ),
            description=overload_note,
            parameters=[],  # Overloads typically don't document parameters in the main docstring
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

        return self._create_suggestion(
            context,
            suggested_docstring,
            "Document overloaded function",
            confidence=0.7,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

    def _handle_generic_edge_case(self, context: SuggestionContext) -> Suggestion:
        """Handle generic edge cases that don't fit other categories."""
        return self._create_fallback_suggestion(
            context, "No specific edge case handling available"
        )

    def _detect_style(self, docstring) -> DocstringStyle:
        """Detect docstring style."""
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
        return DocstringStyle.GOOGLE

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
            f"Unable to generate specific edge case fix: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )
