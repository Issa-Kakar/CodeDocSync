"""
Parameter suggestion generator for handling parameter-related documentation issues.

This module specializes in generating suggestions for parameter mismatches,
missing parameters, type inconsistencies, and other parameter-specific problems.
"""

import re
from typing import Any

from ...parser.ast_parser import FunctionParameter
from ...parser.docstring_models import DocstringParameter
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


class ParameterSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for parameter-related issues."""

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate parameter fix suggestion."""
        issue = context.issue

        if issue.issue_type == "parameter_name_mismatch":
            return self._fix_parameter_name(context)
        elif issue.issue_type == "parameter_missing":
            return self._add_missing_parameter(context)
        elif issue.issue_type == "parameter_type_mismatch":
            return self._fix_parameter_type(context)
        elif issue.issue_type == "parameter_count_mismatch":
            return self._fix_parameter_count(context)
        elif issue.issue_type == "parameter_order_different":
            return self._fix_parameter_order(context)
        elif issue.issue_type == "undocumented_kwargs":
            return self._add_kwargs_documentation(context)
        else:
            # Fallback for unknown parameter issues
            return self._generic_parameter_fix(context)

    def _fix_parameter_name(self, context: SuggestionContext) -> Suggestion:
        """Fix mismatched parameter name."""
        function = context.function
        docstring = context.docstring

        # Extract mismatched parameter info from issue details
        actual_params = self._get_function_parameters(function)
        documented_params = self._get_documented_parameters(docstring)

        # Find the mismatch
        mismatched_pairs = self._find_parameter_mismatches(
            actual_params, documented_params
        )

        if not mismatched_pairs:
            return self._create_fallback_suggestion(
                context, "Could not identify parameter mismatch"
            )

        # Generate corrected docstring
        corrected_docstring = self._generate_corrected_parameter_docstring(
            context, mismatched_pairs
        )

        # Create suggestion
        suggestion = self._create_suggestion(
            context,
            corrected_docstring,
            f"Fix parameter name mismatch: {', '.join([f'{old} → {new}' for old, new in mismatched_pairs])}",
            confidence=0.95,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

        return suggestion

    def _add_missing_parameter(self, context: SuggestionContext) -> Suggestion:
        """Add missing parameter documentation."""
        function = context.function
        docstring = context.docstring

        # Find missing parameters
        actual_params = self._get_function_parameters(function)
        documented_params = self._get_documented_parameters(docstring)

        documented_names = {p.name for p in documented_params}
        missing_params = [p for p in actual_params if p.name not in documented_names]

        # Handle special cases
        missing_params = self._filter_special_parameters(missing_params, function)

        if not missing_params:
            return self._create_fallback_suggestion(
                context, "No missing parameters found"
            )

        # Generate docstring with missing parameters
        updated_docstring = self._add_parameters_to_docstring(context, missing_params)

        missing_names = [p.name for p in missing_params]
        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Add missing parameter documentation: {', '.join(missing_names)}",
            confidence=0.9,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

        return suggestion

    def _fix_parameter_type(self, context: SuggestionContext) -> Suggestion:
        """Fix parameter type mismatch."""
        function = context.function
        docstring = context.docstring

        # Get actual and documented types
        actual_params = self._get_function_parameters(function)
        documented_params = self._get_documented_parameters(docstring)

        # Find type mismatches
        type_fixes = []
        param_map = {p.name: p for p in documented_params}

        for actual_param in actual_params:
            if actual_param.name in param_map:
                doc_param = param_map[actual_param.name]
                if self._types_differ(actual_param.type_annotation, doc_param.type_str):
                    type_fixes.append((actual_param, doc_param))

        if not type_fixes:
            return self._create_fallback_suggestion(context, "No type mismatches found")

        # Generate corrected docstring
        corrected_docstring = self._fix_parameter_types_in_docstring(
            context, type_fixes
        )

        fix_descriptions = [
            f"{param.name}: {param.type_annotation}" for param, _ in type_fixes
        ]
        suggestion = self._create_suggestion(
            context,
            corrected_docstring,
            f"Fix parameter types: {', '.join(fix_descriptions)}",
            confidence=0.85,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

        return suggestion

    def _fix_parameter_count(self, context: SuggestionContext) -> Suggestion:
        """Fix parameter count mismatch."""
        actual_params = self._get_function_parameters(context.function)
        documented_params = self._get_documented_parameters(context.docstring)

        # Filter out special parameters for more accurate count
        filtered_actual = self._filter_special_parameters(
            actual_params, context.function
        )

        if len(filtered_actual) > len(documented_params):
            # More actual parameters - add missing ones
            return self._add_missing_parameter(context)
        else:
            # More documented parameters - remove extras or fix names
            return self._remove_extra_documented_parameters(context)

    def _fix_parameter_order(self, context: SuggestionContext) -> Suggestion:
        """Fix parameter order mismatch."""
        actual_params = self._get_function_parameters(context.function)
        documented_params = self._get_documented_parameters(context.docstring)

        # Reorder documented parameters to match function signature
        reordered_params = self._reorder_documented_parameters(
            actual_params, documented_params
        )

        # Generate docstring with correct order
        corrected_docstring = self._generate_reordered_docstring(
            context, reordered_params
        )

        suggestion = self._create_suggestion(
            context,
            corrected_docstring,
            "Reorder parameters to match function signature",
            confidence=0.8,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

        return suggestion

    def _add_kwargs_documentation(self, context: SuggestionContext) -> Suggestion:
        """Add documentation for **kwargs parameters."""
        function = context.function

        # Find **kwargs parameter
        kwargs_param = None
        for param in function.signature.parameters:
            if param.name.startswith("**"):
                kwargs_param = param
                break

        if not kwargs_param:
            return self._create_fallback_suggestion(
                context, "No **kwargs parameter found"
            )

        # Generate docstring with kwargs documentation
        updated_docstring = self._add_kwargs_to_docstring(context, kwargs_param)

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Add documentation for {kwargs_param.name}",
            confidence=0.7,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

        return suggestion

    def _get_function_parameters(self, function: Any) -> list[FunctionParameter]:
        """Extract function parameters, handling various function types."""
        if hasattr(function, "signature") and hasattr(function.signature, "parameters"):
            return list(function.signature.parameters)
        return []

    def _get_documented_parameters(self, docstring: Any) -> list[DocstringParameter]:
        """Extract documented parameters."""
        if hasattr(docstring, "parameters"):
            return list(docstring.parameters)
        return []

    def _filter_special_parameters(
        self, params: list[FunctionParameter], function: Any
    ) -> list[FunctionParameter]:
        """Filter out special parameters like 'self', 'cls' for accurate comparison."""
        filtered = []

        for param in params:
            # Skip 'self' for instance methods
            if (
                param.name == "self"
                and hasattr(function.signature, "is_method")
                and function.signature.is_method
            ):
                continue

            # Skip 'cls' for class methods
            if param.name == "cls" and self._is_classmethod(function):
                continue

            filtered.append(param)

        return filtered

    def _is_classmethod(self, function: Any) -> bool:
        """Check if function is a classmethod."""
        if hasattr(function.signature, "decorators"):
            return "classmethod" in function.signature.decorators
        return False

    def _find_parameter_mismatches(
        self,
        actual_params: list[FunctionParameter],
        documented_params: list[DocstringParameter],
    ) -> list[tuple]:
        """Find parameter name mismatches using fuzzy matching."""
        from rapidfuzz import fuzz

        mismatches = []
        actual_names = {p.name for p in actual_params}
        documented_names = {p.name for p in documented_params}

        # Find documented parameters that don't exist in function
        for doc_param in documented_params:
            if doc_param.name not in actual_names:
                # Find best match in actual parameters
                best_match = None
                best_score = 0

                for actual_param in actual_params:
                    if actual_param.name not in documented_names:
                        score = fuzz.ratio(doc_param.name, actual_param.name)
                        if (
                            score > best_score and score > 60
                        ):  # Threshold for similarity
                            best_score = int(score)
                            best_match = actual_param.name

                if best_match:
                    mismatches.append((doc_param.name, best_match))

        return mismatches

    def _types_differ(
        self, actual_type: str | None, documented_type: str | None
    ) -> bool:
        """Check if types are significantly different."""
        if not actual_type or not documented_type:
            return bool(actual_type) != bool(documented_type)

        # Normalize types for comparison
        actual_normalized = self._normalize_type(actual_type)
        documented_normalized = self._normalize_type(documented_type)

        return actual_normalized != documented_normalized

    def _normalize_type(self, type_str: str) -> str:
        """Normalize type string for comparison."""
        # Remove whitespace and convert to lowercase
        normalized = re.sub(r"\s+", "", type_str.lower())

        # Handle common equivalencies
        equivalencies = {
            "list[str]": "list",
            "dict[str,any]": "dict",
            "optional[str]": "str",
            "union[str,none]": "str",
        }

        return equivalencies.get(normalized, normalized)

    def _generate_corrected_parameter_docstring(
        self, context: SuggestionContext, mismatched_pairs: list[tuple]
    ) -> str:
        """Generate docstring with corrected parameter names."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        # Update parameter names
        corrected_params = []
        if (
            hasattr(docstring, "parameters")
            and docstring is not None
            and docstring.parameters
        ):
            for param in docstring.parameters:
                # Check if this parameter needs to be renamed
                new_name = param.name
                for old_name, correct_name in mismatched_pairs:
                    if param.name == old_name:
                        new_name = correct_name
                        break

                corrected_param = DocstringParameter(
                    name=new_name,
                    type_str=param.type_annotation,
                    description=param.description,
                    is_optional=param.is_optional,
                    default_value=param.default_value,
                )
                corrected_params.append(corrected_param)

        # Generate new docstring
        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=corrected_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _add_parameters_to_docstring(
        self, context: SuggestionContext, missing_params: list[FunctionParameter]
    ) -> str:
        """Add missing parameters to existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        # Create DocstringParameter objects for missing parameters
        new_doc_params = []
        for param in missing_params:
            doc_param = DocstringParameter(
                name=param.name,
                type_str=param.type_annotation,
                description=self._generate_parameter_description(param),
                is_optional=not param.is_required,
                default_value=param.default_value,
            )
            new_doc_params.append(doc_param)

        # Combine with existing parameters
        existing_params = getattr(docstring, "parameters", [])
        all_params = existing_params + new_doc_params

        # Generate updated docstring
        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=all_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _generate_parameter_description(self, param: FunctionParameter) -> str:
        """Generate a basic description for a parameter."""
        # Basic description based on parameter name and type
        base_desc = f"Description for {param.name}"

        if param.type_annotation:
            base_desc += f" ({param.type_annotation})"

        if param.default_value:
            base_desc += f", defaults to {param.default_value}"

        return base_desc

    def _detect_style(self, docstring: Any) -> str:
        """Detect docstring style from parsed docstring."""
        if hasattr(docstring, "format"):
            # Return the string format directly
            return str(docstring.format.value)

        return "google"  # Default fallback

    def _get_style_enum(self, style_str: str) -> DocstringStyle:
        """Convert style string to DocstringStyle enum."""
        style_map = {
            "google": DocstringStyle.GOOGLE,
            "numpy": DocstringStyle.NUMPY,
            "sphinx": DocstringStyle.SPHINX,
            "rest": DocstringStyle.REST,
            "auto_detect": DocstringStyle.AUTO_DETECT,
            "unknown": DocstringStyle.GOOGLE,  # Default unknown to Google style
        }
        return style_map.get(style_str, DocstringStyle.GOOGLE)

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
            style=self._detect_style(context.docstring),
            copy_paste_ready=True,
        )

    def _create_fallback_suggestion(
        self, context: SuggestionContext, reason: str
    ) -> Suggestion:
        """Create a low-confidence fallback suggestion."""
        # Get original text, handling both RawDocstring and ParsedDocstring
        original_text = ""
        if context.docstring:
            if hasattr(context.docstring, "raw_text"):
                original_text = context.docstring.raw_text
            elif isinstance(context.docstring, str):
                original_text = context.docstring

        # If still empty, create a placeholder
        if not original_text:
            original_text = '"""TODO: Add docstring"""'

        return self._create_suggestion(
            context,
            original_text,  # This will be the suggested text
            f"Unable to generate specific fix: {reason}",
            0.1,  # confidence
            SuggestionType.PARAMETER_UPDATE,
        )

    def _fix_parameter_types_in_docstring(
        self, context: SuggestionContext, type_fixes: list[tuple]
    ) -> str:
        """Fix parameter types in docstring."""
        # Implementation similar to _generate_corrected_parameter_docstring
        # but focused on type corrections
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        # Update parameter types
        updated_params = []
        if (
            hasattr(docstring, "parameters")
            and docstring is not None
            and docstring.parameters
        ):
            param_fixes = {
                actual.name: actual.type_annotation for actual, _ in type_fixes
            }

            for param in docstring.parameters:
                if param.name in param_fixes:
                    updated_param = DocstringParameter(
                        name=param.name,
                        type_str=param_fixes[param.name],
                        description=param.description,
                        is_optional=param.is_optional,
                        default_value=param.default_value,
                    )
                    updated_params.append(updated_param)
                else:
                    updated_params.append(param)

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=updated_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _remove_extra_documented_parameters(
        self, context: SuggestionContext
    ) -> Suggestion:
        """Remove extra documented parameters that don't exist in function."""
        actual_params = self._get_function_parameters(context.function)
        documented_params = self._get_documented_parameters(context.docstring)

        actual_names = {p.name for p in actual_params}
        valid_params = [p for p in documented_params if p.name in actual_names]

        # Generate docstring with only valid parameters
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        corrected_docstring = template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=valid_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

        removed_names = [
            p.name for p in documented_params if p.name not in actual_names
        ]

        return self._create_suggestion(
            context,
            corrected_docstring,
            f"Remove extra documented parameters: {', '.join(removed_names)}",
            confidence=0.8,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

    def _reorder_documented_parameters(
        self,
        actual_params: list[FunctionParameter],
        documented_params: list[DocstringParameter],
    ) -> list[DocstringParameter]:
        """Reorder documented parameters to match function signature order."""
        # Create a mapping from name to documented parameter
        doc_param_map = {p.name: p for p in documented_params}

        # Reorder based on actual parameter order
        reordered = []
        for actual_param in actual_params:
            if actual_param.name in doc_param_map:
                reordered.append(doc_param_map[actual_param.name])

        # Add any documented parameters that don't have matches (in original order)
        actual_names = {p.name for p in actual_params}
        for doc_param in documented_params:
            if doc_param.name not in actual_names:
                reordered.append(doc_param)

        return reordered

    def _generate_reordered_docstring(
        self, context: SuggestionContext, reordered_params: list[DocstringParameter]
    ) -> str:
        """Generate docstring with reordered parameters."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=reordered_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _add_kwargs_to_docstring(
        self, context: SuggestionContext, kwargs_param: FunctionParameter
    ) -> str:
        """Add kwargs documentation to docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        style_enum = self._get_style_enum(style)
        template = get_template(style_enum, max_line_length=self.config.max_line_length)

        # Create documentation for kwargs
        kwargs_doc = DocstringParameter(
            name=kwargs_param.name,
            type_str="Any",
            description="Additional keyword arguments",
            is_optional=True,
        )

        # Add to existing parameters
        existing_params = getattr(docstring, "parameters", [])
        all_params = existing_params + [kwargs_doc]

        return template.render_complete_docstring(
            summary=getattr(docstring, "summary", ""),
            description=getattr(docstring, "description", None),
            parameters=all_params,
            returns=getattr(docstring, "returns", None),
            raises=getattr(docstring, "raises", []),
            examples=getattr(docstring, "examples", []),
        )

    def _generic_parameter_fix(self, context: SuggestionContext) -> Suggestion:
        """Generic parameter fix for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown parameter issue type: {context.issue.issue_type}"
        )
