"""
Parameter suggestion generator for handling parameter-related documentation issues.

This module specializes in generating suggestions for parameter mismatches,
missing parameters, type inconsistencies, and other parameter-specific problems.
"""

import logging
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

logger = logging.getLogger(__name__)


class ParameterSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for parameter-related issues."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the generator."""
        super().__init__(config)
        self._used_rag = False  # Track if RAG was used

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate parameter fix suggestion."""
        # Store context for RAG-enhanced description generation
        self._current_context = context
        self._used_rag = False  # Reset for each generation

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
        elif issue.issue_type == "missing_params":
            return self._add_missing_parameters_complete(context)
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

    def _add_missing_parameters_complete(
        self, context: SuggestionContext
    ) -> Suggestion:
        """Add all parameters when docstring is missing or incomplete."""
        # Check if we have any existing docstring
        if not context.docstring:
            # No docstring at all - create a complete one
            return self._create_new_docstring_with_parameters(context)
        else:
            # Has docstring but missing parameters
            return self._add_missing_parameter(context)

    def _create_new_docstring_with_parameters(
        self, context: SuggestionContext
    ) -> Suggestion:
        """Create a new docstring with all parameters documented."""
        function = context.function
        style = context.project_style

        # Get all function parameters
        params = self._get_function_parameters(function)

        # Build docstring based on style
        if style == "google":
            docstring_lines = ['"""TODO: Add function description.']
            if params:
                docstring_lines.append("")
                docstring_lines.append("Args:")
                for param in params:
                    desc = self._generate_parameter_description(param)
                    if param.type_annotation:
                        docstring_lines.append(
                            f"    {param.name} ({param.type_annotation}): {desc}"
                        )
                    else:
                        docstring_lines.append(f"    {param.name}: {desc}")

            # Add return section if there's a return type
            if (
                function.signature.return_type
                and function.signature.return_type != "None"
            ):
                docstring_lines.append("")
                docstring_lines.append("Returns:")
                docstring_lines.append(
                    f"    {function.signature.return_type}: TODO: Describe return value."
                )

            docstring_lines.append('"""')
            suggested_text = "\n".join(docstring_lines)
        else:
            # For other styles, use a simple fallback
            suggested_text = '"""TODO: Add docstring with parameters."""'

        # Create the suggestion with empty original text handled properly
        return self._create_suggestion(
            context,
            suggested_text,
            f"Add complete docstring with {len(params)} parameters",
            confidence=0.9,
            suggestion_type=SuggestionType.PARAMETER_UPDATE,
        )

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
            "List[str]": "list",
            "Dict[str,any]": "dict",
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
                    type_str=param.type_str,
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
        # Check if we have context with RAG examples
        if hasattr(self, "_current_context") and self._current_context:
            rag_desc = self._generate_rag_enhanced_description(param)
            if rag_desc:
                logger.info(
                    f"Using RAG-enhanced description for parameter '{param.name}'"
                )
                self._used_rag = True  # Mark that RAG was used
                return rag_desc
            else:
                logger.debug(
                    f"No RAG enhancement available for parameter '{param.name}', using basic description"
                )
        else:
            logger.debug(f"No RAG context available for parameter '{param.name}'")

        # Fallback to basic description
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

        # Handle missing docstring case
        if not original_text:
            original_text = '"""TODO: Add docstring"""'

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

    def _generate_rag_enhanced_description(
        self, param: FunctionParameter
    ) -> str | None:
        """Generate parameter description using RAG examples with semantic matching."""
        if not self._current_context or not self._current_context.related_functions:
            return None

        # Extract parameter descriptions from RAG examples with type information
        param_patterns = self._extract_parameter_patterns_from_examples(
            param.name,
            self._current_context.related_functions,
            param.type_annotation,  # Pass type for better matching
        )

        if not param_patterns:
            return None

        # Log when we're using RAG enhancement
        logger.info(
            f"RAG: Found {len(param_patterns)} similar patterns for parameter '{param.name}'"
        )

        # Use multiple patterns to create a comprehensive description
        if len(param_patterns) >= 2:
            # Combine insights from top patterns
            description = self._synthesize_description_from_patterns(
                param, param_patterns[:3]
            )
        else:
            # Use single best pattern
            best_pattern = param_patterns[0]
            description = self._adapt_description_for_parameter(
                param,
                best_pattern["description"],
                best_pattern.get("original_param_name", ""),
            )

        # Add type information if not already present
        if param.type_annotation and param.type_annotation not in description:
            description = self._enhance_description_with_type(
                description, param.type_annotation
            )

        # Add default value if available
        if param.default_value and "defaults to" not in description.lower():
            description = f"{description}, defaults to {param.default_value}"

        logger.debug(
            f"RAG: Generated description for '{param.name}': {description[:100]}..."
        )

        return description

    def _extract_parameter_patterns_from_examples(
        self,
        param_name: str,
        examples: list[dict[str, Any]],
        param_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract parameter description patterns from RAG examples using semantic similarity."""
        patterns = []

        for example in examples:
            docstring_content = example.get("docstring", "")
            if not docstring_content:
                continue

            # Extract ALL parameters from the example, not just exact matches
            all_params = self._extract_all_parameters_from_docstring(docstring_content)

            for param in all_params:
                # Calculate similarity score for parameter names
                name_similarity = self._calculate_name_similarity(
                    param_name, param["name"]
                )

                # Also consider type similarity if type is provided
                type_similarity = 1.0
                if param_type and param.get("type"):
                    type_similarity = self._calculate_type_similarity(
                        param_type, param["type"]
                    )

                # Combined score: weighted average of name and type similarity
                combined_similarity = 0.6 * name_similarity + 0.4 * type_similarity

                # Only include if similarity is above threshold
                if combined_similarity > 0.3:  # Lower threshold for more matches
                    patterns.append(
                        {
                            "description": param["description"],
                            "type_hint": param.get("type", ""),
                            "similarity": example.get("similarity", 0.0)
                            * combined_similarity,
                            "source": example.get("signature", ""),
                            "original_param_name": param["name"],
                        }
                    )

        # Sort by similarity score
        patterns.sort(key=lambda x: x["similarity"], reverse=True)

        # Take top patterns
        return patterns[:5]  # Return up to 5 best matches

    def _extract_parameter_from_docstring(
        self, param_name: str, docstring_content: str
    ) -> list[dict[str, Any]]:
        """Extract parameter description from a docstring."""
        matches = []

        # Google style pattern
        google_pattern = rf"{param_name}\s*\(([^)]*)\):\s*(.+?)(?=\n\s*\w+.*:|$)"
        google_match = re.search(
            google_pattern, docstring_content, re.DOTALL | re.MULTILINE
        )
        if google_match:
            matches.append(
                {
                    "type": google_match.group(1).strip(),
                    "description": google_match.group(2).strip(),
                }
            )

        # NumPy style pattern
        numpy_pattern = rf"{param_name}\s*:\s*([^\n]+)\n\s+(.+?)(?=\n\w+|$)"
        numpy_match = re.search(
            numpy_pattern, docstring_content, re.DOTALL | re.MULTILINE
        )
        if numpy_match:
            matches.append(
                {
                    "type": numpy_match.group(1).strip(),
                    "description": numpy_match.group(2).strip(),
                }
            )

        # Sphinx style pattern
        sphinx_pattern = rf":param\s+{param_name}:\s*(.+?)(?=:param|:type|:return|$)"
        sphinx_match = re.search(sphinx_pattern, docstring_content, re.DOTALL)
        if sphinx_match:
            matches.append({"description": sphinx_match.group(1).strip()})

        # Also check for similar parameter names (fuzzy matching)
        # Note: Removed to prevent recursion issues

        # Return empty matches if nothing found
        if not matches:
            return []

        return matches

    def _extract_all_parameters_from_docstring(
        self, docstring_content: str
    ) -> list[dict[str, Any]]:
        """Extract all parameters from a docstring regardless of name."""
        params = []

        # Google style pattern - match any parameter
        google_pattern = r"(\w+)\s*\(([^)]*)\):\s*(.+?)(?=\n\s*\w+.*:|$|\n\n)"
        for match in re.finditer(
            google_pattern, docstring_content, re.DOTALL | re.MULTILINE
        ):
            params.append(
                {
                    "name": match.group(1).strip(),
                    "type": match.group(2).strip(),
                    "description": match.group(3).strip(),
                }
            )

        # NumPy style pattern - match any parameter
        numpy_pattern = r"^(\w+)\s*:\s*([^\n]+)\n\s+(.+?)(?=^\w+|^\s*$)"
        for match in re.finditer(
            numpy_pattern, docstring_content, re.DOTALL | re.MULTILINE
        ):
            # Avoid duplicates
            param_name = match.group(1).strip()
            if not any(p["name"] == param_name for p in params):
                params.append(
                    {
                        "name": param_name,
                        "type": match.group(2).strip(),
                        "description": match.group(3).strip(),
                    }
                )

        # Sphinx style pattern - match any parameter
        sphinx_pattern = r":param\s+(\w+):\s*(.+?)(?=:param|:type|:return|$)"
        sphinx_type_pattern = r":type\s+(\w+):\s*(.+?)(?=:param|:type|:return|$)"

        # First extract param descriptions
        sphinx_params = {}
        for match in re.finditer(sphinx_pattern, docstring_content, re.DOTALL):
            param_name = match.group(1).strip()
            sphinx_params[param_name] = {
                "name": param_name,
                "description": match.group(2).strip(),
            }

        # Then add types
        for match in re.finditer(sphinx_type_pattern, docstring_content, re.DOTALL):
            param_name = match.group(1).strip()
            if param_name in sphinx_params:
                sphinx_params[param_name]["type"] = match.group(2).strip()

        # Add Sphinx params to the list
        for param_name, param_info in sphinx_params.items():
            if not any(p["name"] == param_name for p in params):
                params.append(param_info)

        return params

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity between parameter names."""
        # Normalize names
        name1_normalized = name1.lower().replace("_", " ")
        name2_normalized = name2.lower().replace("_", " ")

        # Exact match
        if name1_normalized == name2_normalized:
            return 1.0

        # Check if one contains the other
        if name1_normalized in name2_normalized or name2_normalized in name1_normalized:
            return 0.8

        # Check for common patterns and abbreviations
        common_equivalents = [
            # Common abbreviations
            ("msg", "message"),
            ("val", "value"),
            ("param", "parameter"),
            ("config", "configuration"),
            ("cfg", "config"),
            ("dir", "directory"),
            ("func", "function"),
            ("fn", "function"),
            ("cb", "callback"),
            ("idx", "index"),
            ("num", "number"),
            ("str", "string"),
            ("dict", "dictionary"),
            ("obj", "object"),
            ("req", "request"),
            ("res", "response"),
            ("resp", "response"),
            ("err", "error"),
            ("exc", "exception"),
            ("ref", "reference"),
            ("impl", "implementation"),
            ("mgr", "manager"),
            ("conn", "connection"),
            ("src", "source"),
            ("dst", "destination"),
            ("dest", "destination"),
            ("args", "arguments"),
            ("kwargs", "keyword_arguments"),
            ("opts", "options"),
            ("env", "environment"),
            ("ctx", "context"),
            ("auth", "authentication"),
            ("id", "identifier"),
            ("uid", "user_id"),
            ("pwd", "password"),
            ("addr", "address"),
            ("desc", "description"),
            ("max", "maximum"),
            ("min", "minimum"),
            ("tmp", "temporary"),
            ("temp", "temporary"),
            ("proc", "process"),
            ("db", "database"),
            # Semantic equivalents
            ("query", "search"),
            ("find", "search"),
            ("lookup", "search"),
            ("content", "text"),
            ("data", "content"),
            ("info", "information"),
            ("stats", "statistics"),
            ("file", "path"),
            ("filepath", "path"),
            ("filename", "name"),
            ("uri", "url"),
            ("endpoint", "url"),
            ("api_url", "endpoint"),
            ("count", "number"),
            ("amount", "number"),
            ("total", "sum"),
            ("enable", "activate"),
            ("disable", "deactivate"),
            ("start", "begin"),
            ("stop", "end"),
            ("create", "make"),
            ("delete", "remove"),
            ("update", "modify"),
            ("fetch", "get"),
            ("retrieve", "get"),
            ("store", "save"),
            ("load", "read"),
            ("write", "save"),
        ]

        # Check common equivalents with word boundary awareness
        name1_parts = name1_normalized.split("_")
        name2_parts = name2_normalized.split("_")

        for equiv1, equiv2 in common_equivalents:
            # Check exact word matches
            if (equiv1 in name1_parts and equiv2 in name2_parts) or (
                equiv2 in name1_parts and equiv1 in name2_parts
            ):
                return 0.8
            # Check substring matches (less confident)
            elif (equiv1 in name1_normalized and equiv2 in name2_normalized) or (
                equiv2 in name1_normalized and equiv1 in name2_normalized
            ):
                return 0.7

        # Token-based similarity (split by underscore)
        tokens1 = set(name1_parts)
        tokens2 = set(name2_parts)
        if tokens1 and tokens2:
            # Check for shared tokens
            shared_tokens = tokens1.intersection(tokens2)
            if shared_tokens:
                token_similarity = len(shared_tokens) / max(len(tokens1), len(tokens2))
                return min(0.6, token_similarity)

        # Character overlap similarity (fallback)
        set1 = set(name1_normalized)
        set2 = set(name2_normalized)
        if set1 and set2:
            overlap = len(set1.intersection(set2)) / len(set1.union(set2))
            return min(0.4, overlap)

        return 0.0

    def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate similarity between type annotations."""
        # Normalize types
        type1_normalized = type1.lower().strip()
        type2_normalized = type2.lower().strip()

        # Exact match
        if type1_normalized == type2_normalized:
            return 1.0

        # Check if both are string types
        string_types = {"str", "string", "text", "unicode"}
        if type1_normalized in string_types and type2_normalized in string_types:
            return 0.9

        # Check if both are numeric types
        numeric_types = {"int", "integer", "float", "number", "decimal", "numeric"}
        if type1_normalized in numeric_types and type2_normalized in numeric_types:
            return 0.9

        # Check if both are collection types
        collection_types = {"list", "tuple", "set", "array", "sequence", "iterable"}
        if any(t in type1_normalized for t in collection_types) and any(
            t in type2_normalized for t in collection_types
        ):
            return 0.8

        # Check if both are mapping types
        mapping_types = {"dict", "dictionary", "mapping", "map"}
        if any(t in type1_normalized for t in mapping_types) and any(
            t in type2_normalized for t in mapping_types
        ):
            return 0.8

        # Check if both are boolean types
        bool_types = {"bool", "boolean"}
        if type1_normalized in bool_types and type2_normalized in bool_types:
            return 1.0

        # Check for optional/union patterns
        if (
            ("optional" in type1_normalized and "optional" in type2_normalized)
            or ("union" in type1_normalized and "union" in type2_normalized)
            or ("|" in type1_normalized and "|" in type2_normalized)
        ):
            return 0.6

        # Default: no similarity
        return 0.0

    def _synthesize_description_from_patterns(
        self, param: FunctionParameter, patterns: list[dict[str, Any]]
    ) -> str:
        """Synthesize a description from multiple RAG patterns with intelligent merging."""
        descriptions = [p["description"] for p in patterns if p.get("description")]

        if not descriptions:
            return self._generate_fallback_description(param)

        if len(descriptions) == 1:
            # Single pattern - adapt it directly
            return self._adapt_description_for_parameter(
                param, descriptions[0], patterns[0].get("original_param_name", "")
            )

        # Multiple patterns - extract and combine insights
        # Extract core concepts from each description
        core_concepts = []
        for i, desc in enumerate(descriptions[:3]):  # Use top 3 patterns
            core_meaning = self._extract_core_meaning(
                desc, patterns[i].get("original_param_name", "")
            )
            if (
                core_meaning and len(core_meaning) > 10
            ):  # Filter out trivial descriptions
                core_concepts.append(
                    {
                        "meaning": core_meaning,
                        "similarity": patterns[i].get("similarity", 0.0),
                        "original": desc,
                    }
                )

        if not core_concepts:
            # Fallback to best description
            return self._adapt_description_for_parameter(
                param, descriptions[0], patterns[0].get("original_param_name", "")
            )

        # Synthesize based on similarity and content quality
        if core_concepts[0]["similarity"] > 0.8:
            # High confidence - use the best match
            synthesized = core_concepts[0]["meaning"]
        else:
            # Combine insights from multiple sources
            synthesized = self._merge_concepts(core_concepts, param)

        # Ensure proper adaptation for the parameter
        synthesized = self._adapt_based_on_semantics(param.name, synthesized)

        return self._ensure_proper_grammar(synthesized, param.name)

    def _merge_concepts(
        self, concepts: list[dict[str, Any]], param: FunctionParameter
    ) -> str:
        """Merge multiple concept descriptions into a coherent description."""
        # Extract key phrases and patterns
        key_phrases = []
        for concept in concepts:
            meaning = concept["meaning"]

            # Extract action verbs and key nouns
            action_match = re.search(
                r"\b(specifies?|defines?|indicates?|represents?|contains?|provides?)\s+(.+)",
                meaning,
                re.IGNORECASE,
            )
            if action_match:
                key_phrases.append(action_match.group(2))
            else:
                key_phrases.append(meaning)

        # Find commonalities
        if len(set(key_phrases)) == 1:
            # All concepts agree
            return key_phrases[0]

        # Combine unique insights
        combined = []
        seen_concepts = set()

        for phrase in key_phrases:
            # Normalize for comparison
            normalized = phrase.lower().strip(".,")
            if normalized not in seen_concepts:
                seen_concepts.add(normalized)
                combined.append(phrase)

        # Create a coherent description
        if len(combined) == 1:
            return combined[0]
        elif len(combined) == 2:
            # Choose the more specific one
            return combined[0] if len(combined[0]) > len(combined[1]) else combined[1]
        else:
            # Use the most specific/detailed one
            return max(combined, key=len)

    def _adapt_description_for_parameter(
        self, param: FunctionParameter, description: str, original_param_name: str
    ) -> str:
        """Adapt a description from one parameter to another with intelligent context extraction."""
        if not description:
            return self._generate_fallback_description(param)

        # Extract the core meaning from the description, removing parameter-specific references
        core_description = self._extract_core_meaning(description, original_param_name)

        # Adapt based on parameter name semantics
        adapted = self._adapt_based_on_semantics(param.name, core_description)

        # Replace original parameter name references if they still exist
        if original_param_name and original_param_name != param.name:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(original_param_name), re.IGNORECASE)
            adapted = pattern.sub(param.name, adapted)

        # Ensure proper capitalization and grammar
        adapted = self._ensure_proper_grammar(adapted, param.name)

        return adapted

    def _extract_core_meaning(self, description: str, original_param_name: str) -> str:
        """Extract the core meaning from a description, removing parameter-specific references."""
        # Remove common parameter prefixes
        core = description

        # Remove parameter name mentions at the beginning
        patterns = [
            rf"^The {re.escape(original_param_name)} parameter[,.]?\s*",
            rf"^The {re.escape(original_param_name)}[,.]?\s*",
            rf"^{re.escape(original_param_name)}[,:]?\s*",
        ]

        for pattern in patterns:
            core = re.sub(pattern, "", core, flags=re.IGNORECASE)

        # Extract the actual meaning (what the parameter represents/does)
        # Look for key phrases that indicate purpose
        purpose_patterns = [
            r"(?:used to|for|that|which)\s+(.+)",
            r"(?:represents?|specifies?|indicates?|defines?)\s+(.+)",
            r"(?:containing|with)\s+(.+)",
        ]

        for pattern in purpose_patterns:
            match = re.search(pattern, core, re.IGNORECASE)
            if match:
                core = match.group(1)
                break

        # Clean up and ensure it's a complete phrase
        core = core.strip()
        if core and not core[0].isupper():
            core = core[0].upper() + core[1:]

        return core

    def _adapt_based_on_semantics(self, param_name: str, core_description: str) -> str:
        """Adapt description based on parameter name semantics."""
        # Common parameter name patterns and their typical purposes
        semantic_patterns = {
            # Data parameters
            r"^input_data$": "Input data to be processed",
            r"^output_data$": "Processed output data",
            r".*data$": "Data to be processed",
            r"^data_.*": "Data for {suffix}",
            r".*_data$": "{prefix} data",
            # Configuration parameters
            r"^config_file$": "Path to configuration file",
            r".*config$": "Configuration settings",
            r"^config_.*": "Configuration for {suffix}",
            r".*_config$": "{prefix} configuration",
            # File/path parameters
            r".*(?:file|path)$": "Path to the file",
            r"^(?:file|path)_.*": "Path to {suffix}",
            # URL/endpoint parameters
            r"^api_url$": "API endpoint URL",
            r".*url$": "URL endpoint",
            r"^url_.*": "URL for {suffix}",
            r".*_url$": "{prefix} URL",
            # Timeout/duration parameters
            r".*timeout$": "Maximum time to wait in seconds",
            r".*_timeout$": "Timeout for {prefix} in seconds",
            # Boolean flags
            r"^enable_cache$": "Whether to enable caching",
            r"^verbose$": "Whether to enable verbose output",
            r"^(?:is|has|should|enable|disable)_.*": "Whether to {suffix}",
            r".*_(?:enabled|disabled)$": "Whether {prefix} is enabled",
            # Count/number parameters
            r"^retry_count$": "Number of retry attempts",
            r"^(?:num|count|max|min)_.*": "Number of {suffix}",
            r".*_(?:count|num|number)$": "Number of {prefix}",
            # Result/output parameters
            r".*results?$": "Results to process",
            r"^results?_.*": "Results from {suffix}",
            # Format parameters
            r".*format$": "Output format specification",
            r"^format_.*": "Format for {suffix}",
            # Name/ID parameters
            r".*name$": "Name identifier",
            r".*_id$": "{prefix} identifier",
            # Connection/database parameters
            r"^connection_string$": "Database connection string",
            r"^table_name$": "Name of the database table",
            r".*_rows?$": "{prefix} rows",
        }

        # Try to match parameter name with semantic patterns
        for pattern, template in semantic_patterns.items():
            match = re.match(pattern, param_name, re.IGNORECASE)
            if match:
                # Extract prefix/suffix for template
                if "{prefix}" in template:
                    prefix = re.sub(
                        r"_(?:data|config|file|path|url|timeout|enabled|disabled|count|num|number)$",
                        "",
                        param_name,
                    )
                    prefix = prefix.replace("_", " ")
                    description = template.format(prefix=prefix)
                elif "{suffix}" in template:
                    suffix = re.sub(
                        r"^(?:data|config|file|path|url|num|count|max|min|is|has|should|enable|disable)_",
                        "",
                        param_name,
                    )
                    suffix = suffix.replace("_", " ")
                    description = template.format(suffix=suffix)
                else:
                    description = template

                # Only use semantic pattern - core description often creates duplication
                return description

        # If no semantic pattern matches, use core description
        if core_description:
            return core_description

        # Fallback to parameter name based description
        return f"The {param_name.replace('_', ' ')}"

    def _ensure_proper_grammar(self, description: str, param_name: str) -> str:
        """Ensure proper grammar and capitalization in the description."""
        if not description:
            return f"The {param_name} parameter"

        # Ensure first letter is capitalized
        description = description.strip()
        if description and description[0].islower():
            description = description[0].upper() + description[1:]

        # Ensure it ends with proper punctuation
        if description and description[-1] not in ".!?":
            description += "."

        # Clean up any double spaces or awkward phrasing
        description = re.sub(r"\s+", " ", description)
        description = re.sub(r"\bthe the\b", "the", description, flags=re.IGNORECASE)
        description = re.sub(r"\ba a\b", "a", description, flags=re.IGNORECASE)

        # Don't add redundant parameter mentions - the description should stand on its own

        return description

    def _generate_fallback_description(self, param: FunctionParameter) -> str:
        """Generate a fallback description based on parameter information."""
        desc_parts = []

        # Add type information if available
        if param.type_annotation:
            type_desc = self._get_type_description(param.type_annotation)
            desc_parts.append(type_desc)

        # Add requirement information
        if param.is_required:
            desc_parts.append("Required")
        else:
            desc_parts.append("Optional")

        # Add default value if available
        if param.default_value and param.default_value != "None":
            desc_parts.append(f"defaults to {param.default_value}")

        # Construct description based on parameter name and available info
        if desc_parts:
            # Generate a more contextual description
            if param.type_annotation:
                base_desc = desc_parts[0]  # Type description
                if param.default_value and param.default_value != "None":
                    return f"{base_desc.capitalize()}, {desc_parts[-1]}"
                else:
                    return base_desc.capitalize()
            else:
                # No type info, use parameter name intelligently
                param_words = param.name.replace("_", " ")
                return param_words.capitalize()

        # Final fallback - just use parameter name as phrase
        return param.name.replace("_", " ").capitalize()

    def _get_type_description(self, type_annotation: str) -> str:
        """Get a human-readable description for a type annotation."""
        type_descriptions = {
            "str": "string value",
            "int": "integer value",
            "float": "floating-point number",
            "bool": "boolean flag",
            "list": "list of values",
            "dict": "dictionary/mapping",
            "tuple": "tuple of values",
            "set": "set of unique values",
            "Any": "any type",
            "None": "None value",
        }

        # Check for exact match first
        if type_annotation in type_descriptions:
            return type_descriptions[type_annotation]

        # Check for generic types (e.g., list[str], dict[str, int])
        for base_type, desc in type_descriptions.items():
            if type_annotation.startswith(f"{base_type}["):
                return desc

        # Check for Union types
        if "|" in type_annotation:
            types = [t.strip() for t in type_annotation.split("|")]
            if "None" in types:
                types.remove("None")
                if len(types) == 1:
                    return f"optional {self._get_type_description(types[0])}"
            return f"one of: {', '.join(types)}"

        # Default
        return type_annotation

    def _enhance_description_with_type(
        self, description: str, type_annotation: str
    ) -> str:
        """Add type information to a description if not already present."""
        # Check if type is already mentioned
        type_lower = type_annotation.lower()
        desc_lower = description.lower()

        # Common type keywords to check
        type_keywords = {
            "str": ["string", "text", "str"],
            "int": ["integer", "number", "int"],
            "float": ["float", "decimal", "number"],
            "bool": ["boolean", "bool", "true", "false"],
            "list": ["list", "array", "sequence"],
            "dict": ["dictionary", "dict", "mapping"],
        }

        # Check if type is already described
        type_mentioned = False
        for base_type, keywords in type_keywords.items():
            if base_type in type_lower:
                if any(keyword in desc_lower for keyword in keywords):
                    type_mentioned = True
                    break

        # Add type if not mentioned
        if not type_mentioned and type_lower not in desc_lower:
            if description.endswith("."):
                description = description[:-1] + f" ({type_annotation})."
            else:
                description = f"{description} ({type_annotation})"

        return description
