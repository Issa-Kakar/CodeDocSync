"""
Rule-based analysis engine for fast documentation validation.

Performance target: <5ms per function
Confidence threshold: Issues with confidence > 0.9 skip LLM analysis
"""

import re
import time
from typing import List, Optional

from codedocsync.matcher import MatchedPair
from codedocsync.parser import (
    ParsedFunction,
    FunctionParameter,
    ParsedDocstring,
    DocstringParameter,
)
from .models import (
    RuleCheckResult,
    InconsistencyIssue,
)


class RuleEngine:
    """
    Fast rule-based analysis engine for documentation consistency.

    This engine performs deterministic checks on function-documentation pairs
    to catch common inconsistencies without requiring LLM analysis.
    """

    def __init__(
        self, enabled_rules: Optional[List[str]] = None, performance_mode: bool = False
    ):
        """
        Initialize the rule engine.

        Args:
            enabled_rules: List of rule names to run (None = all rules)
            performance_mode: Skip expensive rules for better performance
        """
        self.enabled_rules = enabled_rules
        self.performance_mode = performance_mode

        # Map rule names to methods
        self._rule_methods = {
            # Structural rules
            "parameter_names": self._check_parameter_names,
            "parameter_types": self._check_parameter_types,
            "parameter_count": self._check_parameter_count,
            "return_type": self._check_return_type,
            # Completeness rules
            "missing_params": self._check_missing_params,
            "missing_returns": self._check_missing_returns,
            "missing_raises": self._check_missing_raises,
            "undocumented_kwargs": self._check_undocumented_kwargs,
            # Consistency rules
            "default_mismatches": self._check_default_mismatches,
            "parameter_order": self._check_parameter_order,
        }

    def check_matched_pair(
        self, pair: MatchedPair, confidence_threshold: float = 0.9
    ) -> List[InconsistencyIssue]:
        """
        Main entry point for rule checking.

        Args:
            pair: The matched function-documentation pair to analyze
            confidence_threshold: Minimum confidence to skip LLM analysis

        Returns:
            List of detected inconsistency issues
        """
        if not self._validate_pair(pair):
            return []

        all_issues = []
        enabled_rules = self.enabled_rules or list(self._rule_methods.keys())

        for rule_name in enabled_rules:
            if rule_name in self._rule_methods:
                try:
                    result = self._rule_methods[rule_name](pair)
                    all_issues.extend(result.issues)
                except Exception:
                    # Log error but don't break analysis
                    continue

        # Sort by severity (critical first)
        all_issues.sort(key=lambda x: x.severity_weight, reverse=True)
        return all_issues

    def _validate_pair(self, pair: MatchedPair) -> bool:
        """Validate that the matched pair can be analyzed."""
        if not pair.function:
            return False

        # Check if we have either inline docstring or separate documentation
        has_docstring = (pair.function.docstring is not None) or (
            pair.docstring is not None
        )

        return has_docstring

    def _get_parsed_docstring(self, pair: MatchedPair) -> Optional[ParsedDocstring]:
        """Get the parsed docstring from the matched pair."""
        # Try function's docstring first
        if pair.function.docstring:
            if isinstance(pair.function.docstring, ParsedDocstring):
                return pair.function.docstring

        # Try separate docstring field
        if pair.docstring:
            if isinstance(pair.docstring, ParsedDocstring):
                return pair.docstring

        return None

    def _get_function_params(self, function: ParsedFunction) -> List[FunctionParameter]:
        """Extract parameters from function signature, excluding self/cls."""
        params = []
        for param in function.signature.parameters:
            # Skip 'self' and 'cls' parameters
            if param.name in ("self", "cls"):
                continue
            params.append(param)
        return params

    def _get_doc_params(
        self, docstring: Optional[ParsedDocstring]
    ) -> List[DocstringParameter]:
        """Extract parameters from parsed docstring."""
        if not docstring:
            return []
        return docstring.parameters

    # STRUCTURAL RULES

    def _check_parameter_names(self, pair: MatchedPair) -> RuleCheckResult:
        """Check parameter name consistency."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        # Create sets of parameter names for comparison
        func_param_names = {p.name.lstrip("*") for p in func_params}
        doc_param_names = {p.name.lstrip("*") for p in doc_params}

        # Find mismatched names
        missing_in_docs = func_param_names - doc_param_names
        extra_in_docs = doc_param_names - func_param_names

        # Generate issues for missing parameters
        for param_name in missing_in_docs:
            issues.append(
                InconsistencyIssue(
                    issue_type="parameter_missing",
                    severity="critical",
                    description=f"Parameter '{param_name}' in function signature is not documented",
                    suggestion=f"Add '{param_name}' to the docstring parameters section",
                    line_number=pair.function.line_number,
                    confidence=1.0,
                    details={"missing_param": param_name},
                )
            )

        # Generate issues for extra documented parameters
        for param_name in extra_in_docs:
            issues.append(
                InconsistencyIssue(
                    issue_type="parameter_name_mismatch",
                    severity="critical",
                    description=f"Parameter '{param_name}' documented but not in function signature",
                    suggestion=f"Remove '{param_name}' from docstring or check for naming mismatch",
                    line_number=pair.function.line_number,
                    confidence=0.95,
                    details={"extra_param": param_name},
                )
            )

        return RuleCheckResult(
            rule_name="parameter_names",
            passed=len(issues) == 0,
            confidence=1.0 if not issues else 0.95,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_parameter_types(self, pair: MatchedPair) -> RuleCheckResult:
        """Validate type annotations match documented types."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        # Create mapping of parameter names to types
        func_types = {
            p.name: p.type_annotation for p in func_params if p.type_annotation
        }
        doc_types = {p.name: p.type_str for p in doc_params if p.type_str}

        # Check type consistency for parameters that have both
        for param_name in func_types:
            if param_name in doc_types:
                func_type = self._normalize_type_string(func_types[param_name])
                doc_type = self._normalize_type_string(doc_types[param_name])

                if not self._types_equivalent(func_type, doc_type):
                    issues.append(
                        InconsistencyIssue(
                            issue_type="parameter_type_mismatch",
                            severity="high",
                            description=f"Type mismatch for parameter '{param_name}': "
                            f"function has '{func_types[param_name]}', "
                            f"documentation has '{doc_types[param_name]}'",
                            suggestion=f"Update docstring type for '{param_name}' to match "
                            f"function annotation: '{func_types[param_name]}'",
                            line_number=pair.function.line_number,
                            confidence=0.9,
                            details={
                                "param_name": param_name,
                                "function_type": func_types[param_name],
                                "doc_type": doc_types[param_name],
                            },
                        )
                    )

        return RuleCheckResult(
            rule_name="parameter_types",
            passed=len(issues) == 0,
            confidence=0.9 if not issues else 0.8,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_parameter_count(self, pair: MatchedPair) -> RuleCheckResult:
        """Ensure all parameters are documented."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        func_count = len(func_params)
        doc_count = len(doc_params)

        if func_count != doc_count:
            issues.append(
                InconsistencyIssue(
                    issue_type="parameter_count_mismatch",
                    severity="critical",
                    description=f"Parameter count mismatch: function has {func_count} parameters, "
                    f"documentation has {doc_count}",
                    suggestion=f"Ensure all {func_count} function parameters are documented",
                    line_number=pair.function.line_number,
                    confidence=1.0,
                    details={"function_count": func_count, "doc_count": doc_count},
                )
            )

        return RuleCheckResult(
            rule_name="parameter_count",
            passed=len(issues) == 0,
            confidence=1.0,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_return_type(self, pair: MatchedPair) -> RuleCheckResult:
        """Validate return type consistency."""
        start_time = time.time()
        issues = []

        func_return_type = pair.function.signature.return_type
        docstring = self._get_parsed_docstring(pair)
        doc_return_type = (
            docstring.returns.type_str if docstring and docstring.returns else None
        )

        # Check for missing return documentation
        if func_return_type and func_return_type != "None" and not doc_return_type:
            issues.append(
                InconsistencyIssue(
                    issue_type="missing_returns",
                    severity="high",
                    description=f"Function has return type '{func_return_type}' but no return documentation",
                    suggestion=f"Add Returns section documenting the '{func_return_type}' return value",
                    line_number=pair.function.line_number,
                    confidence=0.95,
                    details={"function_return_type": func_return_type},
                )
            )

        # Check for return type mismatch
        elif func_return_type and doc_return_type:
            func_normalized = self._normalize_type_string(func_return_type)
            doc_normalized = self._normalize_type_string(doc_return_type)

            if not self._types_equivalent(func_normalized, doc_normalized):
                issues.append(
                    InconsistencyIssue(
                        issue_type="return_type_mismatch",
                        severity="high",
                        description=f"Return type mismatch: function returns '{func_return_type}', "
                        f"documentation says '{doc_return_type}'",
                        suggestion=f"Update documentation return type to match function: '{func_return_type}'",
                        line_number=pair.function.line_number,
                        confidence=0.9,
                        details={
                            "function_return_type": func_return_type,
                            "doc_return_type": doc_return_type,
                        },
                    )
                )

        return RuleCheckResult(
            rule_name="return_type",
            passed=len(issues) == 0,
            confidence=0.95 if not issues else 0.9,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # COMPLETENESS RULES

    def _check_missing_params(self, pair: MatchedPair) -> RuleCheckResult:
        """Find undocumented parameters."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        documented_names = {p.name.lstrip("*") for p in doc_params}

        for param in func_params:
            param_name = param.name.lstrip("*")
            if param_name not in documented_names:
                issues.append(
                    InconsistencyIssue(
                        issue_type="missing_params",
                        severity="critical",
                        description=f"Parameter '{param.name}' is not documented",
                        suggestion=f"Add documentation for parameter '{param.name}':\n"
                        f"    {param.name}: [TODO: Add description]",
                        line_number=pair.function.line_number,
                        confidence=1.0,
                        details={"missing_param": param.name},
                    )
                )

        return RuleCheckResult(
            rule_name="missing_params",
            passed=len(issues) == 0,
            confidence=1.0,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_missing_returns(self, pair: MatchedPair) -> RuleCheckResult:
        """Check return documentation completeness."""
        start_time = time.time()
        issues = []

        func_return_type = pair.function.signature.return_type
        docstring = self._get_parsed_docstring(pair)
        has_return_doc = docstring and docstring.returns

        # Only flag if function has a return type that's not None
        if func_return_type and func_return_type != "None" and not has_return_doc:
            issues.append(
                InconsistencyIssue(
                    issue_type="missing_returns",
                    severity="high",
                    description="Function has return type but no return documentation",
                    suggestion=f"Add Returns section:\n    Returns:\n        {func_return_type}: [TODO: Describe return value]",
                    line_number=pair.function.line_number,
                    confidence=0.95,
                    details={"return_type": func_return_type},
                )
            )

        return RuleCheckResult(
            rule_name="missing_returns",
            passed=len(issues) == 0,
            confidence=0.95,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_missing_raises(self, pair: MatchedPair) -> RuleCheckResult:
        """Find undocumented exceptions (basic check)."""
        start_time = time.time()
        issues = []

        # This is a simplified check - full implementation would require AST analysis
        # For now, just pass all cases (keeping it simple for performance)
        # In a full implementation, we'd parse the function body for 'raise' statements

        return RuleCheckResult(
            rule_name="missing_raises",
            passed=True,  # Always pass for now
            confidence=0.5,  # Low confidence since we're not doing full analysis
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_undocumented_kwargs(self, pair: MatchedPair) -> RuleCheckResult:
        """Check for undocumented *args and **kwargs."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        documented_names = {p.name for p in doc_params}

        for param in func_params:
            if param.name.startswith("**") and param.name not in documented_names:
                issues.append(
                    InconsistencyIssue(
                        issue_type="undocumented_kwargs",
                        severity="medium",
                        description=f"**kwargs parameter '{param.name}' is not documented",
                        suggestion=f"Add documentation for {param.name}:\n"
                        f"    {param.name}: Additional keyword arguments",
                        line_number=pair.function.line_number,
                        confidence=1.0,
                        details={"missing_kwarg": param.name},
                    )
                )
            elif (
                param.name.startswith("*")
                and not param.name.startswith("**")
                and param.name not in documented_names
            ):
                issues.append(
                    InconsistencyIssue(
                        issue_type="undocumented_kwargs",
                        severity="medium",
                        description=f"*args parameter '{param.name}' is not documented",
                        suggestion=f"Add documentation for {param.name}:\n"
                        f"    {param.name}: Variable length argument list",
                        line_number=pair.function.line_number,
                        confidence=1.0,
                        details={"missing_arg": param.name},
                    )
                )

        return RuleCheckResult(
            rule_name="undocumented_kwargs",
            passed=len(issues) == 0,
            confidence=1.0,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # CONSISTENCY RULES

    def _check_default_mismatches(self, pair: MatchedPair) -> RuleCheckResult:
        """Verify default values match between function and docs."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        # Create mapping of parameter names to default values
        func_defaults = {
            p.name: p.default_value for p in func_params if p.default_value
        }
        doc_defaults = {p.name: p.default_value for p in doc_params if p.default_value}

        # Check consistency
        for param_name in func_defaults:
            if param_name in doc_defaults:
                func_default = str(func_defaults[param_name])
                doc_default = str(doc_defaults[param_name])

                if func_default != doc_default:
                    issues.append(
                        InconsistencyIssue(
                            issue_type="default_mismatches",
                            severity="medium",
                            description=f"Default value mismatch for '{param_name}': "
                            f"function has '{func_default}', docs have '{doc_default}'",
                            suggestion=f"Update docstring default for '{param_name}' to '{func_default}'",
                            line_number=pair.function.line_number,
                            confidence=0.9,
                            details={
                                "param_name": param_name,
                                "function_default": func_default,
                                "doc_default": doc_default,
                            },
                        )
                    )

        return RuleCheckResult(
            rule_name="default_mismatches",
            passed=len(issues) == 0,
            confidence=0.9,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_parameter_order(self, pair: MatchedPair) -> RuleCheckResult:
        """Check if parameter order matches between function and docs."""
        start_time = time.time()
        issues = []

        func_params = self._get_function_params(pair.function)
        doc_params = self._get_doc_params(self._get_parsed_docstring(pair))

        # Get ordered parameter names
        func_order = [p.name for p in func_params]
        doc_order = [p.name for p in doc_params]

        # Check if orders match (ignoring missing parameters)
        common_params = set(func_order) & set(doc_order)
        if len(common_params) > 1:  # Only check if there are multiple common params
            func_common_order = [p for p in func_order if p in common_params]
            doc_common_order = [p for p in doc_order if p in common_params]

            if func_common_order != doc_common_order:
                issues.append(
                    InconsistencyIssue(
                        issue_type="parameter_order_different",
                        severity="medium",
                        description="Parameter order differs between function and documentation",
                        suggestion=f"Reorder docstring parameters to match function: {', '.join(func_common_order)}",
                        line_number=pair.function.line_number,
                        confidence=0.8,
                        details={
                            "function_order": func_common_order,
                            "doc_order": doc_common_order,
                        },
                    )
                )

        return RuleCheckResult(
            rule_name="parameter_order",
            passed=len(issues) == 0,
            confidence=0.8,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # UTILITY METHODS

    def _normalize_type_string(self, type_str: str) -> str:
        """Normalize type string for comparison."""
        if not type_str:
            return ""

        # Remove whitespace and normalize common variations
        normalized = type_str.strip()

        # Common type mappings
        type_mappings = {
            "string": "str",
            "integer": "int",
            "boolean": "bool",
            "float": "float",
            "list": "List",
            "dict": "Dict",
            "tuple": "Tuple",
        }

        for old, new in type_mappings.items():
            normalized = re.sub(rf"\b{old}\b", new, normalized, flags=re.IGNORECASE)

        return normalized

    def _types_equivalent(self, type1: str, type2: str) -> bool:
        """Check if two type strings are equivalent."""
        if not type1 or not type2:
            return type1 == type2

        # Simple equivalence check
        return type1.lower() == type2.lower()
