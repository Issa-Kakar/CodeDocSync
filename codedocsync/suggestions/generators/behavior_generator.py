"""
Behavioral description generator for improving function behavior documentation.

This module specializes in analyzing function code to generate better behavioral
descriptions, identifying patterns, side effects, and operational characteristics.
"""

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..base import BaseSuggestionGenerator
from ..models import (
    Suggestion,
    SuggestionContext,
    SuggestionDiff,
    SuggestionMetadata,
    SuggestionType,
)
from ..templates.base import get_template

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPattern:
    """A detected behavioral pattern in function code."""

    pattern_type: str
    description: str
    confidence: float
    line_numbers: list[int] | None = None
    details: dict[str, Any] | None = None


class BehaviorAnalyzer:
    """Analyze function code to extract behavioral patterns."""

    def __init__(self) -> None:
        self.patterns: list[BehaviorPattern] = []

    def analyze_behavior(
        self, source_code: str, function_name: str = ""
    ) -> list[BehaviorPattern]:
        """Analyze function code for behavioral patterns."""
        patterns: list[BehaviorPattern] = []

        try:
            tree = ast.parse(source_code)
            self._analyze_ast(tree, patterns, function_name)
        except SyntaxError:
            # If we can't parse, provide basic analysis
            patterns.append(
                BehaviorPattern(
                    pattern_type="unknown",
                    description="Function behavior could not be analyzed",
                    confidence=0.1,
                )
            )

        return patterns

    def _analyze_ast(
        self, tree: ast.AST, patterns: list[BehaviorPattern], function_name: str
    ) -> None:
        """Analyze AST for behavioral patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                self._analyze_function_node(node, patterns, function_name)
                break

    def _analyze_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
        function_name: str,
    ) -> None:
        """Analyze a function node for behavioral patterns."""
        # Analyze different aspects of behavior
        self._analyze_control_flow(node, patterns)
        self._analyze_data_operations(node, patterns)
        self._analyze_side_effects(node, patterns)
        self._analyze_error_handling(node, patterns)
        self._analyze_performance_characteristics(node, patterns, function_name)
        self._analyze_function_purpose(node, patterns, function_name)

    def _analyze_control_flow(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
    ) -> None:
        """Analyze control flow patterns."""
        has_loops = False
        has_conditionals = False
        has_early_return = False
        loop_types = set()

        for child in ast.walk(node):
            if isinstance(child, ast.For | ast.While):
                has_loops = True
                loop_types.add(type(child).__name__.lower())
            elif isinstance(child, ast.If):
                has_conditionals = True
            elif isinstance(child, ast.Return) and child != list(ast.walk(node))[-1]:
                has_early_return = True

        if has_loops:
            loop_desc = " and ".join(loop_types)
            patterns.append(
                BehaviorPattern(
                    pattern_type="iteration",
                    description=f"Iterates through data using {loop_desc} loops",
                    confidence=0.9,
                )
            )

        if has_conditionals:
            patterns.append(
                BehaviorPattern(
                    pattern_type="conditional",
                    description="Applies conditional logic based on input parameters",
                    confidence=0.8,
                )
            )

        if has_early_return:
            patterns.append(
                BehaviorPattern(
                    pattern_type="early_exit",
                    description="Returns early when certain conditions are met",
                    confidence=0.9,
                )
            )

    def _analyze_data_operations(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
    ) -> None:
        """Analyze data manipulation patterns."""
        creates_collections = False
        modifies_collections = False
        transforms_data = False

        for child in ast.walk(node):
            if (
                isinstance(child, ast.ListComp)
                or isinstance(child, ast.DictComp)
                or isinstance(child, ast.SetComp)
            ):
                transforms_data = True
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    attr_name = child.func.attr
                    if attr_name in (
                        "append",
                        "extend",
                        "insert",
                        "remove",
                        "pop",
                        "clear",
                    ):
                        modifies_collections = True
                    elif attr_name in ("map", "filter", "reduce", "join", "split"):
                        transforms_data = True
            elif isinstance(child, ast.List | ast.Dict | ast.Set | ast.Tuple):
                if (
                    len(
                        child.elts
                        if hasattr(child, "elts")
                        else child.keys if hasattr(child, "keys") else []
                    )
                    > 0
                ):
                    creates_collections = True

        if creates_collections:
            patterns.append(
                BehaviorPattern(
                    pattern_type="data_creation",
                    description="Creates and populates new data structures",
                    confidence=0.8,
                )
            )

        if modifies_collections:
            patterns.append(
                BehaviorPattern(
                    pattern_type="data_modification",
                    description="Modifies existing data structures in place",
                    confidence=0.9,
                )
            )

        if transforms_data:
            patterns.append(
                BehaviorPattern(
                    pattern_type="data_transformation",
                    description="Transforms data from one format to another",
                    confidence=0.8,
                )
            )

    def _analyze_side_effects(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
    ) -> None:
        """Analyze side effects and external interactions."""
        has_file_operations = False
        has_network_operations = False
        has_database_operations = False
        has_logging = False
        modifies_globals = False

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = ""
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    func_name = child.func.attr

                # Check for common side effect patterns
                if func_name in ("open", "read", "write", "close"):
                    has_file_operations = True
                elif func_name in ("get", "post", "put", "delete", "request"):
                    has_network_operations = True
                elif func_name in ("execute", "query", "commit", "rollback"):
                    has_database_operations = True
                elif func_name in ("log", "debug", "info", "warning", "error", "print"):
                    has_logging = True

            elif isinstance(child, ast.Global) or isinstance(child, ast.Nonlocal):
                modifies_globals = True

        if has_file_operations:
            patterns.append(
                BehaviorPattern(
                    pattern_type="file_io",
                    description="Performs file input/output operations",
                    confidence=0.9,
                )
            )

        if has_network_operations:
            patterns.append(
                BehaviorPattern(
                    pattern_type="network_io",
                    description="Makes network requests or communications",
                    confidence=0.8,
                )
            )

        if has_database_operations:
            patterns.append(
                BehaviorPattern(
                    pattern_type="database_io",
                    description="Interacts with database systems",
                    confidence=0.8,
                )
            )

        if has_logging:
            patterns.append(
                BehaviorPattern(
                    pattern_type="logging",
                    description="Logs information for debugging or monitoring",
                    confidence=0.7,
                )
            )

        if modifies_globals:
            patterns.append(
                BehaviorPattern(
                    pattern_type="global_modification",
                    description="Modifies global or non-local variables",
                    confidence=0.9,
                )
            )

    def _analyze_error_handling(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
    ) -> None:
        """Analyze error handling patterns."""
        has_try_except = False
        validates_input = False
        has_assertions = False

        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_try_except = True
            elif isinstance(child, ast.Assert):
                has_assertions = True
            elif isinstance(child, ast.If):
                # Check if this looks like input validation
                if self._looks_like_validation(child):
                    validates_input = True

        if has_try_except:
            patterns.append(
                BehaviorPattern(
                    pattern_type="error_handling",
                    description="Handles exceptions and error conditions gracefully",
                    confidence=0.9,
                )
            )

        if validates_input:
            patterns.append(
                BehaviorPattern(
                    pattern_type="input_validation",
                    description="Validates input parameters before processing",
                    confidence=0.8,
                )
            )

        if has_assertions:
            patterns.append(
                BehaviorPattern(
                    pattern_type="assertion_checks",
                    description="Uses assertions to verify program state",
                    confidence=0.8,
                )
            )

    def _analyze_performance_characteristics(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
        function_name: str,
    ) -> None:
        """Analyze performance-related characteristics."""
        has_nested_loops = False
        has_recursion = False
        has_caching = False
        complexity_score = 0

        loop_depth = 0
        max_loop_depth = 0

        for child in ast.walk(node):
            if isinstance(child, ast.For | ast.While):
                loop_depth += 1
                max_loop_depth = max(max_loop_depth, loop_depth)
                loop_depth -= 1
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == function_name:
                    has_recursion = True
                elif isinstance(child.func, ast.Attribute) and child.func.attr in (
                    "cache",
                    "memoize",
                    "lru_cache",
                ):
                    has_caching = True

        if max_loop_depth > 1:
            has_nested_loops = True
            complexity_score = max_loop_depth

        if has_nested_loops:
            patterns.append(
                BehaviorPattern(
                    pattern_type="nested_iteration",
                    description=f"Uses nested loops (depth: {max_loop_depth}) which may impact performance",
                    confidence=0.9,
                    details={"complexity_score": complexity_score},
                )
            )

        if has_recursion:
            patterns.append(
                BehaviorPattern(
                    pattern_type="recursive",
                    description="Uses recursive calls to solve the problem",
                    confidence=0.9,
                )
            )

        if has_caching:
            patterns.append(
                BehaviorPattern(
                    pattern_type="caching",
                    description="Implements caching for performance optimization",
                    confidence=0.9,
                )
            )

    def _analyze_function_purpose(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        patterns: list[BehaviorPattern],
        function_name: str,
    ) -> None:
        """Analyze function purpose based on name and structure."""
        name_lower = function_name.lower()

        purpose_patterns = {
            "get": "Retrieves and returns data",
            "set": "Sets or updates data values",
            "create": "Creates new objects or data structures",
            "build": "Constructs complex objects or structures",
            "calculate": "Performs mathematical calculations",
            "compute": "Computes results from input data",
            "process": "Processes input data through a series of operations",
            "transform": "Transforms data from one format to another",
            "convert": "Converts data between different types or formats",
            "validate": "Validates data against specified criteria",
            "check": "Checks conditions or data integrity",
            "verify": "Verifies data correctness or conditions",
            "parse": "Parses structured data or text",
            "format": "Formats data for display or output",
            "serialize": "Serializes data for storage or transmission",
            "deserialize": "Deserializes data from storage format",
            "load": "Loads data from external sources",
            "save": "Saves data to external storage",
            "update": "Updates existing data or state",
            "delete": "Removes or deletes data",
            "find": "Searches for and locates specific data",
            "search": "Searches through data collections",
            "filter": "Filters data based on criteria",
            "sort": "Sorts data in a specific order",
        }

        for keyword, description in purpose_patterns.items():
            if keyword in name_lower:
                patterns.append(
                    BehaviorPattern(
                        pattern_type="purpose", description=description, confidence=0.7
                    )
                )
                break

    def _looks_like_validation(self, if_node: ast.If) -> bool:
        """Check if an if statement looks like input validation."""
        # Simple heuristic: check for common validation patterns
        if isinstance(if_node.test, ast.Compare):
            # Look for None checks, type checks, etc.
            return True
        elif isinstance(if_node.test, ast.Call):
            if isinstance(if_node.test.func, ast.Name):
                func_name = if_node.test.func.id
                if func_name in ("isinstance", "hasattr", "len", "bool"):
                    return True
        return False


class BehaviorSuggestionGenerator(BaseSuggestionGenerator):
    """Generate suggestions for behavioral descriptions."""

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the generator."""
        super().__init__(config)
        self._used_rag = False  # Track if RAG was used

    def generate(self, context: SuggestionContext) -> Suggestion:
        """Generate behavioral description improvements."""
        # Store context for RAG-enhanced description generation
        self._current_context = context
        self._used_rag = False  # Reset for each generation

        issue = context.issue

        if issue.issue_type == "description_outdated":
            return self._improve_outdated_description(context)
        elif issue.issue_type == "description_vague":
            return self._improve_vague_description(context)
        elif issue.issue_type == "missing_behavior_description":
            return self._add_behavior_description(context)
        elif issue.issue_type == "side_effects_undocumented":
            return self._add_side_effects_documentation(context)
        else:
            return self._generic_behavior_improvement(context)

    def _improve_outdated_description(self, context: SuggestionContext) -> Suggestion:
        """Improve outdated function description."""
        return self._enhance_description(context, "outdated description")

    def _improve_vague_description(self, context: SuggestionContext) -> Suggestion:
        """Improve vague function description."""
        return self._enhance_description(context, "vague description")

    def _add_behavior_description(self, context: SuggestionContext) -> Suggestion:
        """Add missing behavioral description."""
        return self._enhance_description(context, "missing behavior description")

    def _add_side_effects_documentation(self, context: SuggestionContext) -> Suggestion:
        """Add documentation for side effects."""
        return self._enhance_description(
            context, "undocumented side effects", focus_side_effects=True
        )

    def _enhance_description(
        self,
        context: SuggestionContext,
        issue_reason: str,
        focus_side_effects: bool = False,
    ) -> Suggestion:
        """Enhance function description based on code analysis."""
        # Store context for RAG
        self._current_context = context
        self._used_rag = False

        function = context.function
        docstring = context.docstring

        # Get function name for analysis
        function_name = (
            getattr(function.signature, "name", "function")
            if hasattr(function, "signature")
            else "function"
        )

        # Try RAG-enhanced generation first
        if context.related_functions:
            rag_description = self._generate_rag_enhanced_behavior(
                context, focus_side_effects
            )
            if rag_description:
                self._used_rag = True
                # Create suggestion with RAG description
                return self._create_behavior_suggestion(context, rag_description)

        # Fallback to code analysis
        source_code = getattr(function, "source_code", "")
        if not source_code:
            return self._create_fallback_suggestion(
                context, "Source code not available for behavioral analysis"
            )

        analyzer = BehaviorAnalyzer()
        patterns = analyzer.analyze_behavior(source_code, function_name)

        if not patterns:
            return self._create_fallback_suggestion(
                context, "No behavioral patterns detected"
            )

        # Generate enhanced description
        enhanced_description = self._generate_enhanced_description(
            patterns, function_name, docstring, focus_side_effects
        )

        if not enhanced_description:
            return self._create_fallback_suggestion(
                context, "Could not generate improved description"
            )

        # Create updated docstring
        updated_docstring = self._update_description_in_docstring(
            context, enhanced_description
        )

        # Calculate confidence based on pattern strength
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)

        suggestion = self._create_suggestion(
            context,
            updated_docstring,
            f"Improve function description: {issue_reason}",
            confidence=min(0.8, avg_confidence),
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

        return suggestion

    def _generate_enhanced_description(
        self,
        patterns: list[BehaviorPattern],
        function_name: str,
        docstring: Any | None,
        focus_side_effects: bool = False,
    ) -> str:
        """Generate enhanced description based on behavioral patterns."""
        # Get current description
        current_summary = getattr(docstring, "summary", "") if docstring else ""

        # Group patterns by type
        purpose_patterns = [p for p in patterns if p.pattern_type == "purpose"]
        operation_patterns = [
            p
            for p in patterns
            if p.pattern_type
            in ("iteration", "conditional", "data_transformation", "data_creation")
        ]
        side_effect_patterns = [
            p
            for p in patterns
            if p.pattern_type
            in (
                "file_io",
                "network_io",
                "database_io",
                "logging",
                "global_modification",
            )
        ]
        performance_patterns = [
            p
            for p in patterns
            if p.pattern_type in ("nested_iteration", "recursive", "caching")
        ]

        # Build description components
        description_parts = []

        # Start with purpose if available
        if purpose_patterns:
            description_parts.append(purpose_patterns[0].description)
        elif current_summary:
            description_parts.append(current_summary)
        else:
            # Generate basic purpose from function name
            basic_purpose = self._generate_basic_purpose(function_name)
            description_parts.append(basic_purpose)

        # Add operational details
        if operation_patterns:
            operations = [
                p.description.lower() for p in operation_patterns[:2]
            ]  # Limit to avoid verbosity
            if operations:
                description_parts.append(f"The function {', and '.join(operations)}")

        # Add side effects if focusing on them or if they're significant
        if focus_side_effects or (
            side_effect_patterns and len(side_effect_patterns) > 1
        ):
            side_effects = [p.description.lower() for p in side_effect_patterns[:3]]
            if side_effects:
                description_parts.append(
                    f"Side effects include: {', '.join(side_effects)}"
                )

        # Add performance characteristics if notable
        if performance_patterns:
            perf_notes = []
            for pattern in performance_patterns:
                if pattern.pattern_type == "nested_iteration":
                    perf_notes.append("uses nested loops")
                elif pattern.pattern_type == "recursive":
                    perf_notes.append("uses recursion")
                elif pattern.pattern_type == "caching":
                    perf_notes.append("implements caching")

            if perf_notes:
                description_parts.append(f"Performance note: {', '.join(perf_notes)}")

        # Combine parts into coherent description
        if len(description_parts) == 1:
            return description_parts[0]
        elif len(description_parts) > 1:
            # Join with appropriate connectors
            main_desc = description_parts[0]
            additional = ". ".join(description_parts[1:])
            return f"{main_desc}. {additional}"

        return ""

    def _generate_basic_purpose(self, function_name: str) -> str:
        """Generate basic purpose description from function name."""
        # Remove common prefixes/suffixes
        name = function_name.lower()
        name = re.sub(
            r"^(get_|set_|create_|build_|calculate_|compute_)",
            "",
            name,
        )
        name = re.sub(r"(_function|_method)$", "", name)

        # Convert snake_case to words
        words = name.replace("_", " ")

        return f"Handles {words} operations"

    def _update_description_in_docstring(
        self, context: SuggestionContext, new_description: str
    ) -> str:
        """Update description in existing docstring."""
        docstring = context.docstring
        style = self._detect_style(docstring)
        from ..models import DocstringStyle

        template = get_template(
            DocstringStyle(style), max_line_length=self.config.max_line_length
        )

        # Determine if this should be summary or description
        current_summary = getattr(docstring, "summary", "") if docstring else ""

        # If current summary is short or missing, use as summary
        if not current_summary or len(current_summary.split()) < 5:
            updated_summary = new_description
            updated_description = (
                getattr(docstring, "description", None) if docstring else None
            )
        else:
            # Keep existing summary, update description
            updated_summary = current_summary
            updated_description = new_description

        return template.render_complete_docstring(
            summary=updated_summary,
            description=updated_description,
            parameters=getattr(docstring, "parameters", []) if docstring else [],
            returns=getattr(docstring, "returns", None) if docstring else None,
            raises=getattr(docstring, "raises", []) if docstring else [],
            examples=getattr(docstring, "examples", []) if docstring else [],
        )

    def _detect_style(self, docstring: Any | None) -> str:
        """Detect docstring style from parsed docstring."""
        if docstring is not None and hasattr(docstring, "format"):
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

        # Ensure we have some original text (required by Suggestion model)
        if not original_text:
            original_text = f'"""TODO: Document {context.function.signature.name}."""'

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
        original_text = ""
        if context.docstring:
            original_text = getattr(context.docstring, "raw_text", "")
        if not original_text:
            # Provide a minimal docstring placeholder
            original_text = f'"""TODO: Document {context.function.signature.name}."""'

        return self._create_suggestion(
            context,
            original_text,
            f"Unable to generate specific behavior improvement: {reason}",
            confidence=0.1,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

    def _generic_behavior_improvement(self, context: SuggestionContext) -> Suggestion:
        """Generic behavior improvement for unknown issues."""
        return self._create_fallback_suggestion(
            context, f"Unknown behavior issue type: {context.issue.issue_type}"
        )

    def _generate_rag_enhanced_behavior(
        self, context: SuggestionContext, focus_side_effects: bool = False
    ) -> str | None:
        """Generate behavior description using RAG examples."""
        if not context.related_functions:
            return None

        # Filter out examples with low similarity scores
        relevant_examples = [
            example
            for example in context.related_functions
            if example.get("similarity", 0) >= 0.3
        ]

        if not relevant_examples:
            logger.debug("No RAG examples with sufficient similarity (>= 0.3)")
            return None

        # Extract behavior patterns and vocabulary from examples
        vocabulary = self._extract_behavior_vocabulary(relevant_examples)
        descriptions = self._extract_behavior_descriptions(relevant_examples)

        if not descriptions and not vocabulary:
            return None

        # Synthesize new description
        return self._synthesize_behavior_description(
            context, vocabulary, descriptions, focus_side_effects
        )

    def _extract_behavior_vocabulary(
        self, examples: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        """Extract domain-specific vocabulary from examples."""
        vocabulary: dict[str, list[str]] = {
            "verbs": [],  # Action words: processes, validates, transforms
            "patterns": [],  # Common phrases: "based on", "according to"
            "technical": [],  # Domain terms: API, cache, configuration
        }

        for example in examples:
            docstring = example.get("docstring", "")
            if not docstring:
                continue

            # Extract and categorize vocabulary
            self._extract_action_verbs(docstring, vocabulary["verbs"])
            self._extract_common_patterns(docstring, vocabulary["patterns"])
            vocabulary["technical"].extend(self._extract_technical_terms(docstring))

        # Deduplicate and rank by frequency
        return self._rank_vocabulary_by_frequency(vocabulary)

    def _extract_action_verbs(self, description: str, verb_list: list[str]) -> None:
        """Extract action verbs from description."""
        # Common action verbs in documentation (both singular and plural forms)
        action_verbs = {
            "process",
            "processes",
            "validate",
            "validates",
            "transform",
            "transforms",
            "convert",
            "converts",
            "parse",
            "parses",
            "fetch",
            "fetches",
            "retrieve",
            "retrieves",
            "download",
            "downloads",
            "upload",
            "uploads",
            "send",
            "sends",
            "calculate",
            "calculates",
            "compute",
            "computes",
            "generate",
            "generates",
            "create",
            "creates",
            "build",
            "builds",
            "update",
            "updates",
            "modify",
            "modifies",
            "delete",
            "deletes",
            "remove",
            "removes",
            "clear",
            "clears",
            "check",
            "checks",
            "verify",
            "verifies",
            "ensure",
            "ensures",
            "confirm",
            "confirms",
            "test",
            "tests",
            "initialize",
            "initializes",
            "configure",
            "configures",
            "set",
            "sets",
            "reset",
            "resets",
            "load",
            "loads",
            "save",
            "saves",
            "store",
            "stores",
            "cache",
            "caches",
            "persist",
            "persists",
            "write",
            "writes",
            "read",
            "reads",
            "query",
            "queries",
            "search",
            "searches",
            "find",
            "finds",
            "locate",
            "locates",
            "handle",
            "handles",
            "manage",
            "manages",
            "control",
            "controls",
            "coordinate",
            "coordinates",
            "orchestrate",
            "orchestrates",
            "filter",
            "filters",
            "sort",
            "sorts",
            "group",
            "groups",
            "aggregate",
            "aggregates",
            "merge",
            "merges",
            "encrypt",
            "encrypts",
            "decrypt",
            "decrypts",
            "hash",
            "hashes",
            "sign",
            "signs",
            "authenticate",
            "authenticates",
            "format",
            "formats",
            "render",
            "renders",
            "display",
            "displays",
            "present",
            "presents",
            "output",
            "outputs",
        }

        # Split into words and check each word
        words = description.lower().split()
        for word in words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:")
            if clean_word in action_verbs:
                # Normalize to plural form for consistency
                if clean_word.endswith("s"):
                    verb_list.append(clean_word)
                else:
                    verb_list.append(clean_word + "s")

        # Also check for verbs after "This function/method"
        patterns = [
            r"(?:function|method)\s+(\w+)\s",  # "This function validates..."
            r"(?:used to|will)\s+(\w+)\s",  # "Used to parse..."
            r"(?:responsible for)\s+(\w+ing)\s",  # "Responsible for handling..."
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                if match.lower() in action_verbs:
                    if match.lower().endswith("s"):
                        verb_list.append(match.lower())
                    else:
                        verb_list.append(match.lower() + "s")

    def _extract_common_patterns(
        self, description: str, pattern_list: list[str]
    ) -> None:
        """Extract common phrase patterns."""
        # Common documentation patterns
        common_patterns = [
            r"based on\s+(?:the\s+)?(\w+)",
            r"according to\s+(?:the\s+)?(\w+)",
            r"(?:with|using)\s+(?:the\s+)?(?:given|provided|specified)\s+(\w+)",
            r"if\s+(?:the\s+)?(\w+)\s+is\s+(?:not\s+)?(?:None|empty|valid)",
            r"returns?\s+(?:a\s+)?(?:new\s+)?(\w+)",
            r"raises?\s+(\w+)\s+if",
            r"for\s+(?:each|every|all)\s+(\w+)",
            r"from\s+(?:the\s+)?(?:given|provided)\s+(\w+)",
        ]

        for pattern in common_patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                pattern_list.append(match.group(0).lower())

    def _extract_technical_terms(self, description: str) -> list[str]:
        """Extract domain-specific technical terms."""
        technical_terms = []

        # Common technical term patterns
        term_patterns = [
            r"\b(?:API|REST|HTTP|URL|URI|JSON|XML|CSV)\b",
            r"\b(?:database|cache|queue|stream|buffer)\b",
            r"\b(?:async|sync|concurrent|parallel|thread)\b",
            r"\b(?:configuration|settings|options|parameters)\b",
            r"\b(?:authentication|authorization|token|credential)\b",
            r"\b(?:encryption|decryption|hash|signature)\b",
            r"\b(?:client|server|connection|socket|protocol)\b",
            r"\b(?:query|filter|sort|paginate|aggregate)\b",
        ]

        for pattern in term_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            technical_terms.extend(matches)

        # Also extract CamelCase terms (likely class/type names)
        camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", description)
        technical_terms.extend(list(camel_case))

        return technical_terms

    def _rank_vocabulary_by_frequency(
        self, vocabulary: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Rank vocabulary items by frequency."""

        ranked = {}
        for category, items in vocabulary.items():
            # Count occurrences
            counts = Counter(items)
            # Sort by frequency, then alphabetically
            ranked[category] = [
                item for item, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            ][
                :10
            ]  # Keep top 10

        return ranked

    def _extract_behavior_descriptions(
        self, examples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract behavior descriptions from examples."""
        descriptions = []

        for example in examples:
            docstring = example.get("docstring", "")
            if not docstring:
                continue

            # Extract summary and description
            parts = docstring.strip().split("\n\n")
            if parts:
                summary = parts[0].strip()
                if summary:
                    descriptions.append(
                        {
                            "summary": summary,
                            "similarity": example.get("similarity", 0.0),
                            "signature": example.get("signature", ""),
                        }
                    )

        # Sort by similarity
        descriptions.sort(key=lambda x: x["similarity"], reverse=True)
        return descriptions[:5]  # Keep top 5

    def _synthesize_behavior_description(
        self,
        context: SuggestionContext,
        vocabulary: dict[str, list[str]],
        descriptions: list[dict[str, Any]],
        focus_side_effects: bool,
    ) -> str | None:
        """Synthesize a behavior description from vocabulary and examples."""
        # Use sophisticated vocabulary-based generation
        if vocabulary and any(vocabulary.values()):
            return self._generate_behavior_from_vocabulary(
                context.function, vocabulary, descriptions
            )

        # Fallback to existing simple logic if no vocabulary
        function_name = getattr(context.function.signature, "name", "")

        # Start with the most relevant description as base
        if descriptions:
            base_description = descriptions[0]["summary"]
            # Adapt it to current context
            adapted = self._adapt_description_to_context(
                base_description, function_name, vocabulary
            )
            return adapted

        # If no good descriptions, build from vocabulary
        if vocabulary.get("verbs"):
            verb = vocabulary["verbs"][0]
            if vocabulary.get("patterns"):
                pattern = vocabulary["patterns"][0]
                return f"{verb.capitalize()} data {pattern}"
            else:
                return f"{verb.capitalize()} the input data"

        # Fallback
        return None

    def _adapt_description_to_context(
        self, description: str, function_name: str, vocabulary: dict[str, list[str]]
    ) -> str:
        """Adapt a description to the current context."""
        # Simple adaptation - replace generic terms with specific ones
        adapted = description

        # If description is very generic, try to make it more specific
        if len(adapted.split()) < 5 and vocabulary["verbs"]:
            # Enhance with vocabulary
            verb = vocabulary["verbs"][0]
            if verb not in adapted.lower():
                adapted = f"{verb.capitalize()} {adapted.lower()}"

        # Ensure proper capitalization
        if adapted and not adapted[0].isupper():
            adapted = adapted[0].upper() + adapted[1:]

        return adapted

    def _create_behavior_suggestion(
        self, context: SuggestionContext, description: str
    ) -> Suggestion:
        """Create a suggestion with the behavior description."""
        # Update the docstring with new description
        updated_docstring = self._update_description_in_docstring(context, description)

        return self._create_suggestion(
            context,
            updated_docstring,
            "Improve function description using RAG examples",
            confidence=0.85,
            suggestion_type=SuggestionType.DESCRIPTION_UPDATE,
        )

    def _generate_behavior_from_vocabulary(
        self,
        function: Any,
        vocabulary: dict[str, list[str]],
        description_structures: list[dict[str, Any]],
    ) -> str:
        """Generate behavior description using learned vocabulary."""
        # Select appropriate verb
        func_name_tokens = self._tokenize_function_name(function.signature.name)

        # Find matching verb
        selected_verb = None
        for verb in vocabulary.get("verbs", []):
            if any(token.lower() in verb for token in func_name_tokens):
                selected_verb = verb
                break

        if not selected_verb and vocabulary.get("verbs"):
            selected_verb = vocabulary["verbs"][0]  # Use most common

        if not selected_verb:
            selected_verb = "processes"  # Fallback

        # Build description
        description_parts = []

        # Start with verb
        description_parts.append(selected_verb.capitalize())

        # Add object based on parameters
        if function.signature.parameters:
            first_param = function.signature.parameters[0]
            param_desc = self._describe_parameter_for_behavior(first_param)
            description_parts.append(param_desc)

        # Add patterns if found
        if vocabulary.get("patterns"):
            # Use appropriate pattern
            for pattern in vocabulary["patterns"]:
                if "based on" in pattern or "using" in pattern:
                    description_parts.append(pattern)
                    break

        # Add return information
        if function.signature.return_type:
            return_phrase = self._generate_return_phrase_for_behavior(
                function.signature.return_type, vocabulary.get("patterns", [])
            )
            if return_phrase:
                description_parts.append(f"and {return_phrase}")

        # Join parts
        description = " ".join(description_parts)

        # Add technical context if available
        tech_terms = vocabulary.get("technical", [])
        if tech_terms:
            relevant_terms = self._find_relevant_technical_terms(function, tech_terms)
            if relevant_terms:
                description += f". Works with {', '.join(relevant_terms[:2])}"

        return self._ensure_behavior_grammar(description)

    def _tokenize_function_name(self, name: str) -> list[str]:
        """Tokenize function name into words."""
        # Handle snake_case
        tokens = name.split("_")

        # Handle camelCase
        result = []
        for token in tokens:
            # Split on capital letters
            subtokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", token)
            if subtokens:
                result.extend(subtokens)
            else:
                result.append(token)

        return [t.lower() for t in result if t]

    def _describe_parameter_for_behavior(self, param: Any) -> str:
        """Generate behavior description for a parameter."""
        param_name = param.name.replace("_", " ")
        param_name_lower = param_name.lower()

        if param.type_annotation:
            type_str = str(param.type_annotation).lower()
            # Check for container types first (they take precedence)
            if type_str == "list" or type_str.startswith("list["):
                if "list" not in param_name_lower:
                    return f"the {param_name} list"
            elif type_str == "dict" or type_str.startswith("dict["):
                if all(
                    word not in param_name_lower for word in ["dict", "mapping", "map"]
                ):
                    return f"the {param_name} mapping"
            # Only check for base types if not a container type
            elif type_str == "str" and "string" not in param_name_lower:
                return f"the {param_name} string"

            return f"the {param_name}"
        else:
            return f"the given {param_name}"

    def _generate_return_phrase_for_behavior(
        self, return_type: str, patterns: list[str]
    ) -> str | None:
        """Generate return phrase for behavior description."""
        if not return_type or return_type == "None":
            return None

        type_lower = return_type.lower()

        # Check if any patterns already mention returns
        for pattern in patterns:
            if "return" in pattern:
                return None  # Don't duplicate

        if "bool" in type_lower:
            return "returns True if successful"
        elif "list" in type_lower:
            return "returns the resulting list"
        elif "dict" in type_lower:
            return "returns the results as a dictionary"
        elif "str" in type_lower:
            return "returns the processed string"
        elif "int" in type_lower or "float" in type_lower:
            return "returns the calculated value"
        else:
            return f"returns the {return_type.split('[')[0].split('.')[-1].lower()}"

    def _find_relevant_technical_terms(
        self, function: Any, tech_terms: list[str]
    ) -> list[str]:
        """Find technical terms relevant to the function."""
        relevant = []

        # Check function name and parameters
        func_context = function.signature.name.lower()
        for param in function.signature.parameters:
            func_context += " " + param.name.lower()
            if param.type_annotation:
                func_context += " " + str(param.type_annotation).lower()

        for term in tech_terms:
            if term.lower() in func_context:
                relevant.append(term)

        return relevant

    def _ensure_behavior_grammar(self, description: str) -> str:
        """Ensure proper grammar for behavior descriptions."""
        # Capitalize first letter
        if description:
            description = description[0].upper() + description[1:]

        # Ensure ends with period
        if not description.endswith("."):
            description += "."

        # Fix grammar issues
        description = re.sub(r"\s+", " ", description)
        description = re.sub(r"\s+([.,])", r"\1", description)
        description = re.sub(r"\band\s+and\b", "and", description)

        return description.strip()
