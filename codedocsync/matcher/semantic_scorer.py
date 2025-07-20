import logging
import re

from ..parser import ParsedFunction
from .models import MatchConfidence

logger = logging.getLogger(__name__)


class SemanticScorer:
    """
    Scores and validates semantic matches.

    Applies additional heuristics beyond raw similarity scores.
    Part 1 implementation focuses on validation and confidence calculation.
    """

    # Thresholds from architecture
    THRESHOLDS = {
        "high_confidence": 0.85,
        "medium_confidence": 0.75,
        "low_confidence": 0.65,
        "no_match": 0.65,
    }

    def validate_semantic_match(
        self,
        source_function: ParsedFunction,
        candidate_function_id: str,
        raw_similarity: float,
    ) -> tuple[bool, float]:
        """
        Validate and adjust semantic match score.

        Applies architectural rules for semantic matching validation:
        - Module distance checks
        - Naming pattern recognition
        - Parameter count validation (when available)

        Args:
            source_function: Function looking for a match
            candidate_function_id: ID of potential match function
            raw_similarity: Raw similarity score from embedding comparison

        Returns:
            (is_valid, adjusted_score) tuple
        """
        # Start with raw similarity
        adjusted_score = raw_similarity

        # Extract function name from ID
        candidate_name = candidate_function_id.split(".")[-1]
        source_name = source_function.signature.name

        # Check naming patterns
        if self._names_follow_pattern(source_name, candidate_name):
            # Boost score for common refactoring patterns
            adjusted_score += 0.05
            logger.debug(
                f"Boosting score for pattern match: {source_name} -> {candidate_name}"
            )
        else:
            # Penalize very different names
            name_similarity = self._calculate_name_similarity(
                source_name, candidate_name
            )
            if name_similarity < 0.3:
                adjusted_score -= 0.1
                logger.debug(
                    f"Penalizing score for name dissimilarity: {name_similarity:.2f}"
                )

        # Module distance check (from architecture rules)
        source_module = source_function.file_path.replace("/", ".").replace("\\", ".")
        if source_module.endswith(".py"):
            source_module = source_module[:-3]

        candidate_module = ".".join(candidate_function_id.split(".")[:-1])

        module_distance = self._calculate_module_distance(
            source_module, candidate_module
        )
        if module_distance > 2:
            # Penalize distant modules (architecture rule: >2 levels apart)
            adjusted_score -= 0.1
            logger.debug(f"Penalizing score for module distance: {module_distance}")

        # Ensure score stays in valid range
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Apply threshold
        is_valid = adjusted_score >= self.THRESHOLDS["no_match"]

        return is_valid, adjusted_score

    def calculate_semantic_confidence(
        self,
        similarity_score: float,
        source_function: ParsedFunction,
        validation_context: dict | None = None,
    ) -> MatchConfidence:
        """
        Calculate confidence scores for semantic match.

        Args:
            similarity_score: Adjusted similarity score after validation
            source_function: Source function being matched
            validation_context: Optional context from validation process

        Returns:
            MatchConfidence with appropriate scores for semantic matching
        """
        # Determine overall confidence based on thresholds
        if similarity_score >= self.THRESHOLDS["high_confidence"]:
            overall = 0.9  # High confidence but not perfect
        elif similarity_score >= self.THRESHOLDS["medium_confidence"]:
            overall = 0.75
        elif similarity_score >= self.THRESHOLDS["low_confidence"]:
            overall = 0.65
        else:
            overall = 0.5  # Minimum for a match

        # Semantic matches have high name similarity (by definition of embeddings)
        # but lower location score since they may be in different modules
        return MatchConfidence(
            overall=overall,
            name_similarity=similarity_score,  # Embedding similarity represents semantic name similarity
            location_score=0.5,  # Unknown/variable location relationship for semantic matches
            signature_similarity=0.7,  # Assumed similar based on embedding semantic understanding
        )

    def _names_follow_pattern(self, name1: str, name2: str) -> bool:
        """
        Check if names follow common refactoring patterns.

        Recognizes common patterns like:
        - Verb changes (get_user -> fetch_user)
        - Style changes (snake_case -> camelCase)
        - Noun variations (user_dict -> user_map)
        """
        # Convert to lowercase for pattern matching
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        patterns = [
            # Verb changes - (source_pattern, target_template)
            (r"^get_(.+)$", "fetch_{}"),
            (r"^set_(.+)$", "update_{}"),
            (r"^create_(.+)$", "make_{}"),
            (r"^delete_(.+)$", "remove_{}"),
            (r"^build_(.+)$", "construct_{}"),
            (r"^init_(.+)$", "initialize_{}"),
            # Noun changes
            (r"^(.+)_id$", "{}_identifier"),
            (r"^(.+)_dict$", "{}_map"),
            (r"^(.+)_list$", "{}_array"),
            (r"^(.+)_str$", "{}_string"),
            (r"^(.+)_num$", "{}_number"),
            # Common abbreviations
            (r"^(.+)_config$", "{}_configuration"),
            (r"^(.+)_info$", "{}_information"),
            (r"^(.+)_data$", "{}_dataset"),
            (r"^(.+)_proc$", "{}_process"),
        ]

        for pattern1, template2 in patterns:
            # Check name1 -> name2 direction
            match1 = re.match(pattern1, name1_lower)
            if match1:
                expected_name2 = template2.format(match1.group(1))
                if expected_name2 == name2_lower:
                    return True

            # Check name2 -> name1 direction
            match2 = re.match(pattern1, name2_lower)
            if match2:
                expected_name1 = template2.format(match2.group(1))
                if expected_name1 == name1_lower:
                    return True

        # Check for camelCase to snake_case conversion
        if self._is_camelcase_snake_case_pair(name1, name2):
            return True

        return False

    def _is_camelcase_snake_case_pair(self, name1: str, name2: str) -> bool:
        """Check if names are camelCase/snake_case variants of each other."""

        def camelcase_to_snake(name):
            # Convert camelCase to snake_case
            s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
            return s1.lower()

        def snake_to_camelcase(name):
            # Convert snake_case to camelCase
            components = name.split("_")
            return components[0] + "".join(x.title() for x in components[1:])

        # Check if converting one gives the other
        return (
            camelcase_to_snake(name1) == name2.lower()
            or snake_to_camelcase(name2) == name1
            or camelcase_to_snake(name2) == name1.lower()
            or snake_to_camelcase(name1) == name2
        )

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate simple name similarity using character overlap.

        Args:
            name1: First function name
            name2: Second function name

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase
        name1 = name1.lower()
        name2 = name2.lower()

        # Handle empty strings
        if not name1 and not name2:
            return 1.0
        if not name1 or not name2:
            return 0.0

        # Exact match
        if name1 == name2:
            return 1.0

        # Check if one contains the other (but both must be non-empty)
        if len(name1) > 0 and len(name2) > 0:
            if name1 in name2 or name2 in name1:
                return 0.8

        # Simple character overlap (Jaccard similarity)
        common_chars = set(name1) & set(name2)
        all_chars = set(name1) | set(name2)

        if all_chars:
            return len(common_chars) / len(all_chars)

        return 0.0

    def _calculate_module_distance(self, module1: str, module2: str) -> int:
        """
        Calculate distance between modules based on their paths.

        Args:
            module1: First module path (e.g., 'src.utils.helpers')
            module2: Second module path (e.g., 'src.core.main')

        Returns:
            Distance as number of levels between modules
        """
        parts1 = module1.split(".")
        parts2 = module2.split(".")

        # Find common prefix
        common_prefix_len = 0
        for i, (p1, p2) in enumerate(zip(parts1, parts2, strict=False)):
            if p1 == p2:
                common_prefix_len = i + 1
            else:
                break

        # Distance is how many levels apart they are
        distance = (len(parts1) - common_prefix_len) + (len(parts2) - common_prefix_len)

        return distance

    def assess_match_quality(
        self,
        source_function: ParsedFunction,
        similarity_score: float,
        additional_context: dict | None = None,
    ) -> dict:
        """
        Assess overall match quality with detailed breakdown.

        Args:
            source_function: Function being matched
            similarity_score: Semantic similarity score
            additional_context: Optional additional validation context

        Returns:
            Dictionary with detailed quality assessment
        """
        assessment = {
            "similarity_score": similarity_score,
            "confidence_level": "low",
            "quality_factors": [],
            "concerns": [],
        }

        # Determine confidence level
        if similarity_score >= self.THRESHOLDS["high_confidence"]:
            assessment["confidence_level"] = "high"
        elif similarity_score >= self.THRESHOLDS["medium_confidence"]:
            assessment["confidence_level"] = "medium"
        elif similarity_score >= self.THRESHOLDS["low_confidence"]:
            assessment["confidence_level"] = "low"
        else:
            assessment["confidence_level"] = "insufficient"

        # Add quality factors
        if similarity_score > 0.9:
            assessment["quality_factors"].append("Very high semantic similarity")
        elif similarity_score > 0.8:
            assessment["quality_factors"].append("High semantic similarity")
        elif similarity_score > 0.7:
            assessment["quality_factors"].append("Good semantic similarity")

        # Add concerns for low scores
        if similarity_score < 0.7:
            assessment["concerns"].append("Below preferred similarity threshold")

        if additional_context:
            if additional_context.get("module_distance", 0) > 2:
                assessment["concerns"].append("Functions in distant modules")
            if additional_context.get("name_similarity", 1.0) < 0.3:
                assessment["concerns"].append("Very different function names")

        return assessment
