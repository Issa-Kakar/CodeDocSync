import pytest
from codedocsync.matcher.semantic_scorer import SemanticScorer
from codedocsync.parser import (
    ParsedFunction,
    FunctionSignature,
    FunctionParameter,
    RawDocstring,
)
from codedocsync.matcher.models import MatchConfidence


class TestSemanticScorer:
    """Test suite for SemanticScorer validation and scoring."""

    @pytest.fixture
    def scorer(self):
        """Create SemanticScorer instance."""
        return SemanticScorer()

    @pytest.fixture
    def sample_function(self):
        """Create sample function for testing."""
        return ParsedFunction(
            signature=FunctionSignature(
                name="get_user_data",
                parameters=[
                    FunctionParameter(
                        name="user_id",
                        type_annotation="str",
                        default_value=None,
                        is_required=True,
                    )
                ],
                return_type="Dict[str, Any]",
            ),
            docstring=RawDocstring(
                raw_text="Get user data from database", line_number=2
            ),
            file_path="src/utils/user.py",
            line_number=1,
            end_line_number=5,
            source_code="def get_user_data(user_id: str): pass",
        )

    def test_validate_semantic_match_high_similarity(self, scorer, sample_function):
        """Test validation with high similarity score."""
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function, "src.utils.fetch_user_data", 0.92
        )

        assert is_valid == True
        assert adjusted_score >= 0.92  # Should be at least the original score

    def test_validate_semantic_match_low_similarity(self, scorer, sample_function):
        """Test validation with low similarity score."""
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function, "src.other.completely_different_function", 0.45
        )

        assert is_valid == False
        assert adjusted_score < 0.65  # Below threshold

    def test_validate_semantic_match_pattern_boost(self, scorer, sample_function):
        """Test that matching patterns boost the score."""
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function,
            "src.utils.fetch_user_data",  # get_user_data -> fetch_user_data pattern
            0.75,
        )

        assert is_valid == True
        assert adjusted_score > 0.75  # Should be boosted

    def test_validate_semantic_match_name_penalty(self, scorer, sample_function):
        """Test that very different names get penalized."""
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function,
            "src.utils.xyz_random_name",  # Very different name
            0.75,
        )

        # Should be penalized for name dissimilarity
        assert adjusted_score < 0.75

    def test_validate_semantic_match_module_distance_penalty(
        self, scorer, sample_function
    ):
        """Test that distant modules get penalized."""
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function,
            "very.distant.deep.module.structure.get_user_data",  # Distant module
            0.75,
        )

        # Should be penalized for module distance
        assert adjusted_score < 0.75

    def test_calculate_semantic_confidence_high_score(self, scorer, sample_function):
        """Test confidence calculation for high similarity score."""
        confidence = scorer.calculate_semantic_confidence(0.90, sample_function)

        assert isinstance(confidence, MatchConfidence)
        assert confidence.overall == 0.9
        assert confidence.name_similarity == 0.90
        assert confidence.location_score == 0.5  # Standard for semantic matches
        assert confidence.signature_similarity == 0.7  # Assumed for semantic matches

    def test_calculate_semantic_confidence_medium_score(self, scorer, sample_function):
        """Test confidence calculation for medium similarity score."""
        confidence = scorer.calculate_semantic_confidence(0.78, sample_function)

        assert confidence.overall == 0.75
        assert confidence.name_similarity == 0.78

    def test_calculate_semantic_confidence_low_score(self, scorer, sample_function):
        """Test confidence calculation for low similarity score."""
        confidence = scorer.calculate_semantic_confidence(0.67, sample_function)

        assert confidence.overall == 0.65
        assert confidence.name_similarity == 0.67

    def test_names_follow_pattern_verb_changes(self, scorer):
        """Test recognition of verb change patterns."""
        assert scorer._names_follow_pattern("get_user", "fetch_user") == True
        assert scorer._names_follow_pattern("set_value", "update_value") == True
        assert scorer._names_follow_pattern("create_record", "make_record") == True
        assert scorer._names_follow_pattern("delete_item", "remove_item") == True

    def test_names_follow_pattern_noun_changes(self, scorer):
        """Test recognition of noun change patterns."""
        assert scorer._names_follow_pattern("user_id", "user_identifier") == True
        assert scorer._names_follow_pattern("data_dict", "data_map") == True
        assert scorer._names_follow_pattern("item_list", "item_array") == True

    def test_names_follow_pattern_case_changes(self, scorer):
        """Test recognition of case style changes."""
        assert (
            scorer._is_camelcase_snake_case_pair("getUserData", "get_user_data") == True
        )
        assert (
            scorer._is_camelcase_snake_case_pair(
                "updateUserProfile", "update_user_profile"
            )
            == True
        )
        assert (
            scorer._is_camelcase_snake_case_pair("get_user_data", "getUserData") == True
        )

    def test_names_follow_pattern_no_match(self, scorer):
        """Test that unrelated names don't match patterns."""
        assert scorer._names_follow_pattern("get_user", "calculate_tax") == False
        assert scorer._names_follow_pattern("process_data", "send_email") == False

    def test_calculate_name_similarity_exact_match(self, scorer):
        """Test name similarity calculation for exact matches."""
        similarity = scorer._calculate_name_similarity("get_user", "get_user")
        assert similarity == 1.0

    def test_calculate_name_similarity_containment(self, scorer):
        """Test name similarity for name containment."""
        similarity = scorer._calculate_name_similarity("get_user", "get_user_data")
        assert similarity == 0.8

        similarity = scorer._calculate_name_similarity("user", "get_user")
        assert similarity == 0.8

    def test_calculate_name_similarity_character_overlap(self, scorer):
        """Test name similarity using character overlap."""
        similarity = scorer._calculate_name_similarity("user_data", "user_info")
        assert 0.5 < similarity < 1.0  # Should have decent overlap

    def test_calculate_name_similarity_no_overlap(self, scorer):
        """Test name similarity with no overlap."""
        similarity = scorer._calculate_name_similarity("abc", "xyz")
        assert similarity == 0.0

    def test_calculate_module_distance_same_module(self, scorer):
        """Test module distance calculation for same module."""
        distance = scorer._calculate_module_distance(
            "src.utils.helpers", "src.utils.helpers"
        )
        assert distance == 0

    def test_calculate_module_distance_sibling_modules(self, scorer):
        """Test module distance for sibling modules."""
        distance = scorer._calculate_module_distance(
            "src.utils.helpers", "src.utils.database"
        )
        assert distance == 2  # Both need to go up one level

    def test_calculate_module_distance_parent_child(self, scorer):
        """Test module distance for parent-child relationship."""
        distance = scorer._calculate_module_distance("src.utils", "src.utils.helpers")
        assert distance == 1  # Child is one level down

    def test_calculate_module_distance_distant_modules(self, scorer):
        """Test module distance for distant modules."""
        distance = scorer._calculate_module_distance(
            "src.utils.helpers", "tests.integration.test_api"
        )
        assert distance > 2  # Should be considered distant

    def test_assess_match_quality_high_quality(self, scorer, sample_function):
        """Test match quality assessment for high quality match."""
        assessment = scorer.assess_match_quality(
            sample_function, 0.92, {"module_distance": 1, "name_similarity": 0.8}
        )

        assert assessment["confidence_level"] == "high"
        assert assessment["similarity_score"] == 0.92
        assert "Very high semantic similarity" in assessment["quality_factors"]
        assert len(assessment["concerns"]) == 0

    def test_assess_match_quality_medium_quality(self, scorer, sample_function):
        """Test match quality assessment for medium quality match."""
        assessment = scorer.assess_match_quality(
            sample_function, 0.78, {"module_distance": 2, "name_similarity": 0.6}
        )

        assert assessment["confidence_level"] == "medium"
        assert "High semantic similarity" in assessment["quality_factors"]

    def test_assess_match_quality_low_quality(self, scorer, sample_function):
        """Test match quality assessment for low quality match."""
        assessment = scorer.assess_match_quality(
            sample_function, 0.68, {"module_distance": 3, "name_similarity": 0.2}
        )

        assert assessment["confidence_level"] == "low"
        assert "Functions in distant modules" in assessment["concerns"]
        assert "Very different function names" in assessment["concerns"]

    def test_assess_match_quality_insufficient(self, scorer, sample_function):
        """Test match quality assessment for insufficient match."""
        assessment = scorer.assess_match_quality(sample_function, 0.55)

        assert assessment["confidence_level"] == "insufficient"
        assert "Below preferred similarity threshold" in assessment["concerns"]

    def test_threshold_constants(self, scorer):
        """Test that scorer has correct threshold constants."""
        assert scorer.THRESHOLDS["high_confidence"] == 0.85
        assert scorer.THRESHOLDS["medium_confidence"] == 0.75
        assert scorer.THRESHOLDS["low_confidence"] == 0.65
        assert scorer.THRESHOLDS["no_match"] == 0.65

    def test_validation_edge_cases(self, scorer, sample_function):
        """Test validation with edge cases."""
        # Test with empty candidate function ID
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function, "", 0.80
        )
        # Should handle gracefully
        assert isinstance(is_valid, bool)
        assert isinstance(adjusted_score, float)

        # Test with very long module path
        is_valid, adjusted_score = scorer.validate_semantic_match(
            sample_function,
            "very.very.very.long.module.path.with.many.levels.function_name",
            0.80,
        )
        # Should penalize for distance
        assert adjusted_score < 0.80

    def test_confidence_validation_context(self, scorer, sample_function):
        """Test confidence calculation with validation context."""
        context = {
            "module_distance": 1,
            "name_similarity": 0.9,
            "parameter_match": True,
        }

        confidence = scorer.calculate_semantic_confidence(
            0.85, sample_function, validation_context=context
        )

        assert isinstance(confidence, MatchConfidence)
        assert confidence.overall == 0.9  # High confidence

    def test_camelcase_snake_case_edge_cases(self, scorer):
        """Test camelCase/snake_case conversion edge cases."""
        # Test single word
        assert scorer._is_camelcase_snake_case_pair("user", "user") == False

        # Test already matching format
        assert scorer._is_camelcase_snake_case_pair("get_user", "get_user") == False

        # Test with numbers
        assert (
            scorer._is_camelcase_snake_case_pair("getUserData2", "get_user_data2")
            == True
        )

        # Test with acronyms
        assert (
            scorer._is_camelcase_snake_case_pair("parseHTMLData", "parse_html_data")
            == True
        )

    def test_name_similarity_empty_strings(self, scorer):
        """Test name similarity with empty strings."""
        assert scorer._calculate_name_similarity("", "") == 1.0
        assert scorer._calculate_name_similarity("test", "") == 0.0
        assert scorer._calculate_name_similarity("", "test") == 0.0

    def test_module_distance_edge_cases(self, scorer):
        """Test module distance calculation edge cases."""
        # Test with empty modules
        distance = scorer._calculate_module_distance("", "")
        assert distance == 0

        # Test with single level modules
        distance = scorer._calculate_module_distance("module1", "module2")
        assert distance == 2
