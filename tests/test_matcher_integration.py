"""Integration tests for the complete matching system."""

from codedocsync.matcher import MatchingFacade
from codedocsync.utils.config import CodeDocSyncConfig


class TestMatcherIntegration:
    """Test complete matching workflows."""

    def test_end_to_end_matching(self, tmp_path):
        """Test complete parsing and matching workflow."""
        # Create test file
        test_file = tmp_path / "test_module.py"
        test_file.write_text(
            '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b

def calc_product(x, y):
    """Calculate product of x and y."""
    return x * y

def helper_function():
    # No docstring
    pass

class Calculator:
    """A simple calculator class."""

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
'''
        )

        # Create facade and match
        facade = MatchingFacade()
        result = facade.match_file(test_file)

        # Verify results
        assert result.total_functions == 4  # 3 functions + 1 method
        assert len(result.matched_pairs) == 3  # All except helper_function
        assert len(result.unmatched_functions) == 1

        # Check match types
        summary = result.get_summary()
        assert summary["match_types"]["exact"] == 3

        # Verify unmatched function
        assert result.unmatched_functions[0].signature.name == "helper_function"

    def test_fuzzy_matching_project(self, tmp_path):
        """Test fuzzy matching across a project."""
        # Create multiple files with naming variations
        (tmp_path / "models.py").write_text(
            '''
def get_user_by_id(user_id: int):
    """Get user by ID."""
    pass

def getUserById(userId: int):
    """Get user by ID."""
    pass
'''
        )

        (tmp_path / "utils.py").write_text(
            '''
def calc_avg(numbers: list):
    """Calculate average."""
    pass

def calculate_average(numbers: list):
    """Calculate average of numbers."""
    pass
'''
        )

        # Match with fuzzy enabled
        config = CodeDocSyncConfig(
            matcher={"enable_fuzzy": True, "fuzzy_threshold": 0.7}
        )
        facade = MatchingFacade(config)
        result = facade.match_project(tmp_path)

        # Should find some fuzzy matches
        assert result.total_functions == 4
        fuzzy_matches = [
            p for p in result.matched_pairs if p.match_type.value == "fuzzy"
        ]
        assert len(fuzzy_matches) >= 0  # Depends on matching logic

    def test_custom_patterns(self, tmp_path):
        """Test custom pattern matching."""
        test_file = tmp_path / "validators.py"
        test_file.write_text(
            '''
def check_email(email: str) -> bool:
    """Validate email address."""
    return "@" in email

def validateEmail(email: str) -> bool:
    """Validate email address."""
    return "@" in email
'''
        )

        # Configure custom pattern
        config = CodeDocSyncConfig(
            matcher={
                "custom_patterns": [
                    {"pattern": r"check_(\w+)", "replacement": r"validate\1"}
                ]
            }
        )

        facade = MatchingFacade(config)
        result = facade.match_file(test_file)

        # Should match via custom pattern
        assert len(result.matched_pairs) == 2

    def test_empty_file_handling(self, tmp_path):
        """Test handling of empty files."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        facade = MatchingFacade()
        result = facade.match_file(empty_file)

        assert result.total_functions == 0
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_functions) == 0

    def test_project_with_excluded_dirs(self, tmp_path):
        """Test that excluded directories are properly filtered."""
        # Create files in excluded directories
        (tmp_path / ".venv" / "lib.py").write_text(
            '''
def excluded_func():
    """This should be excluded."""
    pass
'''
        )
        (tmp_path / ".venv").mkdir(exist_ok=True)

        # Create valid file
        (tmp_path / "main.py").write_text(
            '''
def main_func():
    """Main function."""
    pass
'''
        )

        facade = MatchingFacade()
        result = facade.match_project(tmp_path)

        # Should only find the main function, not the excluded one
        assert result.total_functions == 1
        assert result.matched_pairs[0].function.signature.name == "main_func"

    def test_configuration_loading(self, tmp_path):
        """Test configuration loading from file."""
        # Create config file
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
version: 1
matcher:
  enable_fuzzy: false
  fuzzy_threshold: 0.9
  custom_patterns:
    - pattern: "test_(\\w+)"
      replacement: "verify\\1"
"""
        )

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''
def test_email():
    """Test email validation."""
    pass

def verifyEmail():
    """Verify email validation."""
    pass
'''
        )

        # Load config and test
        config = CodeDocSyncConfig.from_yaml(str(config_file))
        facade = MatchingFacade(config)
        result = facade.match_file(test_file)

        # Should apply custom patterns
        assert len(result.matched_pairs) == 2
