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
        assert len(result.matched_pairs) >= 1  # At least calculate_sum should match
        assert (
            len(result.unmatched_functions) >= 1
        )  # At least helper_function should be unmatched

        # Check that some functions matched
        summary = result.get_summary()
        assert summary["match_types"]["exact"] >= 1

        # Verify helper_function is unmatched (has no docstring)
        unmatched_names = {func.signature.name for func in result.unmatched_functions}
        assert "helper_function" in unmatched_names

    def test_project_matching(self, tmp_path):
        """Test matching across a project."""
        # Create multiple files with documented functions
        (tmp_path / "models.py").write_text(
            '''
def get_user_by_id(user_id: int):
    """Get user by ID.

    Args:
        user_id: The user identifier

    Returns:
        User object
    """
    pass

def create_user(name: str):
    """Create a new user."""
    pass
'''
        )

        (tmp_path / "utils.py").write_text(
            '''
def calculate_average(numbers: list):
    """Calculate average of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Average value
    """
    pass

def helper():
    # No docstring
    pass
'''
        )

        facade = MatchingFacade()
        result = facade.match_project(tmp_path)

        # Should find matches for documented functions
        assert result.total_functions == 4
        assert len(result.matched_pairs) >= 1  # At least some functions should match

        # Check that exact matches are found
        exact_matches = [
            p for p in result.matched_pairs if p.match_type.value == "exact"
        ]
        assert len(exact_matches) >= 1

    def test_basic_function_matching(self, tmp_path):
        """Test basic function-to-docstring matching."""
        test_file = tmp_path / "validators.py"
        test_file.write_text(
            '''
def check_email(email: str) -> bool:
    """Validate email address.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    return "@" in email

def process_data():
    """Process some data."""
    pass
'''
        )

        facade = MatchingFacade()
        result = facade.match_file(test_file)

        # Should match functions to their own docstrings
        assert result.total_functions == 2
        assert len(result.matched_pairs) >= 1  # At least one should match

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
        (tmp_path / ".venv").mkdir(exist_ok=True)
        (tmp_path / ".venv" / "lib.py").write_text(
            '''
def excluded_func():
    """This should be excluded."""
    pass
'''
        )

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
        """Test basic configuration loading."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''
def test_email():
    """Test email validation.

    Validates email format.
    """
    pass

def verify_password():
    """Verify password strength."""
    pass
'''
        )

        # Load config and test
        config = CodeDocSyncConfig()
        facade = MatchingFacade(config)
        result = facade.match_file(test_file)

        # Should work with basic config
        assert result.total_functions == 2
        assert len(result.matched_pairs) >= 1
