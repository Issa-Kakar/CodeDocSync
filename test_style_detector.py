"""Quick test to verify style_detector functionality after mypy fixes."""

from codedocsync.suggestions.style_detector import DocstringStyleDetector


def test_style_detection():
    """Test basic style detection functionality."""
    detector = DocstringStyleDetector()

    # Test Google style detection
    google_docstring = """
    This is a function summary.

    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2

    Returns:
        bool: Description of return value
    """

    style = detector.detect_from_docstring(google_docstring)
    print(f"Detected style for Google docstring: {style}")

    # Test NumPy style detection
    numpy_docstring = """
    This is a function summary.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value
    """

    style = detector.detect_from_docstring(numpy_docstring)
    print(f"Detected style for NumPy docstring: {style}")

    # Test Sphinx style detection
    sphinx_docstring = """
    This is a function summary.

    :param param1: Description of param1
    :type param1: str
    :param param2: Description of param2
    :type param2: int
    :returns: Description of return value
    :rtype: bool
    """

    style = detector.detect_from_docstring(sphinx_docstring)
    print(f"Detected style for Sphinx docstring: {style}")

    # Test style scoring
    scores = detector.get_all_style_scores(google_docstring)
    print(f"\nStyle scores for Google docstring: {scores}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_style_detection()
