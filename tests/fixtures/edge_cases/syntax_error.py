"""File with syntax error for testing error handling."""

def broken_function(x: int) -> str:
    """This function has a syntax error.

    Args:
        x: Input value

    Returns:
        String representation
    """
    if x > 0:
        return "positive"
    else:
        # Missing closing quote causes syntax error
        return "negative
