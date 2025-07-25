"""Module with correct documentation."""


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: First integer to add
        b: Second integer to add

    Returns:
        The sum of a and b
    """
    return a + b


def format_user_name(first_name: str, last_name: str) -> str:
    """Format a user's full name.

    Args:
        first_name: User's first name
        last_name: User's last name

    Returns:
        Formatted full name as "Last, First"
    """
    return f"{last_name}, {first_name}"


def validate_email(email: str) -> bool:
    """Check if an email address is valid.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    return "@" in email and "." in email.split("@")[1]
