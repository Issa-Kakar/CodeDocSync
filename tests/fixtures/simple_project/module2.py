"""Module with documentation issues."""


def renamed_parameter(user_id: int) -> str:
    """Get user by ID.

    Args:
        id: User ID  # WRONG: parameter renamed to user_id

    Returns:
        str: Username
    """
    return f"user_{user_id}"


def changed_return_type(data: dict) -> list:
    """Process data.

    Args:
        data: Input data

    Returns:
        dict: Processed data  # WRONG: now returns list
    """
    return list(data.values())


def missing_parameter_doc(name: str, age: int, city: str) -> dict:
    """Create user profile.

    Args:
        name: User's name
        age: User's age
        # MISSING: city parameter not documented

    Returns:
        User profile dictionary
    """
    return {"name": name, "age": age, "city": city}


def extra_parameter_doc(x: int, y: int) -> int:
    """Multiply two numbers.

    Args:
        x: First number
        y: Second number
        z: Third number  # WRONG: parameter doesn't exist

    Returns:
        Product of x and y
    """
    return x * y


def wrong_exception_doc(value: str) -> int:
    """Convert string to integer.

    Args:
        value: String to convert

    Returns:
        Integer value

    Raises:
        TypeError: If value is not a string  # WRONG: actually raises ValueError
    """
    return int(value)


def optional_parameter_mismatch(data: dict | None = None) -> list:
    """Process optional data.

    Args:
        data: Input data dictionary  # MISSING: doesn't mention it's optional

    Returns:
        Processed list
    """
    if data is None:
        return []
    return list(data.values())
