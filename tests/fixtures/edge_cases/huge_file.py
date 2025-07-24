"""Large file for performance testing."""

# Generate many similar functions programmatically
# This simulates a large auto-generated or legacy codebase


def process_item_001(item: dict) -> dict:
    """Process item 001.

    Args:
        item: Item to process

    Returns:
        Processed item
    """
    return {"id": 1, "data": item}


def process_item_002(item: dict) -> dict:
    """Process item 002.

    Args:
        item: Item to process

    Returns:
        Processed item
    """
    return {"id": 2, "data": item}


def process_item_003(item: dict) -> dict:
    """Process item 003.

    Args:
        item: Item to process

    Returns:
        Processed item
    """
    return {"id": 3, "data": item}


def process_item_004(item: dict) -> dict:
    """Process item 004.

    Args:
        item: Item to process

    Returns:
        Processed item
    """
    return {"id": 4, "data": item}


def process_item_005(item: dict) -> dict:
    """Process item 005.

    Args:
        item: Item to process

    Returns:
        Processed item
    """
    return {"id": 5, "data": item}


# ... imagine this continues for hundreds of functions
# This is enough to test performance without creating an actually huge file


def validate_item_001(item: dict) -> bool:
    """Validate item 001.

    Args:
        data: Item to validate  # WRONG: parameter is 'item' not 'data'

    Returns:
        bool: True if valid
    """
    return "id" in item


def validate_item_002(item: dict) -> bool:
    """Validate item 002.

    Args:
        item: Item to validate

    Returns:
        True if valid
    """
    return "id" in item and item["id"] > 0


def transform_item_001(item: dict, options: dict | None = None) -> dict:
    """Transform item with options.

    Args:
        item: Item to transform
        # MISSING: options parameter not documented

    Returns:
        Transformed item
    """
    result = item.copy()
    if options:
        result.update(options)
    return result
