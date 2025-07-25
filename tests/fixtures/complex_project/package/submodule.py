"""Data processing submodule."""

from typing import Any

from .utils import clean_data


class DataProcessor:
    """Process various types of data with configurable options."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize processor with optional configuration.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {}
        self._cache: dict[str, Any] = {}

    def process(self, raw_data: str) -> dict[str, Any]:
        """Process raw data into structured format.

        Args:
            raw_data: Raw input data as string

        Returns:
            Processed data dictionary
        """
        # Clean the data first
        cleaned = clean_data(raw_data)

        # Apply processing based on config
        result = {
            "data": cleaned,
            "metadata": {
                "processed_at": "2025-07-24",
                "version": self.config.get("version", "1.0"),
                "length": len(cleaned),
            },
        }

        # Cache result
        self._cache[raw_data[:10]] = result
        return result

    def process_with_validation(self, data: str, strict: bool = True) -> dict:
        """Process data with optional validation.

        Args:
            data: Input data
            validate: Whether to validate  # WRONG: parameter is 'strict' not 'validate'

        Returns:
            dict: Processed result  # Redundant type in docstring
        """
        if strict and not data.strip():
            raise ValueError("Empty data not allowed in strict mode")
        return self.process(data)

    def get_cache_size(self) -> int:
        """Get current cache size.

        Returns:
            Number of cached items
        """
        return len(self._cache)
