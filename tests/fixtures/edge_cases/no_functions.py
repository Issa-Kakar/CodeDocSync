"""Module with no functions, only imports and variables."""

from pathlib import Path

# Module-level constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
API_VERSION = "2.0"

# Configuration dictionary
CONFIG = {
    "debug": False,
    "timeout": DEFAULT_TIMEOUT,
    "retries": MAX_RETRIES,
    "endpoints": ["https://api.example.com/v1", "https://api.example.com/v2"],
}

# Type aliases
PathLike = str | Path
ConfigDict = dict[str, any]

# Module variable
_internal_state = None


# Just a class, no functions
class ConfigManager:
    """Manages configuration settings."""

    def __init__(self):
        self.settings = CONFIG.copy()
