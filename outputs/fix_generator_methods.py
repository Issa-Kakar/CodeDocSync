import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# First, let's see what the base class expects
import inspect

from codedocsync.suggestions.base import BaseSuggestionGenerator

print("BaseSuggestionGenerator methods:")
for name, method in inspect.getmembers(
    BaseSuggestionGenerator, predicate=inspect.isfunction
):
    if not name.startswith("_"):
        try:
            sig = inspect.signature(method)
            print(f"  {name}: {sig}")
        except Exception:
            print(f"  {name}: (signature unavailable)")

# Check if we need to add generate_suggestion as alias
